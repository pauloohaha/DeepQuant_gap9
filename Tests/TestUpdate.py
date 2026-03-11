# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>


import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_cuda.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_cudnn.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_mps.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_mkldnn.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*experimental feature.*"
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import brevitas.nn as qnn
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.graph.calibrate import calibration_mode
from brevitas.fx.brevitas_tracer import _symbolic_trace
from DeepQuant.CustomTracer import CustomBrevitasTracer
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)

from DeepQuant.ExportBrevitas import exportBrevitas
import numpy as np
from devo_onnx.net import UpdateONNX
from devo_onnx.deeploy_placeholder import (
    CustomLayerNorm, CustomElementMul, CustomColSoftmax,
    CustomColSum, CustomColScatter, CustomGemm, TCneighborgather,
)
from devo_onnx.blocks import (
    GatedResidual, GAP9SoftmaxAgg, SoftAgg, SoftAggBasic,
    GradientClip, GradientZero,
)

LEAF_MODULES = (
    CustomLayerNorm, CustomElementMul, CustomColSoftmax,
    CustomColSum, CustomColScatter, CustomGemm, TCneighborgather,
    GatedResidual, GAP9SoftmaxAgg, SoftAgg, SoftAggBasic,
    GradientClip, GradientZero,
)


ctx_feat_dim = 96
use_pyramid = False
use_softagg = True
use_gru = False
use_ctx_features = True
use_temp = True

update_model = UpdateONNX(3, ctx_feat_dim, use_pyramid, use_softagg, use_gru, use_ctx_features, use_temp)


def loadModel(
    corr_model: nn.Module,
    savePath: Path,
) -> nn.Module:
    """Train the model if no saved weights exist."""

    if not savePath.exists():
        raise("specified weight not exist")

    trained_weight = torch.load(savePath, map_location=torch.device('cpu'))

    for name, param in corr_model.named_parameters():
        param.data = trained_weight['model_state_dict']['update.' + name]

    return corr_model


def calibrate_model(
    model: nn.Module, calib_loader: DataLoader, device: torch.device
) -> None:
    """Calibrate the quantized model."""
    model.eval()
    model.to(device)
    with (
        torch.no_grad(),
        calibration_mode(model),
        tqdm(calib_loader, desc="Calibrating") as pbar,
    ):
        for in_net, kk, *_ in pbar:
            in_net = in_net.to(device).float()
            kk = kk.to(device).float()
            model(in_net, kk[0])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPORT_FOLDER = Path().cwd() / "Tests"
MODEL_PATH = EXPORT_FOLDER / "Models"
DATA_PATH = EXPORT_FOLDER / "Data"

def deepQuantTestUpdate() -> None:
    
    EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)


    # Train or load model
    m = update_model
    model = loadModel(m, MODEL_PATH / "TinyDEVO_batchnorm.pth")

    # Trace with custom tracer that treats custom modules as leaf nodes
    custom_tracer = CustomBrevitasTracer(leafClasses=list(LEAF_MODULES))
    model = _symbolic_trace(custom_tracer, model, concrete_args=None)
    # Preprocess (skip internal trace since we already did it)
    model = preprocess_for_quantize(model, trace_model=False)

    # Quantization configurations
    computeLayerMap = {
        nn.Linear: (
            qnn.QuantLinear,
            {
                "input_quant": Int8ActPerTensorFloat,
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "return_quant_tensor": True,
                "output_bit_width": 8,
                "weight_bit_width": 8,
            },
        ),
    }

    quantActMap = {
        nn.ReLU: (
            qnn.QuantReLU,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 7,
            },
        ),
    }

    quantIdentityMap = {
        "signed": (
            qnn.QuantIdentity,
            {
                "act_quant": Int8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 7,
            },
        ),
        "unsigned": (
            qnn.QuantIdentity,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 7,
            },
        ),
    }

    # Quantize and calibrate
    modelQuant = quantize(
        model,
        compute_layer_map=computeLayerMap,
        quant_act_map=quantActMap,
        quant_identity_map=quantIdentityMap,
    )

    # custom layers adaptation
    
    # Remove quantization from kk input
    for node in modelQuant.graph.nodes:
        if node.name == 'kk_quant':
            kk_node = node.args[0]
            node.replace_all_uses_with(kk_node)
            modelQuant.graph.erase_node(node)
            break
    modelQuant.graph.lint()
    modelQuant.recompile()
    del modelQuant.kk_quant
    

    logged = np.load("Tests/Data/TinyDEVO/update.npz")
    in_tensor = torch.from_numpy(logged["in_net"]).float()
    kk_tensor = torch.from_numpy(logged["stacked_kk"]).float().unsqueeze(0)
    out_tensor = torch.from_numpy(logged["out_net"]).float()
    patch_flow_tensor = torch.from_numpy(logged["patch_flow"]).float()
    confidence_tensor = torch.from_numpy(logged["confidence_weights"]).float()
    calib_dataset = torch.utils.data.TensorDataset(in_tensor, kk_tensor, out_tensor, patch_flow_tensor, confidence_tensor)
    testLoader = DataLoader(calib_dataset, batch_size=64, shuffle=False)

    calibrate_model(modelQuant, testLoader, DEVICE)

    # Export and transform
    sampleInput =  (in_tensor[0:1].to(DEVICE), kk_tensor[0].to(DEVICE))

    exportBrevitas(modelQuant, sampleInput, custom_tracer, debug=True)
