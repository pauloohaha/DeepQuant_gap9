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
    QuantTCneighborgather,
)
from devo_onnx.blocks import (
    GatedResidual, GAP9SoftmaxAgg, SoftAgg, SoftAggBasic,
    GradientClip, GradientZero,
)

LEAF_MODULES = (
    CustomLayerNorm, CustomElementMul, CustomColSoftmax,
    CustomColSum, CustomColScatter, CustomGemm, TCneighborgather,
    GatedResidual, SoftAgg, SoftAggBasic,
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
        param.data = trained_weight['model_state_dict']['update.corr.' + name]

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
        for in_net, *_ in pbar:
            in_net = in_net.to(device).float()
            model(in_net)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPORT_FOLDER = Path().cwd() / "Tests"
MODEL_PATH = EXPORT_FOLDER / "Models"
DATA_PATH = EXPORT_FOLDER / "Data"

def deepQuantTestCorr() -> None:
    
    EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    NUM_PATCHES = 24


    # load input data from update.pkl (corr_input field)
    import pickle

    with open("Tests/Data/TinyDEVO/update.pkl", "rb") as f:
        logged = pickle.load(f)

    in_list = []
    for sample in logged:
        corr = sample["corr_input"].float()  # (1, num_edges, 441)
        num_edges = corr.shape[1]
        # Each group of 24 edges is one corr sample
        corr = corr.reshape(num_edges // 24, 24, 441)
        # Pad input from 441 to 448 (multiple of 16 for NE16)
        corr = torch.nn.functional.pad(corr, (0, 7))
        in_list.append(corr)

    in_tensor = torch.cat(in_list, dim=0)  # (total_groups, 24, 448)
    print(f"Loaded {in_tensor.shape[0]} corr samples from {len(logged)} timesteps")

    calib_dataset = torch.utils.data.TensorDataset(in_tensor)
    testLoader = DataLoader(calib_dataset, batch_size=64, shuffle=False)

    # Export and transform
    sampleInput = (in_tensor[0:1].to(DEVICE),)

    # Train or load model
    m = update_model.corr

    # Replace first layer: nn.Linear(441, 96) -> nn.Linear(448, 96) for NE16
    padded_first = nn.Linear(448, m[0].out_features, bias=m[0].bias is not None)
    nn.init.zeros_(padded_first.weight)
    nn.init.zeros_(padded_first.bias)
    m[0] = padded_first

    # Load weights with padding for first layer
    trained_weight = torch.load(MODEL_PATH / "TinyDEVO_batchnorm.pth", map_location='cpu')
    for name, param in m.named_parameters():
        saved = trained_weight['model_state_dict']['update.corr.' + name]
        if name == '0.weight':
            param.data[:, :441] = saved
        else:
            param.data = saved
    model = m


    with torch.no_grad():
        referenceOutput = model.to(DEVICE)(*sampleInput)


    input_dict = {'input0': in_tensor[0:1]}
    np.savez("corr_input.npz", **input_dict)

    output_dic = {
        'out_0': referenceOutput.cpu()
    }

    np.savez("corr_output.npz", **output_dic)


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
        TCneighborgather: (
            QuantTCneighborgather,
            {
                "act_quant": Int8ActPerTensorFloat,
                "return_quant_tensor": True,
            },
        ),
    }

    quantActMap = {
        nn.ReLU: (
            qnn.QuantReLU,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
            },
        ),
    }

    quantIdentityMap = {
        "signed": (
            qnn.QuantIdentity,
            {
                "act_quant": Int8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
            },
        ),
        "unsigned": (
            qnn.QuantIdentity,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
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

    # Change input_quant to unsigned for QuantLinear modules after QuantReLU
    for node in modelQuant.graph.nodes:
        if node.op == 'call_module':
            mod = modelQuant.get_submodule(node.target)
            if isinstance(mod, qnn.QuantReLU):
                for user in node.users:
                    if user.op == 'call_module':
                        user_mod = modelQuant.get_submodule(user.target)
                        if isinstance(user_mod, qnn.QuantLinear):
                            # Create unsigned input_quant proxy and swap
                            unsigned_ref = qnn.QuantLinear(
                                user_mod.in_features, user_mod.out_features,
                                input_quant=Uint8ActPerTensorFloat,
                                weight_quant=Int8WeightPerTensorFloat,
                                bias=user_mod.bias is not None,
                                return_quant_tensor=True,
                            )
                            user_mod.input_quant = unsigned_ref.input_quant
                            print(f"  Changed {user.target} input_quant to unsigned")
                            
    calibrate_model(modelQuant, testLoader, DEVICE)
    

    fxModelUnified = exportBrevitas(modelQuant, sampleInput, referenceOutput, custom_tracer, debug=True)

    import onnx
    from onnx import TensorProto, helper

    # export onnx
    onnxFile: str = EXPORT_FOLDER / "4_model_dequant_moved.onnx"
    torch.onnx.export(
        fxModelUnified,
        args=tuple(sampleInput),
        # f=EXPORT_FOLDER / "4_model_dequant_moved.onnx",
        f=onnxFile,
        opset_version=17,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
    )


    # Step 2: Load the model and run shape inference
    # (All tensors in ONNX graph should have explicit shape information)
    onnxModel = onnx.load(onnxFile)
    inferredModel = onnx.shape_inference.infer_shapes(onnxModel)


    # for renaming
    onnx.save(inferredModel, Path.cwd() / "5_corr_model_adapted_shape.onnx")