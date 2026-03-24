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

    NUM_PATCHES = 24
    NUM_DST = 19

    # load input data
    logged = np.load("Tests/Data/TinyDEVO/update.npz")
    in_tensor = torch.from_numpy(logged["in_net"]).float()
    kk_tensor = torch.from_numpy(logged["stacked_kk"]).float().unsqueeze(0)
    # format the net for gap9 kernels
    # First sort by channel 1
    sort_idx = torch.argsort(kk_tensor[0, 1, :], stable=True)
    kk_tensor = kk_tensor[:, :, sort_idx]
    in_tensor = in_tensor[:, sort_idx]

    # Within groups of same channel 1 value, stable sort by channel 2
    ch1 = kk_tensor[0, 1, :]
    ch2 = kk_tensor[0, 2, :]
    sub_sort_idx = torch.arange(ch1.shape[0])
    for val in ch1.unique():
        mask = ch1 == val
        indices = mask.nonzero(as_tuple=True)[0]
        local_order = torch.argsort(ch2[indices], stable=True)
        sub_sort_idx[indices] = indices[local_order]
    kk_tensor = kk_tensor[:, :, sub_sort_idx]
    in_tensor = in_tensor[:, sub_sort_idx]

    in_tensor = in_tensor.reshape(-1, 19, 24, 96)
    in_tensor = in_tensor[:, 0:NUM_DST, 0:NUM_PATCHES, :]
    in_tensor = in_tensor.reshape(1, -1, 96)

    kk_tensor = kk_tensor.reshape(1, 3, 19, 24)
    kk_tensor = kk_tensor[:, :, 0:NUM_DST, 0:NUM_PATCHES]
    kk_tensor = kk_tensor.reshape(1, 3, -1)


    out_tensor = torch.from_numpy(logged["out_net"]).float()
    patch_flow_tensor = torch.from_numpy(logged["patch_flow"]).float()
    confidence_tensor = torch.from_numpy(logged["confidence_weights"]).float()
    calib_dataset = torch.utils.data.TensorDataset(in_tensor, kk_tensor, out_tensor, patch_flow_tensor, confidence_tensor)
    testLoader = DataLoader(calib_dataset, batch_size=64, shuffle=False)

    # Export and transform
    sampleInput =  (in_tensor[0:1].to(DEVICE), kk_tensor[0].to(DEVICE))

    # Train or load model
    m = update_model
    model = loadModel(m, MODEL_PATH / "TinyDEVO_batchnorm.pth")

    # debug deeploy: read dump.bin as int8 and reshape to (-1, 96)
    dump_path = "/usr/scratch2/larain8/pudeng/deeploy_fix/deeploy_merge/DeeployTest/dump.bin"
    dump_data = torch.tensor(np.fromfile(dump_path, dtype=np.int8)).reshape(-1, 96)
    print(f"dump.bin loaded: shape={dump_data.shape}, min={dump_data.min()}, max={dump_data.max()}")

    with torch.no_grad():
        referenceOutput = model.to(DEVICE)(*sampleInput)

    padd_input = np.zeros([1, 19, 24, 96])
    padd_input[:, 0:NUM_DST, 0:NUM_PATCHES, :] = in_tensor.reshape(1, NUM_DST, NUM_PATCHES, 96)
    padd_input = padd_input.reshape(1, 456, 96)
    padd_kk = np.array([NUM_PATCHES, NUM_DST])
    input_dict = {'input0': padd_input,
                  'inpu1': padd_kk}
    np.savez("padded_input.npz", **input_dict)

    padd_output_net=np.zeros([1, 19, 24, 96])
    padd_output_net[:, 0:NUM_DST, 0:NUM_PATCHES, :] = referenceOutput[0].reshape(1, NUM_DST, NUM_PATCHES, 96).cpu()
    padd_output_net = padd_output_net.reshape(1, 456, 96)

    padd_output_flow = np.zeros([1, 19, 24, 2])
    padd_output_flow[:, 0:NUM_DST, 0:NUM_PATCHES, :] = referenceOutput[1].reshape(1, NUM_DST, NUM_PATCHES, 2).cpu()
    padd_output_flow = padd_output_flow.reshape(1, 456, 2)

    padd_output_weight = np.zeros([1, 19, 24, 2])
    padd_output_weight[:, 0:NUM_DST, 0:NUM_PATCHES, :] = referenceOutput[2].reshape(1, NUM_DST, NUM_PATCHES, 2).cpu()
    padd_output_weight = padd_output_weight.reshape(1, 456, 2)

    output_dic = {
        'out_0': padd_output_net,
        'out_1': padd_output_flow,
        'out_2': padd_output_weight
    }

    np.savez("padded_output.npz", **output_dic)


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

    calibrate_model(modelQuant, testLoader, DEVICE)
    

    exportBrevitas(modelQuant, sampleInput, referenceOutput, custom_tracer, debug=True)

    #fix customized shapes
    import onnx
    from onnx import TensorProto, helper

    onnxFile = Path.cwd() / "4_model_dequant_moved.onnx"
    model = onnx.load(onnxFile)

    for inp in model.graph.input:
        if inp.name == "stacked_kk.1":
            inp.type.tensor_type.elem_type = TensorProto.INT32
            inp.type.tensor_type.shape.ClearField('dim')
            dim = inp.type.tensor_type.shape.dim.add()
            dim.dim_value = 2

    onnx.save(model, Path.cwd() / "5_model_adapted_shape.onnx")