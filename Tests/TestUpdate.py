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
    import pickle

    with open("Tests/Data/TinyDEVO/update.pkl", "rb") as f:
        logged = pickle.load(f)

    EXPECTED_EDGES = NUM_DST * NUM_PATCHES  # 19 * 24 = 456
    in_list, kk_list, out_list, flow_list, conf_list = [], [], [], [], []

    for sample in logged:
        if sample["in_net"].shape[1] != EXPECTED_EDGES:
            continue

        in_tensor = torch.from_numpy(sample["in_net"]).float()
        kk_tensor = torch.from_numpy(sample["stacked_kk"]).float().unsqueeze(0)
        out_tensor = torch.from_numpy(sample["out_net"]).float()
        patch_flow_tensor = torch.from_numpy(sample["patch_flow"]).float()
        confidence_tensor = torch.from_numpy(sample["confidence_weights"]).float()

        # format the net for gap9 kernels
        # First sort by channel 1 (jj)
        sort_idx = torch.argsort(kk_tensor[0, 1, :], stable=True)
        kk_tensor = kk_tensor[:, :, sort_idx]
        in_tensor = in_tensor[:, sort_idx]
        out_tensor = out_tensor[:, sort_idx]
        patch_flow_tensor = patch_flow_tensor[:, sort_idx]
        confidence_tensor = confidence_tensor[:, sort_idx]

        # Within groups of same channel 1 value, stable sort by channel 2 (kk)
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
        out_tensor = out_tensor[:, sub_sort_idx]
        patch_flow_tensor = patch_flow_tensor[:, sub_sort_idx]
        confidence_tensor = confidence_tensor[:, sub_sort_idx]

        in_tensor = in_tensor.reshape(-1, 19, 24, 96)
        in_tensor = in_tensor[:, 0:NUM_DST, 0:NUM_PATCHES, :]
        in_tensor = in_tensor.reshape(1, -1, 96)

        kk_tensor = kk_tensor.reshape(1, 3, 19, 24)
        kk_tensor = kk_tensor[:, :, 0:NUM_DST, 0:NUM_PATCHES]
        kk_tensor = kk_tensor.reshape(1, 3, -1)

        out_tensor = out_tensor.reshape(-1, 19, 24, 96)
        out_tensor = out_tensor[:, 0:NUM_DST, 0:NUM_PATCHES, :]
        out_tensor = out_tensor.reshape(1, -1, 96)

        patch_flow_tensor = patch_flow_tensor.reshape(-1, 19, 24, 2)
        patch_flow_tensor = patch_flow_tensor[:, 0:NUM_DST, 0:NUM_PATCHES, :]
        patch_flow_tensor = patch_flow_tensor.reshape(1, -1, 2)

        confidence_tensor = confidence_tensor.reshape(-1, 19, 24, 2)
        confidence_tensor = confidence_tensor[:, 0:NUM_DST, 0:NUM_PATCHES, :]
        confidence_tensor = confidence_tensor.reshape(1, -1, 2)

        in_list.append(in_tensor)
        kk_list.append(kk_tensor)
        out_list.append(out_tensor)
        flow_list.append(patch_flow_tensor)
        conf_list.append(confidence_tensor)

    in_tensor = torch.cat(in_list, dim=0)
    kk_tensor = torch.cat(kk_list, dim=0)
    out_tensor = torch.cat(out_list, dim=0)
    patch_flow_tensor = torch.cat(flow_list, dim=0)
    confidence_tensor = torch.cat(conf_list, dim=0)
    print(f"Loaded {len(in_list)} / {len(logged)} samples (filtered to {EXPECTED_EDGES} edges)")

    calib_dataset = torch.utils.data.TensorDataset(in_tensor, kk_tensor, out_tensor, patch_flow_tensor, confidence_tensor)
    testLoader = DataLoader(calib_dataset, batch_size=64, shuffle=False)

    # Export and transform
    sampleInput =  (in_tensor[0:1].to(DEVICE), kk_tensor[0].to(DEVICE))

    # Train or load model
    m = update_model
    model = loadModel(m, MODEL_PATH / "TinyDEVO_batchnorm.pth")

    with torch.no_grad():
        referenceOutput = model.to(DEVICE)(*sampleInput)

    padd_input = np.zeros([1, 19, 24, 96])
    padd_input[:, 0:NUM_DST, 0:NUM_PATCHES, :] = in_tensor[0].reshape(1, NUM_DST, NUM_PATCHES, 96)
    padd_input = padd_input.reshape(1, 456, 96)
    padd_kk = np.array([NUM_PATCHES, NUM_DST])
    input_dict = {'input0': padd_input,
                  'inpu1': padd_kk}
    np.savez("padded_update_input.npz", **input_dict)

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

    np.savez("padded_update_output.npz", **output_dic)


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
    

    fxModelUnified = exportBrevitas(modelQuant, sampleInput, referenceOutput, custom_tracer, debug=True)
    import onnx
    from onnx import TensorProto, helper
    # export onnx
    onnxFile: str = "4_model_dequant_moved.onnx"
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

    # Step 3: Save the model with inferred shapes
    onnx.save(inferredModel, onnxFile)

    #fix customized shapes

    model = onnx.load(onnxFile)

    for inp in model.graph.input:
        if inp.name == "stacked_kk.1":
            inp.type.tensor_type.elem_type = TensorProto.INT32
            inp.type.tensor_type.shape.ClearField('dim')
            dim = inp.type.tensor_type.shape.dim.add()
            dim.dim_value = 2

    onnx.save(model, Path.cwd() / "5_update_model_adapted_shape.onnx")