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
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)

from DeepQuant.ExportBrevitas import exportBrevitas
import numpy as np
from devo_onnx.net import UpdateONNX

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
        for images, _ in pbar:
            images = images.to(device)
            images = images.to(torch.float)
            model(images)

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
    model = model.corr

    # Prepare for quantization
    model = preprocess_for_quantize(model)

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

    logged_data = np.load("Tests/Data/TinyDEVO/corr_dump.npz")
    logged_labels = np.load("Tests/Data/TinyDEVO/encoded_corr_dump.npz")
    corr_tensor = torch.from_numpy(logged_data["corr"].squeeze(0)).float()       # (2400, 441)
    label_tensor = torch.from_numpy(logged_labels["encoded_corr"].squeeze(0)).float()  # (2400, 96)
    calib_dataset = torch.utils.data.TensorDataset(corr_tensor, label_tensor)
    testLoader = DataLoader(calib_dataset, batch_size=64, shuffle=False)

    calibrate_model(modelQuant, testLoader, DEVICE)

    # Export and transform
    sampleInput = corr_tensor[0:1]
    print(f"Sample input shape: {sampleInput.shape}")

    exportBrevitas(modelQuant, sampleInput.to(DEVICE), debug=True)
