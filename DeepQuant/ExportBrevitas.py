# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import onnxruntime as ort
import onnx
from brevitas.quant_tensor import IntQuantTensor

from DeepQuant.Injects.Transformations import (
    LinearTransformation,  # Transformation for quantized linear layers (QuantLinear, QuantConv2d)
    ActivationTransformation,  # Transformation for quantized activation functions (QuantReLU, etc.)
    MHATransformation,  # Transformation for quantized multi-head attention modules
)
from DeepQuant.Injects.Executor import (
    TransformationExecutor,
)  # Orchestrates sequential transformations
from .CustomTracer import (
    CustomBrevitasTracer,
    customBrevitasTrace,
)  # Custom FX tracer for Brevitas modules
from DeepQuant.QuantManipulation.ParameterExtractor import (
    extract_brevitas_proxy_params,  # Extracts quantization parameters from Brevitas proxies
    print_quant_params,  # Displays quantization parameters in a readable format
)
from DeepQuant.QuantManipulation.quantDequantMerger import (
    quantDequantMerger,
    mergeReLURequant,
)  # Splits quantization nodes into Quant/Dequant pairs
from DeepQuant.QuantManipulation.QuantNodesDivider import (
    split_quant_nodes,
) 
from brevitas.export.inference import (
    quant_inference_mode,
)  # Inference mode for quantized models
from brevitas.export import (
    export_onnx_qcdq,
)  # Native Brevitas ONNX export functions
from DeepQuant.QuantManipulation.DequantModifier import (
    unifyLinearDequants, unifyTCneighborgather, unifyAdd, unifyColScatter
)  # Unifies dequant nodes in linear layers
from brevitas.fx import brevitas_symbolic_trace  # Brevitas-specific symbolic tracing
from DeepQuant.Utils.GraphPrinter import (
    GraphModulePrinter,
)  # Custom Graph Printer
from DeepQuant.Utils.FxInterpreter import NodeTracer
from torch.fx.graph_module import GraphModule
from typing import Union

from brevitas.fx.brevitas_tracer import Tracer, _symbolic_trace

# ANSI color codes for improved debug output readability
BLUE = "\033[94m"
RED = "\033[31m"
ENDC = "\033[0m"


def _max_tensor_diff(a, b):
    """Compute max absolute difference between two nested tuple/list structures of tensors."""
    if isinstance(a, IntQuantTensor) and isinstance(b, torch.Tensor):
        return torch.max(torch.abs(a[0] - b)).item()
    if isinstance(b, IntQuantTensor) and isinstance(a, torch.Tensor):
        return torch.max(torch.abs(a - b[0])).item()
    
    if isinstance(a, IntQuantTensor) and isinstance(a, IntQuantTensor):
        return torch.max(torch.abs(a[0] - b[0])).item()
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return max(_max_tensor_diff(ai, bi) for ai, bi in zip(a, b))
    if isinstance(a, torch.Tensor) and isinstance(a, torch.Tensor):
        return torch.max(torch.abs(a - b)).item()

    raise RuntimeError("two input a and b have different or unrecognized types")



def exportBrevitas(
    model: nn.Module, exampleInput: Union[torch.Tensor, tuple], referenceOutput : Union[torch.Tensor, tuple], custom_tracer: Tracer = None, debug: bool = False
) -> nn.Module:
    """
    Export a Brevitas model to an FX GraphModule with unrolled quantization operations.

    This function applies a series of transformations to make the quantization steps
    explicit in the model's computation graph, then traces the transformed model using
    a custom FX tracer.

    Args:
        model: The Brevitas-based model to export.
        example_input: A representative input tensor for shape tracing.
        debug: If True, prints transformation progress information.

    Returns:
        nn.Module: An FX GraphModule with explicit quantization operations.
    """

    if custom_tracer == None:
        custom_tracer = CustomBrevitasTracer(debug=debug)

    EXPORT_FOLDER = Path().cwd()
    if Path().cwd().name == "DeepQuant":
        EXPORT_FOLDER = EXPORT_FOLDER / "Tests/ONNX"
        EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)

    printer = GraphModulePrinter()

    ###############################################################################
    # 1. Original Network
    ###############################################################################

    model = _symbolic_trace(custom_tracer, model, concrete_args=None)

    if debug:
        print("\n\n=== 1. Original Network ===\n")
        printer.print_tabular(model)
        print()

    with (
        torch.no_grad(),
        quant_inference_mode(model),
    ):  # Disable gradients and use quantized inference mode
        outputModel = model(
            *exampleInput
        )  # Compute original model output on example input for validation

    # export_onnx_qcdq(  # Export original model to ONNX format with QCDQ (Quant-Cast-DeQuant) nodes
    #     model,  # Model to export
    #     args=exampleInput,  # Example input for tracing
    #     export_path=EXPORT_FOLDER / "1_model_qcdq_original.onnx",
    #     opset_version=13,
    # )

    ###############################################################################
    # 2. Injection of New Modules
    ###############################################################################

    # Create transformation sequence in appropriate order
    transformations = [
        LinearTransformation(),  # Quantized linear layers transformation
        ActivationTransformation(),  # Quantized activation functions transformation
    ]

    # Initialize custom tracer for Brevitas

    # Create and execute transformation sequence using the executor
    executor = TransformationExecutor(transformations, debug=debug, tracer=custom_tracer)
    transformedModel = executor.execute(
        model, exampleInput
    )  # Apply all transformations to the model

    # Generate FX graph using the same tracer for consistency
    fxModel = customBrevitasTrace(
        root=transformedModel,  # Transformed model to trace
        concreteArgs=None,
        tracer=custom_tracer,  # Use same tracer to maintain consistency with transformations
    )
    fxModel.recompile()  # Recompile the FX module to update its forward method
    with torch.no_grad():
        outputFxModel = fxModel(*exampleInput)  # Compute transformed model output

    max_diff = _max_tensor_diff(outputFxModel, outputModel)

    if max_diff < 0.1:  # Check numerical equivalence within tolerance
        if debug:
            print(f"{BLUE} ✓ Injection of New Modules: output is consistent{ENDC}")
    else:
        raise RuntimeError(  # Raise error if outputs differ significantly
            f"{RED} ✗ Injection of New Modules changed the output significantly{ENDC}"
        )

    if debug:
        print(f"{BLUE} ✓ All transformations completed successfully!{ENDC}")
    if debug:
        print("\n=== 2. Network after the Injection of New Modules ===\n")
        printer.print_tabular(fxModel)

    # export_onnx_qcdq(  # Export transformed model to ONNX
    #     fxModel,  # Transformed model
    #     args=exampleInput,
    #     export_path=EXPORT_FOLDER / "2_model_qcdq_transformed.onnx",
    #     opset_version=13,
    # )

    ###############################################################################
    # 3. Extraction of Parameters & Split of Quant Nodes
    ###############################################################################

    # Extract quantization parameters from the network's proxies
    proxyParams = extract_brevitas_proxy_params(
        fxModel
    )  # Get scale, zero_point, bit_width for each quant node

    if debug:
        print_quant_params(
            proxyParams
        )  # Display extracted parameters in a readable format

    # Split quantization nodes into separate Quant and Dequant nodes
    splitFxModel = split_quant_nodes(
        fxModel, proxyParams, debug
    )  # Transform quant nodes into quant-dequant pairs
    splitFxModel.recompile()  # Recompile to update forward method with new nodes

    with torch.no_grad():
        outputFxModelSplitQuant = splitFxModel(
            *exampleInput
        )  # Compute output after node splitting

    max_diff = _max_tensor_diff(outputFxModelSplitQuant, outputModel)

    if max_diff < 1:  # Verify numerical consistency
        if debug:
            print(f"{BLUE} ✓ Split of Quant Nodes: output is consistent， max diff is {max_diff}{ENDC}")
    else:
        raise RuntimeError(  # Raise error if inconsistent
            f"{RED} ✗ Split of Quant Nodes changed the output significantly{ENDC}"
        )

    if debug:
        print("\n=== 3. Network after the Split of Quant Nodes ===\n")
        printer.print_tabular(splitFxModel)
        print()

    torch.onnx.export(
        splitFxModel,
        args=tuple(exampleInput),
        f=EXPORT_FOLDER / "3_model_splitted_quant.onnx",
        opset_version=17,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
    )

    # return split_fx_model

    ###############################################################################
    # 4. Modification of Dequant Nodes (shift them down)
    ###############################################################################

    # Perform the unification of linear dequant nodes (move dequantization after computation)
    fxModelUnified = unifyLinearDequants(splitFxModel, debug=debug)
    fxModelUnified = unifyTCneighborgather(fxModelUnified, debug=debug)
    fxModelUnified = unifyAdd(fxModelUnified, debug=debug)
    # fxModelUnified = unifyColScatter(fxModelUnified, debug=debug)
    fxModelUnified.recompile()  # Recompile to update forward method with new node arrangement

    # Compute output after dequant node unification
    with torch.no_grad():
        outputFxModelDequantModified = fxModelUnified(
            *exampleInput
        )  # Output after dequant modification

    if debug:
        print("\n=== 4. Network after the Modification of Dequant Nodes ===\n")
        printer.print_tabular(fxModelUnified)
        print()


    max_diff = _max_tensor_diff(outputFxModelDequantModified, outputModel)

    # Verify numerical consistency after dequant modification
    if max_diff < 1:  # Verify numerical consistency
        if debug:
            print(f"{BLUE} ✓ Modification of Dequant Nodes: output is consistent， max diff is {max_diff}{ENDC}")
    else:
        raise RuntimeError(  # Raise error if inconsistent
            f"{RED} ✗ Modification of Dequant Nodes changed the output significantly{ENDC}"
        )
    
    ###############################################################################
    # 5. Merge redundent quant/dequants
    ###############################################################################
    fxModelUnified = quantDequantMerger(fxModelUnified, debug=debug)
    fxModelUnified = mergeReLURequant(fxModelUnified, debug=debug)


    if debug:
        print("\n=== 5. Network after the merging quant dequant pairs ===\n")
        printer.print_tabular(fxModelUnified)
        print()
        
    # compute the SNR of output from final model and original fp model
    def _compute_snr(a, b):
        """Compute SNR (dB) between nested tuple/list structures of tensors. Returns a float or list of floats."""
        def _to_tensor(x):
            return x[0].float() if isinstance(x, IntQuantTensor) else x.float()

        if isinstance(a, (torch.Tensor, IntQuantTensor)) and isinstance(b, (torch.Tensor, IntQuantTensor)):
            ta, tb = _to_tensor(a), _to_tensor(b)
            noise = (ta - tb).float()
            return (10 * torch.log10(tb.pow(2).mean() / noise.pow(2).mean())).item()
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return [_compute_snr(ai, bi) for ai, bi in zip(a, b)]
        raise RuntimeError("two input a and b have different or unrecognized types")

    snrs = _compute_snr(outputFxModelDequantModified, referenceOutput)
    print(f"\n{BLUE}=== SNR between quantized output and FP reference ==={ENDC}")
    if isinstance(snrs, list):
        for i, s in enumerate(snrs):
            print(f"  Output[{i}] SNR: {s:.2f} dB")
    else:
        print(f"  SNR: {snrs:.2f} dB")

    # export onnx
    onnxFile: str = EXPORT_FOLDER / "4_model_dequant_moved.onnx"
    torch.onnx.export(
        fxModelUnified,
        args=tuple(exampleInput),
        # f=EXPORT_FOLDER / "4_model_dequant_moved.onnx",
        f=onnxFile,
        opset_version=17,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
    )

    #export inputs and outputs
    inputFile: str = EXPORT_FOLDER / "inputs.npz"
    input_dict = {f"input_{i}": t.cpu().numpy() for i, t in enumerate(exampleInput)}
    input_dict['input_1'] = np.array([19, 24]) #compact kk for actual inference
    np.savez(inputFile, **input_dict)
    print(f"Input data saved to {inputFile} ✓")

    outputFile: str = EXPORT_FOLDER / "outputs.npz"

    def flatten_tensors(data, prefix="output"):
        results = {}
        if isinstance(data, torch.Tensor):
            results[prefix] = data.cpu().numpy()
        elif isinstance(data, (tuple, list)):
            for i, item in enumerate(data):
                results.update(flatten_tensors(item, f"{prefix}_{i}"))
        return results

    output_dict = flatten_tensors(outputFxModelDequantModified)
    np.savez(outputFile, **output_dict)
    print(f"Output data saved to {outputFile} ✓")

    # Step 2: Load the model and run shape inference
    # (All tensors in ONNX graph should have explicit shape information)
    onnxModel = onnx.load(onnxFile)
    inferredModel = onnx.shape_inference.infer_shapes(onnxModel)

    # Step 3: Save the model with inferred shapes
    onnx.save(inferredModel, onnxFile)


    return fxModelUnified  # Return the final optimized FX GraphModule
