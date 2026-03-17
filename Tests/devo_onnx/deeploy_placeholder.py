import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

import brevitas

from . import fastba

NUM_PATCHES = 24
NUM_MAXEDGES = 19

class CustomElementMulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        """
        Forward pass: takes x and kk, returns x unchanged (placeholder)
        ctx is used to save tensors for backward pass
        """
        if type(x) == brevitas.quant_tensor.int_quant_tensor.IntQuantTensor:
            x = x[0]

        if type(y) == brevitas.quant_tensor.int_quant_tensor.IntQuantTensor:
            y = y[0]

        w = x * y
        ctx.save_for_backward(x, y)
        return w

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients for both inputs
        Returns: (grad_x, grad_kk)
        """
        x, y = ctx.saved_tensors
        grad_x = grad_output
        grad_y = None  # kk is typically integer indices, no gradient needed
        return grad_x, grad_y

    @staticmethod
    def symbolic(g, x, y):
        """
        ONNX symbolic function - defines how to export this as a black box.
        This is called during ONNX export and creates a custom operator node.
        Takes both x and kk as inputs.
        """
        # Create a custom operator node that ONNX will NOT decompose
        # The domain "custom_domain" makes it a custom operator
        # The op_type "CustomElementMul" is the name of your custom operation
        output = g.op("custom_domain::CustomElementMul",
                      x, y,
                      outputs=1)

        # Set the output type to match input x for shape propagation
        # This is critical for ONNX shape inference to work correctly
        output.setType(x.type())

        return output


# Custom layer that uses the autograd function
class CustomElementMul(nn.Module):
    def __init__(self):
        super(CustomElementMul, self).__init__()

    def forward(self, x, y):
        # Use the custom autograd function
        # This ensures ONNX treats it as a black box
        return CustomElementMulFunc.apply(x, y)






# Custom Autograd Function - This prevents ONNX from tracing through the operation
class CustomColSoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stacked_kk, dir):
        """
        Forward pass: takes x and kk, returns x unchanged (placeholder)
        ctx is used to save tensors for backward pass
        """
        if type(x) == brevitas.quant_tensor.int_quant_tensor.IntQuantTensor:
            x = x[0]

        if dir == 1:
            #frame agg, need to reconstruct edges
            kk = stacked_kk[0]*1234+stacked_kk[1]
        else:
            kk = stacked_kk[2]
        
        _, jx = torch.unique(kk, return_inverse=True)

        w = torch_scatter.scatter_softmax(x, jx, dim=1)
        ctx.save_for_backward(x, kk)
        return w

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients for both inputs
        Returns: (grad_x, grad_kk)
        """
        x, kk = ctx.saved_tensors
        grad_x = grad_output
        grad_kk = None  # kk is typically integer indices, no gradient needed
        return grad_x, grad_kk

    @staticmethod
    def symbolic(g, x, kk,dir):
        """
        ONNX symbolic function - defines how to export this as a black box.
        This is called during ONNX export and creates a custom operator node.
        Takes both x and kk as inputs.
        """
        # Create a custom operator node that ONNX will NOT decompose
        # The domain "custom_domain" makes it a custom operator
        # The op_type "CustomColSoftmax" is the name of your custom operation
        output = g.op("custom_domain::CustomColSoftmax",
                      x, kk,
                      dir_i=dir,
                      outputs=1)

        # Set the output type to match input x for shape propagation
        # This is critical for ONNX shape inference to work correctly
        output.setType(x.type())

        return output


# Custom layer that uses the autograd function
class CustomColSoftmax(nn.Module):
    def __init__(self,dir):
        super(CustomColSoftmax, self).__init__()
        self.dir = dir
    def forward(self, x, stacked_kk):
        # Use the custom autograd function
        # This ensures ONNX treats it as a black box
    
        return CustomColSoftmaxFunc.apply(x, stacked_kk, self.dir)
    






# Custom Autograd Function - This prevents ONNX from tracing through the operation
class CustomColSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stacked_kk, dir):
        """
        Forward pass: takes x and kk, returns x unchanged (placeholder)
        ctx is used to save tensors for backward pass
        """
        if type(x) == brevitas.quant_tensor.int_quant_tensor.IntQuantTensor:
            x = x[0]
            
        if dir == 1:
            kk = stacked_kk[0]*1234+stacked_kk[1]
        else:
            kk = stacked_kk[2]
        
        _, jx = torch.unique(kk, return_inverse=True)

        sum = torch_scatter.scatter_sum(x, jx, dim=1)
        ctx.save_for_backward(x, kk)
        return sum

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients for both inputs
        Returns: (grad_x, grad_kk)
        """
        x, kk = ctx.saved_tensors
        grad_x = grad_output
        grad_kk = None  # kk is typically integer indices, no gradient needed
        return grad_x, grad_kk

    @staticmethod
    def symbolic(g, x, kk, dir):
        """
        ONNX symbolic function - defines how to export this as a black box.
        This is called during ONNX export and creates a custom operator node.
        Takes both x and kk as inputs.
        """
        # Create a custom operator node that ONNX will NOT decompose
        # The domain "custom_domain" makes it a custom operator
        # The op_type "CustomColSum" is the name of your custom operation
        output = g.op("custom_domain::CustomColSum",
                      x, kk,
                      dir_i=dir,
                      outputs=1)

        # Output dim1 depends on dir: NUM_PATCHES if dir==0, NUM_MAXEDGES if dir==1
        input_sizes = x.type().symbolic_sizes()
        dim1 = NUM_PATCHES if dir == 0 else NUM_MAXEDGES
        output_sizes = [input_sizes[0], dim1, input_sizes[2]]
        output.setType(x.type().with_sizes(output_sizes))

        return output


# Custom layer that uses the autograd function
class CustomColSum(nn.Module):
    def __init__(self, dir):
        super(CustomColSum, self).__init__()
        self.dir = dir
    def forward(self, x, stacked_kk):
        # Use the custom autograd function
        # This ensures ONNX treats it as a black box

        return CustomColSumFunc.apply(x, stacked_kk, self.dir)
    








# Custom Autograd Function - This prevents ONNX from tracing through the operation
class CustomColScatterFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, net, stacked_kk, dir):
        """
        Forward pass: takes x and kk, returns x unchanged (placeholder)
        ctx is used to save tensors for backward pass
        """
        if type(x) == brevitas.quant_tensor.int_quant_tensor.IntQuantTensor:
            x = x[0]
        
        if type(net) == brevitas.quant_tensor.int_quant_tensor.IntQuantTensor:
            net = net[0]

        if dir == 1:
            #frame agg, need to reconstruct edges
            kk = stacked_kk[0]*1234+stacked_kk[1]
        else:
            kk = stacked_kk[2]

        _, jx = torch.unique(kk, return_inverse=True)
        scatter = x[:, jx] 
        result = net + scatter
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients for both inputs
        Returns: (grad_x, grad_kk)
        """
        x, kk = ctx.saved_tensors
        grad_x = grad_output
        grad_kk = None  # kk is typically integer indices, no gradient needed
        return grad_x, grad_kk

    @staticmethod
    def symbolic(g, x, net, kk, dir):
        """
        ONNX symbolic function - defines how to export this as a black box.
        This is called during ONNX export and creates a custom operator node.
        Takes both x and kk as inputs.
        """
        # Create a custom operator node that ONNX will NOT decompose
        # The domain "custom_domain" makes it a custom operator
        # The op_type "CustomColScatter" is the name of your custom operation
        output = g.op("custom_domain::CustomColScatter",
                      x, net, kk,
                      dir_i=dir,
                      outputs=1)

        # Output dim1 is NUM_PATCHES x NUM_MAXEDGES 
        input_sizes = x.type().symbolic_sizes()
        dim1 = NUM_PATCHES * NUM_MAXEDGES
        output_sizes = [input_sizes[0], dim1, input_sizes[2]]
        output.setType(x.type().with_sizes(output_sizes))

        return output


# Custom layer that uses the autograd function
class CustomColScatter(nn.Module):
    def __init__(self, dir):
        super(CustomColScatter, self).__init__()
        self.dir = dir

    def forward(self, x, net, stacked_kk):
        # Use the custom autograd function
        # This ensures ONNX treats it as a black box

        return CustomColScatterFunc.apply(x, net, stacked_kk, self.dir)



# Custom LayerNorm Autograd Function
class CustomLayerNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, normalized_shape, eps):
        """
        Forward pass: Implements LayerNorm using PyTorch's original implementation

        Args:
            x: Input tensor
            weight: Scale parameter (gamma)
            bias: Shift parameter (beta) - this is an input instead of a parameter
            normalized_shape: Shape to normalize over (typically the last dimension(s))
            eps: Epsilon for numerical stability (attribute)

        Returns:
            Normalized and scaled output
        """
        # Use PyTorch's native LayerNorm implementation
        output = F.layer_norm(x, normalized_shape, weight, bias, eps)

        # Save tensors needed for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradients for inputs

        Returns: (grad_x, grad_weight, grad_bias, None, None)
        """
        x, weight, bias = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        # Compute gradients using PyTorch's autograd
        # We need to enable grad for the inputs to compute gradients
        x_requires_grad = x.requires_grad
        weight_requires_grad = weight.requires_grad if weight is not None else False
        bias_requires_grad = bias.requires_grad if bias is not None else False

        with torch.enable_grad():
            x_temp = x.detach().requires_grad_(True)
            weight_temp = weight.detach().requires_grad_(True) if weight is not None else None
            bias_temp = bias.detach().requires_grad_(True) if bias is not None else None

            output = F.layer_norm(x_temp, normalized_shape, weight_temp, bias_temp, eps)

            # Compute gradients
            grad_inputs = torch.autograd.grad(
                output,
                [x_temp, weight_temp, bias_temp],
                grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )

        grad_x = grad_inputs[0] if x_requires_grad else None
        grad_weight = grad_inputs[1] if weight_requires_grad else None
        grad_bias = grad_inputs[2] if bias_requires_grad else None

        # Return gradients for all inputs (None for non-tensor arguments)
        return grad_x, grad_weight, grad_bias, None, None

    @staticmethod
    def symbolic(g, x, weight, bias, normalized_shape, eps):
        """
        ONNX symbolic function - exports as a custom LayerNorm operator

        Args:
            g: ONNX graph builder
            x: Input tensor
            weight: Weight/scale tensor
            bias: Bias/shift tensor (as input)
            normalized_shape: Python tuple (not exported to ONNX)
            eps: Epsilon value (exported as attribute)

        Returns:
            ONNX operator node
        """
        # Create a custom operator node
        # epsilon is an attribute, bias is an input
        output = g.op("custom_domain::LayerNormalization",
                      x, weight, bias,
                      epsilon_f=eps,  # Export epsilon as a float attribute
                      outputs=1)

        # Set the output type to match input x
        output.setType(x.type())

        return output


# Custom LayerNorm layer
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Custom LayerNorm that exports to ONNX as a single fused operator

        Args:
            normalized_shape: Input shape from an expected input of size
                [* x normalized_shape[0] x normalized_shape[1] x ...]
            eps: Epsilon for numerical stability (default: 1e-5)
            elementwise_affine: Whether to learn affine parameters (default: True)
        """
        super(CustomLayerNorm, self).__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        Forward pass through custom LayerNorm

        Args:
            x: Input tensor

        Returns:
            Normalized output
        """
        # Use the custom autograd function
        # This ensures ONNX treats it as a single fused operator
        return CustomLayerNormFunc.apply(
            x,
            self.weight,
            self.bias,
            self.normalized_shape,
            self.eps
        )

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)


# Custom Gemm Autograd Function
class CustomGemmFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, alpha, beta, transA, transB):
        """
        Forward pass: Implements Gemm operation

        Gemm computes: Y = alpha * A' * B' + beta * C
        where A' = A if transA=0 else A^T
              B' = B if transB=0 else B^T
              C = bias (broadcast)

        Args:
            input: Input tensor (A)
            weight: Weight tensor (B)
            bias: Bias tensor (C), can be None
            alpha: Scalar multiplier for A*B
            beta: Scalar multiplier for bias
            transA: Whether to transpose A (0 or 1)
            transB: Whether to transpose B (0 or 1)

        Returns:
            Output tensor Y
        """
        # Apply transpositions if needed
        A = input.transpose(-2, -1) if transA else input
        B = weight.transpose(-2, -1) if transB else weight

        # Compute matrix multiplication
        output = torch.matmul(A, B)

        # Apply alpha scaling
        if alpha != 1.0:
            output = alpha * output

        # Add bias with beta scaling if bias exists
        if bias is not None and beta != 0.0:
            if beta != 1.0:
                output = output + beta * bias
            else:
                output = output + bias

        # Save tensors for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.transA = transA
        ctx.transB = transB

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradients for inputs

        Returns: (grad_input, grad_weight, grad_bias, None, None, None, None)
        """
        input, weight, bias = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        transA = ctx.transA
        transB = ctx.transB

        grad_input = grad_weight = grad_bias = None

        # Compute gradient w.r.t. input
        if ctx.needs_input_grad[0]:
            if transA:
                if transB:
                    grad_input = alpha * torch.matmul(weight, grad_output.transpose(-2, -1)).transpose(-2, -1)
                else:
                    grad_input = alpha * torch.matmul(weight.transpose(-2, -1), grad_output.transpose(-2, -1)).transpose(-2, -1)
            else:
                if transB:
                    grad_input = alpha * torch.matmul(grad_output, weight)
                else:
                    grad_input = alpha * torch.matmul(grad_output, weight.transpose(-2, -1))

        # Compute gradient w.r.t. weight
        if ctx.needs_input_grad[1]:
            if transA:
                if transB:
                    grad_weight = alpha * torch.matmul(grad_output.transpose(-2, -1), input).transpose(-2, -1)
                else:
                    grad_weight = alpha * torch.matmul(grad_output.transpose(-2, -1), input)
            else:
                if transB:
                    grad_weight = alpha * torch.matmul(input.transpose(-2, -1), grad_output).transpose(-2, -1)
                else:
                    grad_weight = alpha * torch.matmul(input.transpose(-2, -1), grad_output)

        # Compute gradient w.r.t. bias
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = beta * grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

        # Return gradients (None for non-tensor arguments: alpha, beta, transA, transB)
        return grad_input, grad_weight, grad_bias, None, None, None, None

    @staticmethod
    def symbolic(g, input, weight, bias, alpha, beta, transA, transB):
        """
        ONNX symbolic function - exports as standard Gemm operator with attributes

        Args:
            g: ONNX graph builder
            input: Input tensor A
            weight: Weight tensor B
            bias: Bias tensor C (can be None)
            alpha: Alpha attribute
            beta: Beta attribute
            transA: TransA attribute
            transB: TransB attribute

        Returns:
            ONNX Gemm operator node
        """
        if bias is not None:
            # Create Gemm with all three inputs
            output = g.op("Gemm",
                         input, weight, bias,
                         alpha_f=float(alpha),
                         beta_f=float(beta),
                         transA_i=int(transA),
                         transB_i=int(transB),
                         outputs=1)
        else:
            # Create Gemm with only two inputs (no bias)
            output = g.op("Gemm",
                         input, weight,
                         alpha_f=float(alpha),
                         beta_f=float(beta),
                         transA_i=int(transA),
                         transB_i=int(transB),
                         outputs=1)

        # Set the output type to match input
        output.setType(input.type())

        return output


# Custom Gemm Layer (replaces nn.Linear)
class CustomGemm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, alpha=1.0, beta=1.0, transA=0, transB=1):
        """
        Custom Gemm layer that exports to ONNX as a single Gemm operator with proper attributes

        This layer is designed to replace nn.Linear with explicit control over Gemm attributes.
        By default, it behaves like nn.Linear with transB=1 (weight is transposed).

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias (default: True)
            alpha: Alpha multiplier for matmul (default: 1.0)
            beta: Beta multiplier for bias (default: 1.0)
            transA: Whether to transpose input (default: 0)
            transB: Whether to transpose weight (default: 1, matches nn.Linear behavior)
        """
        super(CustomGemm, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB

        # Initialize weight
        # For transB=1 (default), weight shape is [in_features, out_features] like nn.Linear
        # For transB=0, weight shape is [out_features, in_features]
        if transB == 1:
            self.weight = nn.Parameter(torch.empty(in_features, out_features))
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters (use same initialization as nn.Linear)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('linear'))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / torch.sqrt(torch.tensor(fan_in, dtype=torch.float32))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Forward pass through custom Gemm

        Args:
            x: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., out_features]
        """
        return CustomGemmFunc.apply(
            x,
            self.weight,
            self.bias,
            self.alpha,
            self.beta,
            self.transA,
            self.transB
        )

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, alpha={}, beta={}, transA={}, transB={}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.alpha, self.beta, self.transA, self.transB
        )


liner_implementation = nn.Linear
# liner_implementation = CustomGemm



class TCneighborgatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, net, stacked_kk, dir):
        
        if(type(net) != torch.Tensor):
            net = net[0]

        kk  = stacked_kk[2]
        jj  = stacked_kk[1]

        ix, jx = fastba.neighbors(kk.long(), jj.long())
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        if(dir == 0):
            return mask_ix * net[:,ix]
        else:
            return mask_jx * net[:,jx]

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients for both inputs
        Returns: (grad_x, grad_kk)
        """
        x, y = ctx.saved_tensors
        grad_x = grad_output
        grad_y = None  # kk is typically integer indices, no gradient needed
        return grad_x, grad_y

    @staticmethod
    def symbolic(g, net, kk, dir):
        """
        ONNX symbolic function - defines how to export this as a black box.
        This is called during ONNX export and creates a custom operator node.
        Takes both x and kk as inputs.
        """
        # Create a custom operator node that ONNX will NOT decompose
        # The domain "custom_domain" makes it a custom operator
        # The op_type "CustomElementMul" is the name of your custom operation
        output = g.op("custom_domain::TCneighborGather",
                      net, kk,
                      dir_i=dir,
                      outputs=1)

        # Set the output type to match input x for shape propagation
        # This is critical for ONNX shape inference to work correctly
        output.setType(net.type())

        return output


# Custom layer that uses the autograd function
class TCneighborgather(nn.Module):
    def __init__(self, dir = 0):
        super(TCneighborgather, self).__init__()
        self.dir = dir

    def forward(self, net, stacked_kk):
        # Use the custom autograd function
        # This ensures ONNX treats it as a black box
        return TCneighborgatherFunc.apply(net, stacked_kk, self.dir)


class QuantTCneighborgather(nn.Module):
    """Quantized TCneighborgather for compute_layer_map replacement.
    dir is auto-transferred from old module by ModuleToModuleByInstance.
    """
    def __init__(self, dir=0, act_quant=None, return_quant_tensor=True, **kwargs):
        super().__init__()
        self.tc_neighbor_gather = TCneighborgather(dir)

    def forward(self, net, stacked_kk):
        out = self.tc_neighbor_gather(net, stacked_kk)
        return out

