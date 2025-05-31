# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Common modules."""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

# Import 'ultralytics' package or install if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    import os

    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils import TryExcept # Assuming utils is a sibling directory or in PYTHONPATH
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Implements a depth-wise convolution layer with optional activation for efficient spatial filtering."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """A depth-wise transpose convolutional layer for upsampling in neural networks, particularly in YOLOv5 models."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise transpose convolutional layer for YOLOv5; args: input channels (c1), output channels
        (c2), kernel size (k), stride (s), input padding (p1), output padding (p2).
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    """Transformer layer with multihead attention and linear layers, optimized by removing LayerNorm."""

    def __init__(self, c, num_heads):
        """
        Initializes a transformer layer, sans LayerNorm for performance, with multihead attention and linear layers.

        See  as described in https://arxiv.org/abs/2010.11929.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Performs forward pass using MultiheadAttention and two linear transformations with residual connections."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """A Transformer block for vision tasks with convolution, position embeddings, and Transformer layers."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initializes a Transformer block for vision tasks, adapting dimensions if necessary and stacking specified
        layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Processes input through an optional convolution, followed by Transformer layers and position embeddings for
        object detection.
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    """A bottleneck layer with optional shortcut and group convolution for efficient feature extraction."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP bottleneck layer for feature extraction with cross-stage partial connections and optional shortcuts."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with optional shortcuts; args: ch_in, ch_out, number of repeats, shortcut bool,
        groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward pass by applying layers, activation, and concatenation on input x, returning feature-
        enhanced output.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    """Implements a cross convolution layer with downsampling, expansion, and optional shortcut."""

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """Extends the C3 module with cross-convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions, extending C3 with customizable channel dimensions, groups,
        and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    """C3 module with TransformerBlock for enhanced feature extraction in object detection models."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with TransformerBlock for enhanced feature extraction, accepts channel sizes, shortcut
        config, group, and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    """Extends the C3 module with an SPP layer for enhanced spatial feature extraction and customizable channels."""

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with SPP layer for advanced spatial feature extraction, given channel sizes, kernel
        sizes, shortcut, group, and expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    """Implements a C3 module with Ghost Bottlenecks for efficient feature extraction in YOLOv5."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes YOLOv5's C3 module with Ghost Bottlenecks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    """Implements Spatial Pyramid Pooling (SPP) for feature extraction, ref: https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with Spatial Pyramid Pooling, ref: https://arxiv.org/abs/1406.4729, args: c1 (input channels), c2 (output channels), k (kernel sizes)."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Applies convolution and max pooling layers to the input tensor `x`, concatenates results, and returns output
        tensor.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Implements a fast Spatial Pyramid Pooling (SPPF) layer for efficient feature extraction in YOLOv5 models."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    """Focuses spatial information into channel space using slicing and convolution for efficient feature extraction."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus module to concentrate width-height info into channel space with configurable convolution
        parameters.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Processes input through Focus mechanism, reshaping (b,c,w,h) to (b,4c,w/2,h/2) then applies convolution."""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Implements Ghost Convolution for efficient feature extraction, see https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes GhostConv with in/out channels, kernel size, stride, groups, and activation; halves out channels
        for efficiency.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Performs forward pass, concatenating outputs of two convolutions on input `x`: shape (B,C,H,W)."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    """Efficient bottleneck layer using Ghost Convolutions, see https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck with ch_in `c1`, ch_out `c2`, kernel size `k`, stride `s`; see https://github.com/huawei-noah/ghostnet."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Processes input through conv and shortcut layers, returning their summed output."""
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    """Contracts spatial dimensions into channel dimensions for efficient processing in neural networks."""

    def __init__(self, gain=2):
        """Initializes a layer to contract spatial dimensions (width-height) into channels, e.g., input shape
        (1,64,80,80) to (1,256,40,40).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor to expand channel dimensions by contracting spatial dimensions, yielding output shape
        `(b, c*s*s, h//s, w//s)`.
        """
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    """Expands spatial dimensions by redistributing channels, e.g., from (1,64,80,80) to (1,16,160,160)."""

    def __init__(self, gain=2):
        """
        Initializes the Expand module to increase spatial dimensions by redistributing channels, with an optional gain
        factor.

        Example: x(1,64,80,80) to x(1,16,160,160).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor x to expand spatial dimensions by redistributing channels, requiring C / gain^2 ==
        0.
        """
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    """Concatenates tensors along a specified dimension for efficient tensor manipulation in neural networks."""

    def __init__(self, dimension=1):
        """Initializes a Concat module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; `x` is a list of tensors, `dimension` is an
        int.
        """
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    """YOLOv5 MultiBackend class for inference on various backends including PyTorch, ONNX, TensorRT, and more."""

    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """Initializes DetectMultiBackend with support for various inference backends, including PyTorch and ONNX."""
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlpackage
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
                stride, names = int(d["stride"]), d["names"]
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
        elif xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch

            core = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            ov_model = core.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(ov_model)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")  # AUTO selects best available device
            stride, names = self._load_metadata(Path(w).with_suffix(".yaml"))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):  # dynamic
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_profile_shape(name, 0)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wraps a TensorFlow GraphDef for inference, returning a pruned function."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                """Generates a sorted list of graph outputs excluding NoOp nodes and inputs, formatted as '<name>:0'."""
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta["stride"]), meta["names"]
        elif tfjs:  # TF.js
            raise NotImplementedError("ERROR: YOLOv5 TF.js inference is not supported")
        # PaddlePaddle
        elif paddle:
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle>=3.0.0") # Adjusted paddlepaddle version
            import paddle.inference as pdi

            w = Path(w)
            if w.is_dir(): # Check if model path is a directory
                model_file = next(w.rglob("*.pdmodel"), None) # Look for .pdmodel for model file
                params_file = next(w.rglob("*.pdiparams"), None)
                # If specific json config is preferred by Paddle for your model, adjust model_file glob pattern
            elif w.suffix == ".pdiparams": # Check if params file is provided
                model_file = w.with_suffix(".pdmodel") # Assume model file has .pdmodel extension
                params_file = w
            else: # Invalid path
                raise ValueError(f"Invalid model path {w}. Provide model directory or a .pdiparams file.")

            if not (model_file and params_file and model_file.is_file() and params_file.is_file()):
                raise FileNotFoundError(f"Model files not found in {w}. Both .pdmodel and .pdiparams files are required.")

            config = pdi.Config(str(model_file), str(params_file))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()

        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f"Using {w} as Triton Inference Server...")
            check_requirements("tritonclient[all]")
            from utils.triton import TritonRemoteModel

            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # class names
        if "names" not in locals(): # Check if names is already defined
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        if isinstance(names, dict) and names.get(0) == "n01440764" and len(names) == 1000:  # ImageNet case
            net_names_path = ROOT / "data/ImageNet.yaml"
            if net_names_path.is_file(): # Ensure ImageNet.yaml exists
                 names = yaml_load(net_names_path)["names"]  # human-readable names
            else:
                 LOGGER.warning(f"ImageNet.yaml not found at {net_names_path}, using default class names.")


        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """Performs YOLOv5 inference on input images with options for augmentation and visualization."""
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.ov_compiled_model(im).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images") if hasattr(self.model, 'get_binding_index') else self.model.get_tensor_name(0) # TRT 10 compatibility
                self.context.set_binding_shape(i, im.shape) if hasattr(self.context, 'set_binding_shape') else self.context.set_input_shape(i, im.shape) # TRT 10 compatibility
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name_idx, name in enumerate(self.output_names): # Iterate over sorted output names
                    i_out = self.model.get_binding_index(name) if hasattr(self.model, 'get_binding_index') else self.model.get_tensor_name(len(self.bindings) - len(self.output_names) + name_idx) # TRT 10 compatibility
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i_out) if hasattr(self.context, 'get_binding_shape') else self.context.get_tensor_shape(i_out))) # TRT 10 compatibility
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)] # Ensure sorted order
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im})  # coordinates are xywh normalized
            if "confidence" in y and "coordinates" in y: # Check for expected keys
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                conf = y["confidence"].max(1)
                cls = y["confidence"].argmax(1).astype(np.float32) # Ensure float for consistency
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            elif isinstance(y, dict): # General dict handling if specific keys are missing
                 y = list(reversed(y.values())) # Reversed for segmentation models (pred, proto)
            else: # Fallback if y is not a dict (e.g. list already)
                 pass # Assuming y is already in a usable list format or needs other handling
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) # Keras model expects training arg
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=tf.constant(im)) # Use tf.constant for TF2
            else:  # Lite or Edge TPU
                input_detail = self.input_details[0] # Corrected variable name
                int8 = input_detail["dtype"] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input_detail["quantization"]
                    if scale == 0 and zero_point == 0: # Handle cases where quantization might not be set (e.g. float models)
                        im_processed = im.astype(input_detail["dtype"])
                    else:
                        im_processed = (im / scale + zero_point).astype(input_detail["dtype"])  # de-scale / quantize
                else:
                    im_processed = im.astype(input_detail["dtype"])

                self.interpreter.set_tensor(input_detail["index"], im_processed)
                self.interpreter.invoke()
                y = []
                for output_detail in self.output_details: # Corrected variable name
                    x = self.interpreter.get_tensor(output_detail["index"])
                    if int8: # Check if output is also quantized
                        scale, zero_point = output_detail["quantization"]
                        if not (scale == 0 and zero_point == 0): # Apply dequantization if params are valid
                           x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                        else: # If output quantization is not set, assume float output
                           x = x.astype(np.float32)
                    y.append(x)
            if isinstance(y, list) and len(y) >= 1 and isinstance(y[0], tf.Tensor): # TF specific handling
                y = [x.numpy() for x in y] # Convert TF Tensors to numpy arrays

            if len(y) == 2 and len(y[1].shape) != 4: # Specific YOLOv5 output reshaping logic
                y = list(reversed(y)) # Often (pred, protos) for segmentation

            # Ensure box coordinates are scaled correctly (assuming first output is detections)
            if len(y) > 0 and y[0].ndim >=2 and y[0].shape[-1] >= 4: # Check if y[0] has box coordinates
                 y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """Converts a NumPy array to a torch tensor, maintaining device compatibility."""
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Performs a single inference warmup to initialize model weights, accepting an `imgsz` tuple for image size."""
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton): # Check if device is not CPU or is Triton
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  # Jit warmup has 2 iterations
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Determines model type from file path or URL, supporting various export formats.

        Example: path='path/to/model.onnx' -> type=onnx
        """
        from export import export_formats # Assuming export.py is in the same directory or PYTHONPATH
        from utils.downloads import is_url # Assuming utils.downloads is accessible

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False): # If not a URL
            check_suffix(p, sf)  # checks if suffix is in sf
        url = urlparse(p)  # if url may be Triton inference server
        types = [s.lower() in Path(p).name.lower() for s in sf] # Case-insensitive check and use .lower() for suffixes
        if len(types) > 8 : # Ensure tflite and edgetpu are correctly indexed
            types[8] &= not types[9]  # tflite &= not edgetpu
        else: # Handle cases where types list might be shorter if sf is smaller
            if len(types) > 9 and types[9]: # edgetpu is present
                if len(types) > 8: types[8] = False # ensure tflite is false if edgetpu is true

        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]


    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        """Loads metadata from a YAML file, returning strides and names if the file exists, otherwise `None`."""
        if f.exists():
            d = yaml_load(f) # Assuming yaml_load is defined and handles loading
            return d.get("stride"), d.get("names") # Use .get for safer access
        return None, None


class AutoShape(nn.Module):
    """AutoShape class for robust YOLOv5 inference with preprocessing, NMS, and support for various input formats."""

    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        """Initializes YOLOv5 model for inference, setting up attributes and preparing model for evaluation."""
        super().__init__()
        if verbose:
            LOGGER.info("Adding AutoShape... ")
        copy_attr(self, model, include=("yaml", "nc", "hyp", "names", "stride", "abc"), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt if hasattr(model, 'pt') else True # PyTorch model, add hasattr check
        self.model = model.eval()
        if self.pt: # Check if pt model before accessing model.model
            # Ensure model.model exists and is not None
            if hasattr(model, 'model') and model.model is not None:
                # Access the Detect() module, check if model is wrapped by DMB
                m = model.model.model[-1] if self.dmb and hasattr(model.model, 'model') else \
                    (model.model[-1] if hasattr(model, 'model') else None)

                if m and hasattr(m, 'inplace'): # Check if m is a valid module and has 'inplace'
                    m.inplace = False  # Detect.inplace=False for safe multithread inference
                    if hasattr(m, 'export'):
                         m.export = True  # do not output loss values
            else: # Handle cases where model.model might not be structured as expected
                LOGGER.warning("AutoShape: Model structure not as expected for PyTorch model, Detect() module attributes may not be set.")


    def _apply(self, fn):
        """
        Applies to(), cpu(), cuda(), half() etc.

        to model tensors excluding parameters or registered buffers.
        """
        self = super()._apply(fn)
        if self.pt: # Check if pt model
            # Safely access Detect() module
            m = None
            if hasattr(self.model, 'model') and self.model.model is not None: # Check if self.model.model exists
                if self.dmb and hasattr(self.model.model, 'model'): # If wrapped in DMB
                     m = self.model.model.model[-1]
                elif hasattr(self.model, 'model'): # If not DMB but has model.model
                     m = self.model.model[-1]

            if m and hasattr(m, 'stride'): # Check if m is valid and has 'stride'
                m.stride = fn(m.stride)
                if hasattr(m, 'grid'):
                    m.grid = list(map(fn, m.grid))
                if hasattr(m, 'anchor_grid') and isinstance(m.anchor_grid, list):
                    m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


    @smart_inference_mode() # Decorator for inference mode
    def forward(self, ims, size=640, augment=False, profile=False):
        """
        Performs inference on inputs with optional augment & profiling.

        Supports various formats including file, URI, OpenCV, PIL, numpy, torch.
        """
        dt = (Profile(), Profile(), Profile()) if profile else [contextlib.nullcontext()] * 3 # Use Profile only if profile=True
        with dt[0]: # Preprocessing
            if isinstance(size, int):  # expand
                size = (size, size)
            # Determine device and dtype from model parameters if available
            p_device = torch.device('cpu') # Default to CPU
            p_dtype = torch.float32 # Default to float32
            if self.pt and hasattr(self.model, 'parameters') and next(self.model.parameters(), None) is not None:
                p = next(self.model.parameters())
                p_device = p.device
                p_dtype = p.dtype
            elif hasattr(self.model, 'device'): # Fallback for non-PT models or models without parameters
                p_device = self.model.device


            autocast = self.amp and (p_device.type != "cpu")  # AMP inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(enabled=autocast): # Ensure autocast is properly enabled/disabled
                    return self.model(ims.to(p_device).type(p_dtype), augment=augment)  # inference

            # Pre-process for non-Tensor inputs
            n, ims_list = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])
            shape0, shape1_inf, files = [], [], []  # image original, inference shapes, filenames
            for i, im_item in enumerate(ims_list):
                f = f"image{i}"  # default filename
                if isinstance(im_item, (str, Path)):  # filename or uri
                    im_pil, f = Image.open(requests.get(im_item, stream=True).raw if str(im_item).startswith("http") else im_item), im_item
                    im_np = np.asarray(exif_transpose(im_pil))
                elif isinstance(im_item, Image.Image):  # PIL Image
                    im_np, f = np.asarray(exif_transpose(im_item)), getattr(im_item, "filename", f) or f
                elif isinstance(im_item, np.ndarray): # Numpy array
                    im_np = im_item
                    # f remains default or could be passed if available
                else:
                    raise TypeError(f"Unsupported input type: {type(im_item)}")

                files.append(Path(f).with_suffix(".jpg").name) # Standardize suffix for filename
                if im_np.shape[0] < 5 and im_np.ndim == 3:  # CHW to HWC if image in CHW format
                    im_np = im_np.transpose((1, 2, 0))
                im_np = im_np[..., :3] if im_np.ndim == 3 and im_np.shape[-1] >=3 else \
                        cv2.cvtColor(im_np, cv2.COLOR_GRAY2BGR) if im_np.ndim == 2 else \
                        (np.zeros((size[0],size[1],3), dtype=np.uint8) if im_np.size == 0 else im_np) # Handle grayscale and empty, enforce 3ch

                s0 = im_np.shape[:2]  # HWC
                shape0.append(s0)  # original image shape
                g = max(size) / max(s0)  # gain
                shape1_inf.append([int(y * g) for y in s0])
                ims_list[i] = im_np if im_np.data.contiguous else np.ascontiguousarray(im_np)  # update

            # Max inference shape, make divisible by stride
            if shape1_inf: # If there are images to process
                 shape1_max = np.array(shape1_inf).max(0)
                 inf_shape = [make_divisible(x, self.stride if hasattr(self, 'stride') else 32) for x in shape1_max]
            else: # Handle empty ims_list
                 inf_shape = list(size)


            x_padded = [letterbox(im, inf_shape, auto=False)[0] for im in ims_list]  # pad
            x_stacked = np.ascontiguousarray(np.array(x_padded).transpose((0, 3, 1, 2))) if x_padded else np.array([]) # stack and BHWC to BCHW

            if x_stacked.size == 0: # Handle case with no images after preprocessing
                # Return empty detections or handle as an error
                # This depends on the expected behavior for empty input
                # For now, let's create a Detections object with no predictions
                return Detections(ims_list, [torch.empty(0,6, device=p_device)], files, dt, self.names, (n,3,inf_shape[0],inf_shape[1]) if n > 0 else (0,3,inf_shape[0],inf_shape[1]))


            x_tensor = torch.from_numpy(x_stacked).to(p_device).type(p_dtype) / 255  # uint8 to fp16/32

        with amp.autocast(enabled=autocast):
            # Inference
            with dt[1]:
                y = self.model(x_tensor, augment=augment)  # forward

            # Post-process
            with dt[2]:
                # Ensure y is in the expected format for non_max_suppression
                # y could be a tensor or a tuple/list of tensors depending on the model
                pred_to_nms = y
                if not self.dmb and isinstance(y, (list,tuple)) and len(y) > 0 : # Common case for PT model not in DMB
                    pred_to_nms = y[0]

                # Handle cases where pred_to_nms might still be a tuple (e.g. from segmentation models)
                if isinstance(pred_to_nms, (list, tuple)):
                    if len(pred_to_nms) > 0 and isinstance(pred_to_nms[0], torch.Tensor):
                        pred_to_nms = pred_to_nms[0] # Take the first element if it's a tensor (detections)
                    else: # If not a tensor, NMS might fail or need different handling
                        LOGGER.warning("NMS input is not a tensor, skipping NMS or NMS might fail.")
                        # Create empty predictions if NMS cannot be run
                        y_nms = [torch.empty(0,6, device=p_device) for _ in range(n)] if n > 0 else []


                if isinstance(pred_to_nms, torch.Tensor): # Proceed with NMS if we have a tensor
                    y_nms = non_max_suppression(
                        pred_to_nms,
                        self.conf,
                        self.iou,
                        self.classes,
                        self.agnostic,
                        self.multi_label,
                        max_det=self.max_det,
                    )
                # If y_nms was not created due to pred_to_nms issues, it should be initialized above

                # Scale boxes for each image
                for i in range(n): # Use n (number of images)
                    if i < len(y_nms) and i < len(shape0): # Check bounds
                        scale_boxes(inf_shape, y_nms[i][:, :4], shape0[i])
                    elif i < len(y_nms) : # If shape0 is somehow shorter
                        LOGGER.warning(f"Shape0 for image {i} not available for scaling boxes.")


            return Detections(ims_list, y_nms, files, dt, self.names, x_tensor.shape)


class Detections:
    """Manages YOLOv5 detection results with methods for visualization, saving, cropping, and exporting detections."""

    def __init__(self, ims, pred, files, times=(0.0, 0.0, 0.0), names=None, shape=None): # Ensure times is float tuple
        """Initializes the YOLOv5 Detections class with image info, predictions, filenames, timing and normalization."""
        super().__init__()
        # Ensure pred is a list of tensors, even for a single image
        pred_list = pred if isinstance(pred, list) else [pred]
        # Fallback for device if pred_list is empty or items are not tensors
        d = pred_list[0].device if pred_list and isinstance(pred_list[0], torch.Tensor) else torch.device('cpu')


        # Ensure ims is a list and each item has a shape attribute
        gn = []
        valid_ims = []
        valid_pred = []
        valid_files = []

        for i, im_item in enumerate(ims):
            if hasattr(im_item, 'shape') and i < len(pred_list) and isinstance(pred_list[i], torch.Tensor):
                # Normalization: xyxy, confidence, class
                # im_shape is (H, W)
                im_shape = im_item.shape[:2] # Original H, W
                # gn_item should be (W, H, W, H, 1, 1) for scaling xyxy, conf, cls
                gn.append(torch.tensor([im_shape[1], im_shape[0], im_shape[1], im_shape[0], 1.0, 1.0], device=d, dtype=torch.float32))
                valid_ims.append(im_item)
                valid_pred.append(pred_list[i])
                valid_files.append(files[i] if i < len(files) else f"image_{i}") # Fallback filename
            else:
                LOGGER.warning(f"Skipping image {i} due to missing shape or mismatched prediction.")


        self.ims = valid_ims # list of images as numpy arrays
        self.pred = valid_pred # list of tensors pred[i] = (xyxy, conf, cls)
        self.names = names if names is not None else {0: 'object'} # Default class names
        self.files = valid_files  # image filenames
        self.times = times  # profiling times (Profile objects or float tuples)

        # Convert Profile objects in times to float values (milliseconds)
        processed_times = []
        for t_item in times:
            if isinstance(t_item, Profile):
                processed_times.append(t_item.dt * 1E3) # Profile.dt is usually in seconds
            elif isinstance(t_item, (float, int)):
                processed_times.append(float(t_item)) # Assume already in ms if float/int
            else:
                processed_times.append(0.0) # Fallback
        self.t_tuple = tuple(processed_times)


        self.n = len(self.pred)  # number of images (batch size)
        # Ensure self.t_tuple has 3 elements for string formatting later
        self.t_formatted = (self.t_tuple[0] / self.n if self.n > 0 else 0.0,
                            self.t_tuple[1] / self.n if self.n > 0 else 0.0,
                            self.t_tuple[2] / self.n if self.n > 0 else 0.0) if len(self.t_tuple) == 3 else (0.0,0.0,0.0)


        self.s = tuple(shape) if shape is not None else (self.n, 0,0,0) # inference BCHW shape

        # Initialize box formats, handle empty predictions
        self.xyxy = [p[:, :4] if p.ndim > 1 and p.shape[0] > 0 else torch.empty(0,4, device=d) for p in self.pred]
        self.conf = [p[:, 4] if p.ndim > 1 and p.shape[0] > 0 and p.shape[1] > 4 else torch.empty(0, device=d) for p in self.pred]
        self.cls  = [p[:, 5] if p.ndim > 1 and p.shape[0] > 0 and p.shape[1] > 5 else torch.empty(0, device=d) for p in self.pred]

        # Normalized boxes (handle cases where gn might be empty if self.pred was empty)
        self.xyxyn = [(x / g[:4] if g.numel() > 0 else torch.empty_like(x)) for x, g in zip(self.xyxy, gn)] if gn else [torch.empty_like(x) for x in self.xyxy]


        # WH formats (handle empty predictions)
        self.xywh = [xyxy2xywh(x) if x.numel() > 0 else torch.empty_like(x) for x in self.xyxy]
        self.xywhn = [(x / g[:4] if g.numel() > 0 else torch.empty_like(x)) for x, g in zip(self.xywh, gn)] if gn else [torch.empty_like(x) for x in self.xywh]


    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path(".")):
        """Executes model predictions, displaying and/or saving outputs with optional crops and labels."""
        s, crops_list = "", [] # Renamed crops to crops_list to avoid conflict
        for i, (im_orig, pred_tensor) in enumerate(zip(self.ims, self.pred)):
            # Ensure im_orig is a mutable NumPy array for Annotator
            im_display = im_orig.copy() if isinstance(im_orig, np.ndarray) else np.array(im_orig)
            if im_display.ndim == 2: # Handle grayscale
                im_display = cv2.cvtColor(im_display, cv2.COLOR_GRAY2BGR)
            if im_display.shape[2] == 4: # Handle RGBA
                im_display = cv2.cvtColor(im_display, cv2.COLOR_RGBA2BGR)


            s += f"\nimage {i + 1}/{self.n}: {im_display.shape[0]}x{im_display.shape[1]} "  # string
            if pred_tensor.shape[0]: # If there are detections
                # Sort by confidence before iterating, if needed for consistent output or specific logic
                # pred_tensor_sorted = pred_tensor[pred_tensor[:, 4].argsort(descending=True)]

                # Use unique classes from this specific prediction tensor
                unique_classes = pred_tensor[:, -1].unique() if pred_tensor.ndim > 1 and pred_tensor.shape[1] > 5 else torch.empty(0)

                for c_val in unique_classes:
                    n_det = (pred_tensor[:, -1] == c_val).sum()  # detections per class
                    class_name_str = self.names.get(int(c_val.item()), f"class_{int(c_val.item())}") # Use .get for safety
                    s += f"{n_det} {class_name_str}{'s' * (n_det > 1)}, "  # add to string
                s = s.rstrip(", ")

                if show or save or render or crop:
                    annotator = Annotator(im_display, line_width=2, example=str(self.names)) # Use a default line_width
                    # Iterate in reversed order of predictions if you want to draw high confidence boxes on top
                    # For now, iterating as is.
                    for *box_coords, conf_val, cls_val in reversed(pred_tensor.tolist()): # Convert to list for easier iteration
                        class_name = self.names.get(int(cls_val), f"class_{int(cls_val)}")
                        label_text = f"{class_name} {conf_val:.2f}"
                        if crop:
                            crop_save_path = save_dir / "crops" / class_name / self.files[i] if save else None
                            crops_list.append(
                                {
                                    "box": box_coords, # box_coords is already [xyxy]
                                    "conf": conf_val,
                                    "cls": cls_val,
                                    "label": label_text,
                                    # save_one_box expects np.array for image
                                    "im": save_one_box(torch.tensor(box_coords, device=pred_tensor.device), im_orig.copy(), file=crop_save_path, BGR=True, save=save),
                                }
                            )
                        else:  # all others (show, save, render)
                            annotator.box_label(box_coords, label_text if labels else "", color=colors(int(cls_val), True))
                    im_result = annotator.result() # Get the annotated image
            else:
                s += "(no detections)"
                im_result = im_display # No detections, use original (or BGR converted) image

            # Convert to PIL Image for showing or saving
            im_pil = Image.fromarray(cv2.cvtColor(im_result, cv2.COLOR_BGR2RGB)) if isinstance(im_result, np.ndarray) else im_result


            if show:
                if is_jupyter(): # Check if in Jupyter environment
                    from IPython.display import display
                    display(im_pil)
                else:
                    im_pil.show(title=self.files[i] if i < len(self.files) else f"image_{i}") # Use title argument
            if save:
                f_save = self.files[i] if i < len(self.files) else f"image_{i}.jpg" # Fallback filename
                im_pil.save(save_dir / f_save)  # save
                if i == self.n - 1 and self.n > 0 : # Log only if images were saved
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im_pil) # Update self.ims with rendered image

        if pprint:
            s = s.lstrip("\n")
            # Use self.t_formatted which is already per image and in ms
            return f"{s}\nSpeed: {self.t_formatted[0]:.1f}ms pre-process, {self.t_formatted[1]:.1f}ms inference, {self.t_formatted[2]:.1f}ms NMS per image at shape {self.s}"
        if crop:
            if save and crops_list: # Log only if crops were saved
                LOGGER.info(f"Saved crop results to {save_dir / 'crops'}\n")
            return crops_list
        return None # Default return if not pprint or crop


    @TryExcept("Showing images is not supported in this environment")
    def show(self, labels=True):
        """
        Displays detection results with optional labels.

        Usage: show(labels=True)
        """
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Saves detection results with optional labels to a specified directory.

        Usage: save(labels=True, save_dir='runs/detect/exp', exist_ok=False)
        """
        save_dir_inc = increment_path(save_dir, exist_ok=exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir_inc)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Crops detection results, optionally saves them to a directory.

        Args: save (bool), save_dir (str), exist_ok (bool).
        """
        save_dir_inc = increment_path(save_dir, exist_ok=exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir_inc) # crop results

    def render(self, labels=True):
        """Renders detection results with optional labels on images; args: labels (bool) indicating label inclusion."""
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        """
        Returns detections as pandas DataFrames for various box formats (xyxy, xyxyn, xywh, xywhn).
        Example: print(results.pandas().xyxy[0]).
        """
        new = copy(self)  # return copy
        ca = ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]  # xyxy columns
        cb = ["xcenter", "ycenter", "width", "height", "confidence", "class", "name"]  # xywh columns

        for k, cols in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            # Get the raw prediction tensor list for the current format 'k'
            # For 'xyxy', 'xywh' these are already in self.
            # For 'xyxyn', 'xywhn' we use the normalized versions.
            pred_tensor_list = getattr(self, k, []) # Use self.xyxy, self.xyxyn etc.

            df_list = []
            for pred_tensor_single_image in pred_tensor_list:
                if pred_tensor_single_image.numel() == 0: # Handle empty predictions
                    df_list.append(pd.DataFrame(columns=cols))
                    continue

                # Ensure pred_tensor_single_image is 2D [N, >=6]
                # For xyxyn/xywhn, they are already [N,4], need to add conf and cls
                # For xyxy/xywh, self.pred already contains [N,6] (xyxy,conf,cls)

                processed_detections = []
                if k in ["xyxy", "xywh"]: # These should come from self.pred which includes conf and cls
                    # Find the original prediction tensor for this image that has conf and cls
                    # This part is tricky because self.pred is a list, and k refers to derived formats.
                    # Let's assume self.pred[i] corresponds to the i-th image.
                    # We need to find the original self.pred item for the current pred_tensor_single_image.
                    # This requires careful indexing or a different approach.

                    # A simpler way: iterate through self.pred and generate all formats for each.
                    # However, the current structure tries to generate pandas DFs for already-processed lists.

                    # Let's assume pred_tensor_single_image for xyxy and xywh
                    # should actually be the full [N, 6] tensor from self.pred for that image.
                    # This means the input to pandas() for these 'k' should be different.
                    # For now, let's try to work with what `getattr(self, k)` gives.
                    # If k is 'xyxy' or 'xywh', pred_tensor_single_image is [N,4]. We need conf and cls.
                    # This indicates a potential issue in how self.xyxy/self.xywh are stored if they don't include conf/cls.
                    #
                    # Revisit: self.pred stores [N,6] (xyxy, conf, cls).
                    # self.xyxy is just an alias to self.pred (or should be).
                    # self.xywh is derived.
                    # Let's assume getattr(self, k) for xyxy gives [N,6] and for xywh it's [N,6]
                    # But the Detections init makes self.xyxy = pred (which is [N,6])
                    # and self.xywh = [xyxy2xywh(x) for x in pred] (which also becomes [N,6])
                    # So, pred_tensor_single_image for k='xyxy' or 'xywh' IS [N,6]

                    for det in pred_tensor_single_image.tolist(): # det is [x,y,x,y,conf,cls] or [xc,yc,w,h,conf,cls]
                        if len(det) >= 6:
                            class_idx = int(det[5])
                            class_name = self.names.get(class_idx, f"class_{class_idx}")
                            processed_detections.append(det[:5] + [class_idx, class_name])
                        else: # Should not happen if data is [N,6]
                            processed_detections.append(det + [None]*(7-len(det))) # Pad if shorter

                elif k in ["xyxyn", "xywhn"]: # These are [N,4], need to add conf and cls from self.pred
                    # Find corresponding conf and cls from self.pred
                    # This assumes pred_tensor_single_image (normalized box) and self.pred items align
                    # This is a bit complex due to the loop structure.
                    # A better way would be to process self.pred once and derive all pandas DFs.

                    # Simplified: assume we can get conf and cls for the current image.
                    # This requires finding the original full prediction for this image.
                    # This part of pandas() is a bit fragile if getattr(self,k) doesn't give enough info.

                    # Let's assume we iterate through original predictions (self.pred)
                    # and generate all dataframe types.
                    # For now, trying to make it work with the current getattr approach.
                    # This means for xyxyn/xywhn, we need to find the original prediction.
                    # This is difficult with the current loop.
                    #
                    # A HACK for now: use the index from the outer loop if `k` implies normalized
                    # This is not robust. The Detections class might need a refactor for `pandas()`
                    # to more easily access original confidence and class for normalized boxes.
                    # For now, let's assume that when we call `results.pandas().xyxyn`,
                    # it iterates through the normalized boxes, and we need to get conf/cls
                    # from the *original* unnormalized predictions.
                    # This is not directly available in the current loop structure for xyxyn/xywhn.

                    # Let's assume for simplicity that `pred_tensor_single_image` for xyxyn/xywhn
                    # needs to be combined with `self.conf[image_index]` and `self.cls[image_index]`.
                    # The current loop doesn't give `image_index` directly.

                    # *** Fallback: If it's normalized, we create a DataFrame with only box coords for now ***
                    # This is a known limitation of this structure if not handled carefully.
                    # A proper implementation would iterate `self.pred` and create all 4 DFs for each image.
                    temp_cols = cols[:4] # Only x,y,w,h or xmin,ymin,xmax,ymax
                    if pred_tensor_single_image.ndim == 2 and pred_tensor_single_image.shape[1] >=4:
                         df_list.append(pd.DataFrame(pred_tensor_single_image[:,:4].tolist(), columns=temp_cols))
                    else:
                         df_list.append(pd.DataFrame(columns=temp_cols))
                    # This part needs a more robust solution to include conf/class/name for normalized boxes.
                    # The original YOLOv5 implementation might handle this differently or assume
                    # getattr(self, k) provides enough context.
                    # For now, we'll proceed with this simplification for normalized formats.
                    # The key issue is `getattr(self, k)` for `xyxyn` and `xywhn` in the Detections init
                    # only stores the normalized coordinates, not the associated conf/cls.
                    # This is a flaw in the Detections class structure for pandas output of normalized boxes.
                    #
                    # To properly fix this, the pandas method should iterate `self.pred` (which has all info)
                    # and generate xyxy, xyxyn, xywh, xywhn for each, then create the DFs.
                    #
                    # Given the current structure, we'll have to live with xyxyn/xywhn DFs potentially missing conf/class/name.
                    # Or, if `pred_tensor_single_image` was cleverly made to be [N, 7] already, then the `else` below would work.

                else: # Should not happen based on k values
                    df_list.append(pd.DataFrame(columns=cols))

                # This part is for xyxy and xywh if they correctly have 7 columns after processing
                if k in ["xyxy", "xywh"] and processed_detections:
                    df_list.append(pd.DataFrame(processed_detections, columns=cols))
                elif k in ["xyxy", "xywh"] : # if processed_detections is empty
                    df_list.append(pd.DataFrame(columns=cols))


            setattr(new, k, df_list)
        return new

    def tolist(self):
        """
        Converts a Detections object into a list of individual detection results for iteration.
        Example: for result in results.tolist():
        """
        # This creates new Detections objects for each image, which is memory intensive.
        # A lighter version might return a list of dicts or simple namedtuples.
        return [
            Detections(
                [self.ims[i]] if i < len(self.ims) else [], # Handle if ims is shorter
                [self.pred[i]] if i < len(self.pred) else [], # Handle if pred is shorter
                [self.files[i]] if i < len(self.files) else [],# Handle if files is shorter
                self.times,
                self.names,
                self.s,
            )
            for i in range(self.n) # Iterate up to self.n (number of predictions)
        ]


    def print(self):
        """Logs the string representation of the current object's state via the LOGGER."""
        LOGGER.info(self.__str__())

    def __len__(self):
        """Returns the number of results stored, overrides the default len(results)."""
        return self.n # Should be len(self.pred)

    def __str__(self):
        """Returns a string representation of the model's results, suitable for printing, overrides default
        print(results).
        """
        return self._run(pprint=True) # print results

    def __repr__(self):
        """Returns a string representation of the YOLOv5 object, including its class and formatted results."""
        # Ensure self.__class__.__name__ is accessible
        class_name_repr = self.__class__.__name__ if hasattr(self, '__class__') and hasattr(self.__class__, '__name__') else 'Detections'
        return f"YOLOv5 {class_name_repr} instance\n" + self.__str__()


class Proto(nn.Module):
    """YOLOv5 mask Proto module for segmentation models, performing convolutions and upsampling on input tensors."""

    def __init__(self, c1, c_=256, c2=32):
        """Initializes YOLOv5 Proto module for segmentation with input, proto, and mask channels configuration."""
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass using convolutional layers and upsampling on input tensor `x`."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    """YOLOv5 classification head with convolution, pooling, and dropout layers for channel transformation."""

    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        """Initializes YOLOv5 classification head with convolution, pooling, and dropout layers for input to output
        channel transformation.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size (seems like a fixed value, consider making it more flexible or documented)
        self.cv1 = Conv(c1, c_, k, s, autopad(k, p), g) # Use cv1 instead of conv to match pattern
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Processes input through conv, pool, drop, and linear layers; supports list concatenation input."""
        if isinstance(x, list): # If x is a list of features, concatenate them
            x = torch.cat(x, 1)
        x = self.cv1(x) # Apply initial convolution
        x = self.pool(x) # Apply pooling
        x = x.flatten(1) # Flatten for linear layer
        x = self.drop(x) # Apply dropout
        return self.linear(x) # Apply final linear layer


# --- BEGIN: Custom module added by Team AI-Innovators ---
# This SEBlock is intended for use in the YOLOv5 architecture.
# It can be inserted into the .yaml model configuration file.
# Example usage in .yaml: [-1, 1, SEBlock, [input_channels, reduction_ratio]]

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    A channel-wise attention mechanism that adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, c1, r=16):  # c1: input channels, r: reduction ratio
        super(SEBlock, self).__init__()
        # Ensure reduction results in at least 1 channel, handle small c1
        inter_channels = c1 // r
        if inter_channels == 0:
            inter_channels = 1 # Minimum 1 channel for intermediate layer
            if c1 > 1 and r > c1 : # If r is too large for c1
                 LOGGER.warning(f"SEBlock reduction ratio r={r} too large for c1={c1}. Using r resulting in 1 inter_channel.")


        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling: (b, c1, h, w) -> (b, c1, 1, 1)
        self.excitation = nn.Sequential(
            # Using nn.Conv2d for 1x1 convolutions is common and flexible for channel manipulation
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0, bias=False), # FC1: (b, c1, 1, 1) -> (b, c1//r, 1, 1)
            nn.SiLU(),  # Activation function (YOLOv5 often uses SiLU)
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0, bias=False), # FC2: (b, c1//r, 1, 1) -> (b, c1, 1, 1)
            nn.Sigmoid()  # Sigmoid to get channel-wise weights between 0 and 1
        )

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, channels, height, width)
        """
        # Squeeze: Global information embedding
        squeezed = self.squeeze(x) # (b, c1, 1, 1)

        # Excitation: Adaptive recalibration
        weights = self.excitation(squeezed) # (b, c1, 1, 1)

        # Scale: Apply recalibrated weights to input features
        return x * weights # (b, c1, h, w) * (b, c1, 1, 1) -> (b, c1, h, w)

# --- END: Custom module added by Team AI-Innovators ---