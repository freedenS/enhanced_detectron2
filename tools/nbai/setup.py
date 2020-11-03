#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from torch.utils.hipify import hipify_python

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 4], "Requires PyTorch >= 1.4"

def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "nbai", "extension")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    is_rocm_pytorch = False
    if torch_ver >= [1, 5]:
        from torch.utils.cpp_extension import ROCM_HOME

        is_rocm_pytorch = (
            True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
        )

    if is_rocm_pytorch:
        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes="/nbai/extension/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )

        # Current version of hipify function in pytorch creates an intermediate directory
        # named "hip" at the same level of the path hierarchy if a "cuda" directory exists,
        # or modifying the hierarchy, if it doesn't. Once pytorch supports
        # "same directory" hipification (PR pendeing), the source_cuda will be set
        # similarly in both cuda and hip paths, and the explicit header file copy
        # (below) will not be needed.
        source_cuda = glob.glob(path.join(extensions_dir, "**", "hip", "*.hip")) + glob.glob(
            path.join(extensions_dir, "hip", "*.hip")
        )

        shutil.copy(
            "nbai/extension/box_iou_rotated/box_iou_rotated_utils.h",
            "nbai/extension/box_iou_rotated/hip/box_iou_rotated_utils.h",
        )

    else:
        source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
            path.join(extensions_dir, "*.cu")
        )

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "nbai",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="nbai",
    version='0.1',
    author="neurobot",
    url="https://www.nb-ai.com",
    description="nbai for new op",
    packages=find_packages(exclude=("backup",)),
    python_requires=">=3.6",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
