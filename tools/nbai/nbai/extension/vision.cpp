// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <torch/script.h>
//#include "ROIAlignRotated/ROIAlignRotated.h"
#include "box_iou_rotated/box_iou_rotated.h"
#include "nms_rotated/nms_rotated.h"

static auto registry = 
    torch::RegisterOperators()
        .op("nbai::nms_rotated", &nms_rotated);
