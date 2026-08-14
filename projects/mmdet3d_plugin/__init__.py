from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
  NormalizeMultiviewImage,  CustomCollect3D)
from .models.utils import *
from .models.opt.adamw import AdamW2
from .bevformer import *

# DD3D is only required by BEVFormerV2 configs. Keep the base BEVFormer
# plugin usable when its optional Detectron2 dependency is not installed.
try:
    from .dd3d import *
except ModuleNotFoundError as exc:
    if exc.name != 'detectron2':
        raise
