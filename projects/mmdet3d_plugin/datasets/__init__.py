from .nuscenes_dataset import CustomNuScenesDataset
from .builder import custom_build_dataset

__all__ = ['CustomNuScenesDataset']

try:
    from .nuscenes_dataset_v2 import CustomNuScenesDatasetV2
except ModuleNotFoundError as exc:
    if exc.name != 'detectron2':
        raise
else:
    __all__.append('CustomNuScenesDatasetV2')
