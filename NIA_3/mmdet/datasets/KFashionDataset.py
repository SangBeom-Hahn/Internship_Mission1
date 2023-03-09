from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module(force=True)
class KFashionDataset(CocoDataset):
    CLASSES = ('blouse', 'cardigan', 'coat', 'jacket', 'jumper', 'shirt', 'sweater', 't-shirt', 'vest', 'pants', 'skirt', 'onepiece(dress)', 'onepiece(jumpsuite)')
    # CLASSES = ('blouse', 'cardigan', 'coat', 'jacket', 'jumper', 'shirt')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 20, 32)]