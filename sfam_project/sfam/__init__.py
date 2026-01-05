# 1. Expose the Core Engine
from .models.sfam_net import SFAM

# 2. Expose the Feature Managers (This allows 'from sfam import image_fm')
from .modalities import image_fm
from .modalities import gesture_fm

__version__ = "1.2.4"