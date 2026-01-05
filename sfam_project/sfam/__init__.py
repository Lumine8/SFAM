# ==========================================
# Core Engines
# ==========================================
# This import is critical, so we leave it outside the try-except
from .models.sfam_net import SFAM, SFAM_Adaptive, generate_user_key

# ==========================================
# Feature Managers (Optional Load)
# ==========================================
try:
    from .modalities import image_fm
    from .modalities import gesture_fm
except ImportError:
    # If these modules don't exist yet, we just set them to None
    # This prevents the "ImportError" from crashing your whole app
    image_fm = None
    gesture_fm = None

__all__ = [
    "SFAM_Adaptive",
    "generate_user_key",
    "image_fm",
    "gesture_fm",
]

__version__ = "1.2.6"
__author__ = "lumine8"
__license__ = "MIT" 
__copyright__ = "Copyright 2025 Lumine8"