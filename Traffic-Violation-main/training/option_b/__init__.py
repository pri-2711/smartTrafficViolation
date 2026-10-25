"""Option B: our own detector. Import the pieces from their own modules."""
from .model import OptionBDetector, flatten_outputs
from .grids import scale_ranges_for
from .loss import detection_loss
from .postprocess import postprocess

__all__ = ["OptionBDetector", "flatten_outputs", "scale_ranges_for", "detection_loss", "postprocess"]
