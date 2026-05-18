"""Volume analysis — spike detection and relative volume confirmation."""
from .confirmation import (
    VolumeContext,
    compute_volume_context,
    is_volume_spike,
    displacement_has_volume,
)

__all__ = [
    "VolumeContext",
    "compute_volume_context",
    "is_volume_spike",
    "displacement_has_volume",
]
