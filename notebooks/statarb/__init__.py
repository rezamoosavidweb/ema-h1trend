"""FX statistical-arbitrage research and execution toolkit."""

# TODO: stat_arb/identity/detector.py is not implemented yet — re-enable
# once SyntheticIdentityDetector exists.
# from stat_arb.identity.detector import SyntheticIdentityDetector
from notebooks.statarb.identity.results import IdentityResult, RejectionReason
from notebooks.statarb.config import DetectorConfig

__all__ = [
    "IdentityResult",
    "RejectionReason",
    "DetectorConfig",
]

__version__ = "0.1.0"
