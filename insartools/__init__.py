"""
InSARTools

A Python toolkit for InSAR deformation analysis.
"""
"""
InSARTools
==========

Python tools for processing, visualizing and exporting InSAR products.

Author
------
Dibyashakti Panda
"""

from .version import __version__

# Core modules
from . import amplitude
from . import export
from . import geocode
from . import io
from . import metadata
from . import phase
from . import plot
from . import plotting
from . import sensors
from . import unwrapped
from . import wrapped
from . import coherence
from . import displacement

__all__ = [
    "__version__",
    "amplitude",
    "export",
    "geocode",
    "io",
    "metadata",
    "phase",
    "plot",
    "plotting",
    "sensors",
    "unwrapped",
    "wrapped",
    "coherence",
    "displacement",
]
