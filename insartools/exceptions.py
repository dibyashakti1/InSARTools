"""
Custom exceptions for InSARTools.
"""


class InSARToolsError(Exception):
    """Base exception."""


class UnsupportedSensorError(InSARToolsError):
    """Unsupported sensor."""


class UnsupportedProcessorError(InSARToolsError):
    """Unsupported processor."""


class InvalidRasterError(InSARToolsError):
    """Invalid raster."""
