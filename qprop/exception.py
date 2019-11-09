"""Collection of Custom Exception classes"""

class UnexpectedQpropDim(Exception):
    """Raised when an unexpected Qprop dimension is given.
    The 'dimension' of Qprop as of version 2.0 or lower
    includes either `34` for linearly polarized pulse 
    and `44` for elliptically polarized pulse."""

