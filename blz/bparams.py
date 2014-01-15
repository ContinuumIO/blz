from __future__ import absolute_import

"""
bparams

configuration parameters for barray
"""

from blz import blosc_compressor_list

class bparams(object):
    """
    bparams(clevel=5, shuffle=True, cname=b"blosclz")

    Class to host parameters for compression and other filters.

    Parameters
    ----------
    clevel : int (0 <= clevel < 10)
        The compression level.
    shuffle : bool
        Whether the shuffle filter is active or not.
    cname : string ('blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', others?)
        Select the compressor to use inside Blosc.

    Notes
    -----
    The shuffle filter may be automatically disabled in case it makes
    non-sense to use it (e.g. itemsize == 1).

    """

    @property
    def clevel(self):
        """The compression level."""
        return self._clevel

    @property
    def shuffle(self):
        """Shuffle filter is active?"""
        return self._shuffle

    @property
    def cname(self):
        """The compressor name."""
        return self._cname

    def __init__(self, clevel=5, shuffle=True, cname="blosclz"):
        if not isinstance(clevel, int):
            raise ValueError("`clevel` must an int.")
        if not isinstance(shuffle, (bool, int)):
            raise ValueError("`shuffle` must a boolean.")
        shuffle = bool(shuffle)
        if clevel < 0:
            raise ValueError("clevel must be a positive integer")
        self._clevel = clevel
        self._shuffle = shuffle
        list_cnames = blosc_compressor_list()
        # Store the cname as bytes object internally
        if hasattr(cname, 'encode'):
            cname = cname.encode()
        if cname not in list_cnames:
            raise ValueError(
                "Compressor '%s' is not available in this build" % cname)
        self._cname = cname

    def __repr__(self):
        args = ["clevel=%d"%self._clevel,
                "shuffle=%s"%self._shuffle,
                "cname=%s"%self._cname,
                ]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(args))

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 78
