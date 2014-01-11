#------------------------------------------------------------------------
# BLZ Namespace
#------------------------------------------------------------------------

from __future__ import absolute_import

min_numexpr_version = '2.2'  # the minimum version of Numexpr needed
numexpr_here = False
try:
    import numexpr
except ImportError:
    pass
else:
    if numexpr.__version__ >= min_numexpr_version:
        numexpr_here = True

# Print array functions (copied from NumPy)
from .arrayprint import (
    array2string, set_printoptions, get_printoptions)

from .blz_ext import (
    barray, blosc_version, blosc_compressor_list,
     _blosc_set_nthreads as blosc_set_nthreads,
    _blosc_init, _blosc_destroy,
     )
from .btable import btable
from .vtable import vtable
from .chunked_eval import eval, defaults
from .bfuncs import (
    open, zeros, ones, fill, arange, fromiter, iterblocks, whereblocks, walk)
from .bparams import bparams
from .version import __version__
from .tests import test, print_versions


def detect_number_of_cores():
    """
    detect_number_of_cores()

    Detect the number of cores in this system.

    Returns
    -------
    out : int
        The number of cores in this system.

    """
    import os
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default


# Initialization code for the Blosc library
_blosc_init()
ncores = detect_number_of_cores()
blosc_set_nthreads(ncores)
import atexit
atexit.register(_blosc_destroy)
