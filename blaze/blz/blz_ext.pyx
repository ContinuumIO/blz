#########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted -  francesc@continuum.io
#
########################################################################


import operator
import sys
import blaze.blz as blz
from blaze.blz import utils, attrs, array2string
import os, os.path
import struct
import shutil
import tempfile
import json
import datetime
import cython
import dynd
from dynd import nd, ndt, _lowlevel

_KB = 1024
_MB = 1024*_KB

# Directories for saving the data and metadata for BLZ persistency
DATA_DIR = 'data'
META_DIR = 'meta'
SIZES_FILE = 'sizes'
STORAGE_FILE = 'storage'

# For the persistence layer
EXTENSION = '.blp'
MAGIC = b'blpk'
BLOSCPACK_HEADER_LENGTH = 16
BLOSC_HEADER_LENGTH = 16
FORMAT_VERSION = 1
MAX_FORMAT_VERSION = 255
MAX_CHUNKS = (2**63)-1

#-----------------------------------------------------------------

cimport blosc
cimport cpython

from libc.stdint cimport intptr_t, uintptr_t
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memset

ctypedef intptr_t blz_int_t


cdef extern from *:
  ctypedef unsigned long Py_uintptr_t

# ----------------------------------------------------------------------

# NumPy related imports and initialization
import numpy as np
cimport numpy_definitions as npdefs

if sys.version_info >= (3, 0):
    _MAXINT = 2**31 - 1
    _inttypes = (int, np.integer)
else:
    _MAXINT = sys.maxint
    _inttypes = (int, long, np.integer)

# The type used for size values: indexes, coordinates, dimension
# lengths, row numbers, shapes, chunk shapes, byte counts...
# XXX Replace this by ndt.intptr as soon it is supported
SizeType = ndt.int64

# The native int type for this platform
#IntType = np.dtype(np.int_)
# XXX Replace this by ndt.intptr as soon it is supported
IntType = ndt.int64

# The numpy API requires this function to be called before
# using any numpy facilities in an extension module.
npdefs.import_array()

#-------------------------------------------------------------

# Some utilities
def _blosc_set_nthreads(nthreads):
  """
  _blosc_set_nthreads(nthreads)

  Sets the number of threads that Blosc can use.

  Parameters
  ----------
  nthreads : int
      The desired number of threads to use.

  Returns
  -------
  out : int
      The previous setting for the number of threads.

  """
  return blosc.set_nthreads(nthreads)

def _blosc_init():
  """
  _blosc_init()

  Initialize the Blosc library.

  """
  blosc.init()

def _blosc_destroy():
  """
  _blosc_destroy()

  Finalize the Blosc library.

  """
  blosc.destroy()

def blosc_version():
  """
  blosc_version()

  Return the version of the Blosc library.

  """
  return (<char *>blosc.BLOSC_VERSION_STRING,
          <char *>blosc.BLOSC_VERSION_DATE)

def list_bytes_to_str(lst):
    """The Python 3 JSON encoder doesn't accept 'bytes' objects,
    this utility function converts all bytes to strings.
    """
    if isinstance(lst, bytes):
        return lst.decode('ascii')
    elif isinstance(lst, list):
        return [list_bytes_to_str(x) for x in lst]
    else:
        return lst

# This is the same than in utils.py, but works faster in extensions
cdef get_len_of_range(blz_int_t start, blz_int_t stop, blz_int_t step):
  """Get the length of a (start, stop, step) range."""
  cdef blz_int_t n

  n = 0
  if start < stop:
    # Do not use a cython.cdiv here (do not ask me why!)
    n = ((stop - start - 1) // step + 1)
  return n

cdef clip_chunk(blz_int_t nchunk,
                blz_int_t chunklen,
                blz_int_t start,
                blz_int_t stop,
                blz_int_t step):
  """Get the limits of a certain chunk based on its length."""
  cdef blz_int_t startb, stopb, blen, distance

  startb = start - nchunk * chunklen
  stopb = stop - nchunk * chunklen

  # Check limits
  if (startb >= chunklen) or (stopb <= 0):
    return startb, stopb, 0   # null size
  if startb < 0:
    startb = 0
  if stopb > chunklen:
    stopb = chunklen

  # step corrections
  if step > 1:
    # Just correcting startb is enough
    distance = (nchunk * chunklen + startb) - start
    if distance % step > 0:
      startb += (step - (distance % step))
      if startb > chunklen:
        return startb, stopb, 0  # null size

  # Compute size of the clipped block
  blen = get_len_of_range(startb, stopb, step)

  return startb, stopb, blen

cdef int check_zeros(char *data, int nbytes):
  """Check whether [data, data+nbytes] is zero or not."""
  cdef int i, iszero, chunklen, leftover
  cdef size_t *sdata

  iszero = 1
  sdata = <size_t *>data
  chunklen = cython.cdiv(nbytes, sizeof(size_t))
  leftover = nbytes % sizeof(size_t)
  with nogil:
    for i from 0 <= i < chunklen:
      if sdata[i] != 0:
        iszero = 0
        break
    else:
      data += nbytes - leftover
      for i from 0 <= i < leftover:
        if data[i] != 0:
          iszero = 0
          break
  return iszero

cdef int true_count(char *data, int nbytes):
  """Count the number of true values in data (boolean)."""
  cdef int i, count

  with nogil:
    count = 0
    for i from 0 <= i < nbytes:
      count += <int>(data[i])
  return count

#-------------------------------------------------------------

kinds = {'bool': 'b', 'int': 'i', 'real': 'r', 'complex': 'c',
         'bytes': 'B', 'string': 'U', 'struct': 's'}

cdef class chunk:
  """
  chunk(array, atom, bparams)

  Compressed in-memory container for a data chunk.

  This class is meant to be used only by the `barray` class.

  """
  cdef char typekind, isconstant
  cdef public int atomsize, itemsize, blocksize
  cdef public int nbytes, cbytes, cdbytes
  cdef int true_count
  cdef char *data
  cdef object atom, constant, dobject

  cdef void _getitem(self, int start, int stop, char *dest)
  cdef compress_data(self, char *data, size_t itemsize, size_t nbytes,
                     object bparams)
  cdef compress_arrdata(self, object array, int itemsize,
                        object bparams, object _memory)

  property dtype:
    "The atom for this chunk."
    def __get__(self):
      return self.atom

  def __cinit__(self, object dobject, object atom, object bparams,
                object _memory=True, object _compr=False):
    cdef int itemsize, footprint
    cdef size_t nbytes, cbytes, blocksize
    cdef object dtype_
    cdef char *data

    self.atom = atom
    self.atomsize = atom.data_size
    dtype_ = atom.dtype
    self.typekind = ord(kinds[dtype_.kind])

    if self.typekind == ord('B'):
      itemsize = 1
    elif self.typekind == ord('U'):
      itemsize = 4
    else:
      itemsize = dtype_.data_size
    # Temporary hack for allowing strings with len > BLOSC_MAX_TYPESIZE
    # In the future DyND should offer more flexibility for coping
    # with strings.
    if itemsize > blosc.BLOSC_MAX_TYPESIZE:
      raise TypeError(
        "typesize is %d and BLZ does not currently support data types larger "
        "than %d bytes" % (itemsize, blosc.BLOSC_MAX_TYPESIZE))
    self.itemsize = itemsize
    self.dobject = None
    footprint = 0

    if _compr:
      # Data comes in an already compressed state inside a Python bytes object
      self.data = dobject
      # Increment the reference so that data don't go away
      self.dobject = dobject
      # Set size info for the instance
      blosc.cbuffer_sizes(self.data, &nbytes, &cbytes, &blocksize)
    else:
      # Compress the data object (a NumPy object)
      nbytes, cbytes, blocksize, footprint = self.compress_arrdata(
        dobject, itemsize, bparams, _memory)
    footprint += 128  # add the (aprox) footprint of this instance in bytes

    # Fill instance data
    self.nbytes = nbytes
    self.cbytes = cbytes + footprint
    self.cdbytes = cbytes
    self.blocksize = blocksize

  cdef compress_arrdata(self, object array, int itemsize,
                       object bparams, object _memory):
    """Compress data in `array` and put it in ``self.data``"""
    cdef size_t nbytes, cbytes, blocksize, footprint #, itemsize
    cdef char* data

    # Compute the total number of bytes in this array
    nbytes = np.prod(array.shape) * itemsize
    data = <char *><Py_uintptr_t>_lowlevel.data_address_of(array)
    cbytes = 0
    footprint = 0

    # Check whether incoming data can be expressed as a constant or not.
    # Disk-based chunks are not allowed to do this.
    self.isconstant = 0
    self.constant = None
    if _memory and (array.strides[0] == 0
                    or check_zeros(data, nbytes)):

      self.isconstant = 1
      self.constant = array[0]
      # Add overhead (64 bytes for the overhead of the numpy container)
      footprint += 64 + nd.dtype_of(self.constant).data_size

    if self.isconstant:
      blocksize = 4*1024  # use 4 KB as a cache for blocks
      # Make blocksize a multiple of itemsize
      if blocksize % itemsize > 0:
        blocksize = cython.cdiv(blocksize, itemsize) * itemsize
      # Correct in case we have a large itemsize
      if blocksize == 0:
        blocksize = itemsize
    else:
      if self.typekind == ord('b'):
        self.true_count = true_count(data, nbytes)

      if array.strides[0] == 0:
        # The chunk is made of constants.  Regenerate the actual data.
        array = array.copy()

      # Compress data
      cbytes, blocksize = self.compress_data(data, itemsize, nbytes,
                                             bparams)

    return (nbytes, cbytes, blocksize, footprint)

  cdef compress_data(self, char *data, size_t itemsize, size_t nbytes,
                     object bparams):
    """Compress data with `caparms` and return metadata."""
    cdef size_t nbytes_, cbytes, blocksize
    cdef int clevel, shuffle
    cdef char *dest

    clevel = bparams.clevel
    shuffle = bparams.shuffle
    dest = <char *>malloc(nbytes + blosc.BLOSC_MAX_OVERHEAD)
    with nogil:
      cbytes = blosc.compress(clevel, shuffle, itemsize, nbytes, data,
                              dest, nbytes + blosc.BLOSC_MAX_OVERHEAD)
    if cbytes <= 0:
      raise RuntimeError, "fatal error during Blosc compression: %d" % cbytes
    # Free the unused data
    self.data = <char *>realloc(dest, cbytes)
    # Set size info for the instance
    blosc.cbuffer_sizes(self.data, &nbytes_, &cbytes, &blocksize)
    assert nbytes_ == nbytes

    return (cbytes, blocksize)

  def getdata(self):
    """Get a compressed string object out of this chunk (for persistence)."""
    cdef object string

    assert (not self.isconstant,
            "This function can only be used for persistency")
    string = self.data[:self.cdbytes]
    return string

  cdef void _getitem(self, int start, int stop, char *dest):
    """Read data from `start` to `stop` and return it as a numpy array."""
    cdef int ret, bsize, blen, nitems, nstart
    cdef object constants

    blen = stop - start
    bsize = blen * self.atomsize
    nitems = cython.cdiv(bsize, self.itemsize)
    nstart = cython.cdiv(start * self.atomsize, self.itemsize)

    if self.isconstant:
      # The chunk is made of constants
      constants = utils.nd_empty_easy((blen,), self.dtype)
      constants[:] = self.constant
      data = <char *><Py_uintptr_t>_lowlevel.data_address_of(constants)
      memcpy(dest, data, bsize)
      return

    # Fill dest with uncompressed data
    with nogil:
      if bsize == self.nbytes:
        ret = blosc.decompress(self.data, dest, bsize)
      else:
        ret = blosc.getitem(self.data, nstart, nitems, dest)
    if ret < 0:
      raise RuntimeError, "fatal error during Blosc decompression: %d" % ret

  def __getitem__(self, object key):
    """__getitem__(self, key) -> values."""
    cdef npdefs.ndarray array
    cdef object ndarray
    cdef char* data
    cdef object start, stop, step, clen, idx

    if isinstance(key, _inttypes):
      # Quickly return a single element
      ndarray = nd.empty('1, %s' % self.dtype)
      data = <char *><Py_uintptr_t>_lowlevel.data_address_of(ndarray)
      self._getitem(key, key+1, data)
      return ndarray[0]
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
    else:
      raise IndexError, "key not suitable:", key

    # Get the corrected values for start, stop, step
    clen = cython.cdiv(self.nbytes, self.atomsize)
    (start, stop, step) = slice(start, stop, step).indices(clen)

    # Build a dynd container
    ndarray = nd.empty('%d, %s' % (stop-start, self.dtype))
    data = <char *><Py_uintptr_t>_lowlevel.data_address_of(ndarray)
    # Read actual data
    self._getitem(start, stop, data)

    # Return the value depending on the step
    if step > 1:
      return ndarray[::step]
    return ndarray

  @property
  def pointer(self):
      return <uintptr_t> self.data+BLOSCPACK_HEADER_LENGTH

  @property
  def viewof(self):
      return self.data[:self.cdbytes]

  def __setitem__(self, object key, object value):
    """__setitem__(self, key, value) -> None."""
    raise NotImplementedError

  def __str__(self):
    """Represent the chunk as an string."""
    return str(self[:])

  def __repr__(self):
    """Represent the chunk as an string, with additional info."""
    cratio = self.nbytes / float(self.cbytes)
    fullrepr = "chunk(%s)  nbytes: %d; cbytes: %d; ratio: %.2f\n%r" % \
        (self.dtype, self.nbytes, self.cbytes, cratio, str(self))
    return fullrepr

  def __dealloc__(self):
    """Release C resources before destruction."""
    if self.dobject:
      self.dobject = None  # DECREF pointer to data object
    else:
      free(self.data)   # explictly free the data area


cdef create_bloscpack_header(nchunks=None, format_version=FORMAT_VERSION):
    """ Create the bloscpack header string.

    Parameters
    ----------
    nchunks : int
        the number of chunks, default: None
    format_version : int
        the version format for the compressed file

    Returns
    -------
    bloscpack_header : string
        the header as string

    Notes
    -----

    The bloscpack header is 16 bytes as follows:

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
    | b   l   p   k | ^ | RESERVED  |           nchunks             |
                   version

    The first four are the magic string 'blpk'. The next one is an 8 bit
    unsigned little-endian integer that encodes the format version. The next
    three are reserved, and the last eight are a signed  64 bit little endian
    integer that encodes the number of chunks

    The value of '-1' for 'nchunks' designates an unknown size and can be
    inserted by setting 'nchunks' to None.

    Raises
    ------
    ValueError
        if the nchunks argument is too large or negative
    struct.error
        if the format_version is too large or negative

    """
    if not 0 <= nchunks <= MAX_CHUNKS and nchunks is not None:
      raise ValueError(
        "'nchunks' must be in the range 0 <= n <= %d, not '%s'" %
        (MAX_CHUNKS, str(nchunks)))
    return (MAGIC + struct.pack('<B', format_version) + b'\x00\x00\x00' +
            struct.pack('<q', nchunks if nchunks is not None else -1))

if sys.version_info >= (3, 0):
    def decode_byte(byte):
      return byte
else:
    def decode_byte(byte):
      return int(byte.encode('hex'), 16)
def decode_uint32(fourbyte):
  return struct.unpack('<I', fourbyte)[0]

cdef decode_blosc_header(buffer_):
    """ Read and decode header from compressed Blosc buffer.

    Parameters
    ----------
    buffer_ : string of bytes
        the compressed buffer

    Returns
    -------
    settings : dict
        a dict containing the settings from Blosc

    Notes
    -----

    The Blosc 1.1.3 header is 16 bytes as follows:

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
      ^   ^   ^   ^ |     nbytes    |   blocksize   |    ctbytes    |
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------versionlz
      +--------------version

    The first four are simply bytes, the last three are are each unsigned ints
    (uint32) each occupying 4 bytes. The header is always little-endian.
    'ctbytes' is the length of the buffer including header and nbytes is the
    length of the data when uncompressed.

    """
    return {'version': decode_byte(buffer_[0]),
            'versionlz': decode_byte(buffer_[1]),
            'flags': decode_byte(buffer_[2]),
            'typesize': decode_byte(buffer_[3]),
            'nbytes': decode_uint32(buffer_[4:8]),
            'blocksize': decode_uint32(buffer_[8:12]),
            'ctbytes': decode_uint32(buffer_[12:16])}


cdef class chunks(object):
  """Store the different barray chunks in a directory on-disk."""
  cdef object _rootdir, _mode
  cdef object dtype, bparams, lastchunkarr
  cdef object chunk_cached
  cdef blz_int_t nchunks, nchunk_cached, len

  property mode:
    "The mode used to create/open the `mode`."
    def __get__(self):
      return self._mode
    def __set__(self, value):
      self._mode = value

  property rootdir:
    "The on-disk directory used for persistency."
    def __get__(self):
      return self._rootdir
    def __set__(self, value):
      self._rootdir = value

  property datadir:
    """The directory for data files."""
    def __get__(self):
      return os.path.join(self.rootdir, DATA_DIR)

  def __cinit__(self, rootdir, metainfo=None, _new=False):
    cdef object lastchunkarr
    cdef char *decompressed, *compressed
    cdef int leftover
    cdef char *lastchunk
    cdef size_t chunksize
    cdef object scomp
    cdef int ret
    cdef int itemsize, atomsize

    self._rootdir = rootdir
    self.nchunks = 0
    self.nchunk_cached = -1    # no chunk cached initially
    self.dtype, self.bparams, self.len, lastchunkarr, self._mode = metainfo
    atomsize = self.dtype.data_size
    itemsize = self.dtype.dtype.data_size

    # Initialize last chunk
    if not _new:
      self.nchunks = cython.cdiv(self.len, len(lastchunkarr))
      chunksize = len(lastchunkarr) * atomsize
      lastchunk = <char *><Py_uintptr_t>_lowlevel.data_address_of(lastchunkarr)
      leftover = (self.len % len(lastchunkarr)) * atomsize
      if leftover:
        # Fill lastchunk with data on disk
        scomp = self.read_chunk(self.nchunks)
        compressed = scomp
        with nogil:
          ret = blosc.decompress(compressed, lastchunk, chunksize)
        if ret < 0:
          raise RuntimeError(
            "error decompressing the last chunk (error code: %d)" % ret)

  cdef read_chunk(self, nchunk):
    """Read a chunk and return it in compressed form."""
    dname = "__%d%s" % (nchunk, EXTENSION)
    schunkfile = os.path.join(self.datadir, dname)
    if not os.path.exists(schunkfile):
      raise ValueError("chunkfile %s not found" % schunkfile)
    with open(schunkfile, 'rb') as schunk:
      bloscpack_header = schunk.read(BLOSCPACK_HEADER_LENGTH)
      blosc_header_raw = schunk.read(BLOSC_HEADER_LENGTH)
      blosc_header = decode_blosc_header(blosc_header_raw)
      ctbytes = blosc_header['ctbytes']
      nbytes = blosc_header['nbytes']
      # seek back BLOSC_HEADER_LENGTH bytes in file relative to current
      # position
      schunk.seek(-BLOSC_HEADER_LENGTH, 1)
      scomp = schunk.read(ctbytes)
    return scomp

  def __getitem__(self, nchunk):
    cdef void *decompressed, *compressed

    if nchunk == self.nchunk_cached:
      # Hit!
      return self.chunk_cached
    else:
      scomp = self.read_chunk(nchunk)
      # Data chunk should be compressed already
      chunk_ = chunk(scomp, self.dtype, self.bparams,
                     _memory=False, _compr=True)
      # Fill cache
      self.nchunk_cached = nchunk
      self.chunk_cached = chunk_
    return chunk_

  def __setitem__(self, nchunk, chunk_):
    self._save(nchunk, chunk_)

  def __len__(self):
    return self.nchunks

  def free_cachemem(self):
      self.nchunk_cached = -1
      self.chunk_cached = None

  def append(self, chunk_):
    """Append an new chunk to the barray."""
    self._save(self.nchunks, chunk_)
    self.nchunks += 1

  cdef _save(self, nchunk, chunk_):
    """Save the `chunk_` as chunk #`nchunk`. """

    if self.mode == "r":
      raise RuntimeError(
        "cannot modify data because mode is '%s'" % self.mode)

    dname = "__%d%s" % (nchunk, EXTENSION)
    schunkfile = os.path.join(self.datadir, dname)
    bloscpack_header = create_bloscpack_header(1)
    with open(schunkfile, 'wb') as schunk:
      schunk.write(bloscpack_header)
      data = chunk_.getdata()
      schunk.write(data)
    # Mark the cache as dirty if needed
    if nchunk == self.nchunk_cached:
      self.nchunk_cached = -1

  def flush(self, chunk_):
    """Flush the leftover chunk."""
    self._save(self.nchunks, chunk_)

  def pop(self):
    """Remove the last chunk and return it."""
    nchunk = self.nchunks - 1
    chunk_ = self.__getitem__(nchunk)
    dname = "__%d%s" % (nchunk, EXTENSION)
    schunkfile = os.path.join(self.datadir, dname)
    if not os.path.exists(schunkfile):
      raise RuntimeError("chunk filename %s does exist" % schunkfile)
    os.remove(schunkfile)

    # When poping a chunk, we must be sure that we don't leave anything
    # behind (i.e. the lastchunk)
    dname = "__%d%s" % (nchunk+1, EXTENSION)
    schunkfile = os.path.join(self.datadir, dname)
    if os.path.exists(schunkfile):
      os.remove(schunkfile)

    self.nchunks -= 1
    return chunk_


cdef class barray:
  """
  barray(array, bparams=None, dtype=None, dflt=None, expectedlen=None,
         chunklen=None, rootdir=None, mode='a')

  A compressed and enlargeable in-memory data container.

  `barray` exposes a series of methods for dealing with the compressed
  container in a NumPy-like way.

  Parameters
  ----------
  array : a NumPy-like object
      This is taken as the input to create the barray.  It can be any Python
      object that can be converted into a NumPy object.  The data type of
      the resulting barray will be the same as this NumPy object.
  bparams : instance of the `bparams` class, optional
      Parameters to the internal Blosc compressor.
  dtype : NumPy dtype
      Force this `dtype` for the barray (rather than the `array` one).
  dflt : Python or NumPy scalar
      The value to be used when enlarging the barray.  If None, the default is
      filling with zeros.
  expectedlen : int, optional
      A guess on the expected length of this object.  This will serve to
      decide the best `chunklen` used for compression and memory I/O
      purposes.
  chunklen : int, optional
      The number of items that fits into a chunk.  By specifying it you can
      explicitely set the chunk size used for compression and memory I/O.
      Only use it if you know what are you doing.
  rootdir : str, optional
      The directory where all the data and metadata will be stored.  If
      specified, then the barray object will be disk-based (i.e. all chunks
      will live on-disk, not in memory) and persistent (i.e. it can be
      restored in other session, e.g. via the `open()` top-level function).
  mode : str, optional
      The mode that a *persistent* barray should be created/opened.  The
      values can be:

        * 'r' for read-only
        * 'w' for read/write.  During barray creation, the `rootdir` will be
          removed if it exists.  During barray opening, the barray will be
          resized to 0.
        * 'a' for append (possible data inside `rootdir` will not be removed).

  """

  cdef public int itemsize, atomsize
  cdef int _chunksize, _chunklen, leftover
  cdef int nrowsinbuf, _row
  cdef int sss_mode, wheretrue_mode, where_mode
  cdef blz_int_t startb, stopb
  cdef blz_int_t start, stop, step, nextelement
  cdef blz_int_t _nrow, nrowsread
  cdef blz_int_t _nbytes, _cbytes
  cdef blz_int_t nhits, limit, skip
  cdef blz_int_t expectedlen
  cdef char *lastchunk
  cdef object lastchunkarr, where_arr
  cdef object _bparams, _dflt
  cdef object _dtype, _shape
  cdef public object chunks
  cdef object _rootdir, datadir, metadir, _mode
  cdef object _attrs
  cdef object iobuf, where_buf

  property leftovers:
    def __get__(self):
      # Pointer to the leftovers chunk
      #return self.lastchunkarr.ctypes.data
      return <char *><Py_uintptr_t>_lowlevel.data_address_of(self.lastchunkarr)

  property nchunks:
    def __get__(self):
      # TODO: do we need to handle the last chunk specially?
      return <blz_int_t>cython.cdiv(self._nbytes, self._chunksize)

  property partitions:
    def __get__(self):
      # Return a sequence of tuples indicating the bounds
      # of each of the chunks.
      nchunks = <blz_int_t>cython.cdiv(self._nbytes, self._chunksize)
      chunklen = cython.cdiv(self._chunksize, self.atomsize)
      return [(i*chunklen,(i+1)*chunklen) for i in xrange(nchunks)]

  property leftover_array:
      def __get__(self):
          return self.lastchunkarr

  property attrs:
    "The attribute accessor."
    def __get__(self):
      return self._attrs

  property cbytes:
    "The compressed size of this object (in bytes)."
    def __get__(self):
      return self._cbytes

  property chunklen:
    "The chunklen of this object (in rows)."
    def __get__(self):
      return self._chunklen

  property bparams:
    "The compression parameters for this object."
    def __get__(self):
      return self._bparams

  property dflt:
    "The default value of this object."
    def __get__(self):
      return self._dflt

  property dtype:
    "The dtype of this object."
    def __get__(self):
      return self._dtype.dtype

  property len:
    "The length (leading dimension) of this object."
    def __get__(self):
      # Important to do the cast in order to get a npy_intp result
      return <blz_int_t>cython.cdiv(self._nbytes, self.atomsize)

  property mode:
    "The mode used to create/open the `mode`."
    def __get__(self):
      return self._mode
    def __set__(self, value):
      self._mode = value
      self.chunks.mode = value

  property nbytes:
    "The original (uncompressed) size of this object (in bytes)."
    def __get__(self):
      return self._nbytes

  property ndim:
    "The number of dimensions of this object."
    def __get__(self):
      return len(self.shape)

  property shape:
    "The shape of this object."
    def __get__(self):
      return tuple((self.len,) + self._shape)

  property size:
    "The size of this object."
    def __get__(self):
      return np.prod(self.shape)

  property rootdir:
    "The on-disk directory used for persistency."
    def __get__(self):
      return self._rootdir
    def __set__(self, value):
      if not self.rootdir:
        raise ValueError(
          "cannot modify the rootdir value of an in-memory barray")
      self._rootdir = value
      self.chunks.rootdir = value

  def __cinit__(self, object array=None, object bparams=None,
                object dtype=None, object dflt=None,
                object expectedlen=None, object chunklen=None,
                object rootdir=None, object mode="a"):

    self._rootdir = rootdir
    if mode not in ('r', 'w', 'a'):
      raise ValueError("mode should be 'r', 'w' or 'a'")
    self._mode = mode

    if array is not None:
      self.create_barray(array, bparams, dtype, dflt,
                         expectedlen, chunklen, rootdir, mode)
      _new = True
    elif rootdir is not None:
      meta_info = self.read_meta()
      self.open_barray(*meta_info)
      _new = False
    else:
      raise ValueError("You need at least to pass an array or/and a rootdir")

    # Attach the attrs to this object
    self._attrs = attrs.attrs(self._rootdir, self.mode, _new=_new)

    # Sentinels
    self.sss_mode = False
    self.wheretrue_mode = False
    self.where_mode = False

  def set_default(self, dtype, dflt):
    _dflt = nd.empty(dtype)
    if dflt is not None:
      if not hasattr(dflt, "eval") and dflt.shape == ():
        # Convert zero-dim numpy array into an scalar so as to avoid a
        # segfault (see https://github.com/ContinuumIO/dynd-python/issues/25)
        dflt = dflt[()]
      _dflt[()] = dflt
    else:
      # Provide sensible defaults here
      if dtype.kind in ('int', 'real', 'complex'):
        _dflt[()] = 0
      elif dtype.kind in ('bool',):
        _dflt[()] = False
      elif dtype.kind in ('bytes', 'string'):
        _dflt[()] = ""
    return _dflt

  def create_barray(self, array, bparams, dtype, dflt,
                    expectedlen, chunklen, rootdir, mode):
    """Create a new array. """
    cdef int itemsize, atomsize, chunksize
    cdef object lastchunkarr, array_, _dflt

    # Check defaults for bparams
    if bparams is None:
      bparams = blz.bparams()

    if not isinstance(bparams, blz.bparams):
      raise ValueError, "`bparams` param must be an instance of `bparams` class"

    # Convert input to an appropriate type
    if type(dtype) is str:
        dtype = np.dtype(dtype)

    # avoid bad performance with another barray, as in utils it would
    # construct the temp ndarray using a slow iterator.
    #
    # TODO: There should be a fast path creating barrays from other barrays
    # (especially when dtypes and compression params match)
    if isinstance(array, barray):
      array = array[:]

    # If no base dtype is provided, use the dtype from the array.
    if dtype is None:
      if hasattr(array, 'eval'):
        dtype = nd.type_of(array).dtype   # dynd array
      else:
        dtype = ndt.make_fixed_dim((), str(array.dtype))  # numpy array

    # Build a new array with the possible new dtype
    array_ = utils.to_ndarray(array, dtype)

    # Multidimensional array.  The atom will have array_.shape[1:] dims.
    # atom dimensions will be stored in `self._dtype`, which is different
    # than `self.dtype` in that `self._dtype` dimensions are borrowed
    # from `self.shape`.  `self.dtype` will always be scalar.
    #
    # Note that objects are a special case. barray does not support object
    # arrays of more than one dimensions.
    self._dtype = dtype = ndt.make_fixed_dim(array_.shape[1:], dtype)
    self._shape = tuple(array_.shape[1:])

    # Check that atom size is less than 2 GB
    if dtype.data_size >= 2**31:
      raise ValueError, "atomic size is too large (>= 2 GB)"

    self.atomsize = atomsize = dtype.data_size
    self.itemsize = itemsize = dtype.dtype.data_size

    # Check defaults for dflts
    self._dflt = self.set_default(dtype, dflt)

    # Compute the chunklen/chunksize
    if expectedlen is None:
      # Try a guess
      try:
        expectedlen = len(array_)
      except TypeError:
        raise NotImplementedError(
          "creating barrays from scalar objects not supported")
    try:
      self.expectedlen = expectedlen
    except OverflowError:
      raise OverflowError(
        "The size cannot be larger than 2**31 on 32-bit platforms")

    if chunklen is None:
      # Try a guess
      chunksize = utils.calc_chunksize((expectedlen * atomsize) / float(_MB))
      # Chunksize must be a multiple of atomsize
      chunksize = cython.cdiv(chunksize, atomsize) * atomsize
      # Protection against large itemsizes
      if chunksize < atomsize:
        chunksize = atomsize
    else:
      if not isinstance(chunklen, int) or chunklen < 1:
        raise ValueError, "chunklen must be a positive integer"
      chunksize = chunklen * atomsize
    chunklen = cython.cdiv(chunksize, atomsize)
    self._chunksize = chunksize
    self._chunklen = chunklen

    # Book memory for last chunk (uncompressed)
    # Use np.zeros here because they compress better
    #lastchunkarr = np.zeros(dtype=dtype, shape=(chunklen,))
    # XXX Use nd.zeros when this would be implemented
    lastchunkarr = nd.empty(chunklen, dtype)
    lastchunkarr[:] = 0
    self.lastchunk = <char *><Py_uintptr_t>_lowlevel.data_address_of(lastchunkarr)
    self.lastchunkarr = lastchunkarr

    # Create layout for data and metadata
    self._bparams = bparams
    self.chunks = []
    if rootdir is not None:
      self.mkdirs(rootdir, mode)
      metainfo = (dtype, bparams, self.shape[0], lastchunkarr, self._mode)
      self.chunks = chunks(self._rootdir, metainfo=metainfo, _new=True)
      # We can write the metainfo already
      self.write_meta()

    # Finally, fill the chunks
    self.fill_chunks(array_)

    # and flush the data pending...
    self.flush()

  def open_barray(self, shape, bparams, dtype, dflt,
                  expectedlen, cbytes, chunklen):
    """Open an existing array."""
    cdef npdefs.ndarray lastchunkarr
    cdef object array_, _dflt
    cdef blz_int_t calen

    if len(shape) == 1:
        self._dtype = dtype
        self._shape = ()
    else:
      # Multidimensional array.  The atom will have array_.shape[1:] dims.
      # atom dimensions will be stored in `self._dtype`, which is different
      # than `self.dtype` in that `self._dtype` dimensions are borrowed
      # from `self.shape`.  `self.dtype` will always be scalar (NumPy
      # convention).
      self._dtype = dtype = np.dtype((dtype.base, shape[1:]))
      self._shape = tuple(shape[1:])

    self._bparams = bparams
    self.atomsize = dtype.itemsize
    self.itemsize = dtype.base.itemsize
    self._chunklen = chunklen
    self._chunksize = chunklen * self.atomsize
    self._dflt = dflt
    self.expectedlen = expectedlen

    # Book memory for last chunk (uncompressed)
    # Use np.zeros here because they compress better
    #lastchunkarr = np.zeros(dtype=dtype, shape=(chunklen,))
    # XXX Use nd.zeros when this would be implemented
    lastchunkarr = nd.empty("%d, %s" % (chunklen, dtype))
    lastchunkarr[:] = 0
    #self.lastchunk = lastchunkarr.data
    self.lastchunk = <char *><Py_uintptr_t>_lowlevel.data_address_of(lastchunkarr)
    self.lastchunkarr = lastchunkarr

    # Check rootdir hierarchy
    if not os.path.isdir(self._rootdir):
      raise RuntimeError("root directory does not exist")
    self.datadir = os.path.join(self._rootdir, DATA_DIR)
    if not os.path.isdir(self.datadir):
      raise RuntimeError("data directory does not exist")
    self.metadir = os.path.join(self._rootdir, META_DIR)
    if not os.path.isdir(self.metadir):
      raise RuntimeError("meta directory does not exist")

    calen = shape[0]    # the length ot the barray
    # Finally, open data directory
    metainfo = (dtype, bparams, calen, lastchunkarr, self._mode)
    self.chunks = chunks(self._rootdir, metainfo=metainfo, _new=False)

    # Update some counters
    self.leftover = (calen % chunklen) * self.atomsize
    self._cbytes = cbytes
    self._nbytes = calen * self.atomsize

    if self._mode == "w":
      # Remove all entries when mode is 'w'
      self.resize(0)

  def fill_chunks(self, object array_):
    """Fill chunks, either in-memory or on-disk."""
    cdef int leftover, chunklen
    cdef blz_int_t i, nchunks
    cdef blz_int_t nbytes, cbytes
    cdef chunk chunk_
    cdef object remainder
    cdef uintptr_t arrsize
    cdef char* data

    # The number of bytes in incoming array
    arrsize = np.prod(array_.shape)
    nbytes = self.itemsize * arrsize
    self._nbytes = nbytes

    # Compress data in chunks
    cbytes = 0
    chunklen = self._chunklen
    nchunks = <blz_int_t>cython.cdiv(nbytes, self._chunksize)
    for i from 0 <= i < nchunks:
      assert i*chunklen < arrsize, "i, nchunks: %d, %d" % (i, nchunks)
      chunk_ = chunk(array_[i*chunklen:(i+1)*chunklen],
                     self._dtype, self._bparams,
                     _memory = self._rootdir is None)
      self.chunks.append(chunk_)
      cbytes += chunk_.cbytes
    self.leftover = leftover = nbytes % self._chunksize
    if leftover:
      remainder = array_[nchunks*chunklen:]
      data = <char *><Py_uintptr_t>_lowlevel.data_address_of(remainder)
      memcpy(self.lastchunk, data, leftover)
    cbytes += self._chunksize  # count the space in last chunk
    self._cbytes = cbytes

  def mkdirs(self, object rootdir, object mode):
    """Create the basic directory layout for persistent storage."""
    if os.path.exists(rootdir):
      if self._mode != "w":
        raise RuntimeError(
          "specified rootdir path '%s' already exists "
          "and creation mode is '%s'" % (rootdir, mode))
      if os.path.isdir(rootdir):
        shutil.rmtree(rootdir)
      else:
        os.remove(rootdir)
    os.mkdir(rootdir)
    self.datadir = os.path.join(rootdir, DATA_DIR)
    os.mkdir(self.datadir)
    self.metadir = os.path.join(rootdir, META_DIR)
    os.mkdir(self.metadir)

  def write_meta(self):
      """Write metadata persistently."""
      storagef = os.path.join(self.metadir, STORAGE_FILE)
      with open(storagef, 'wb') as storagefh:
        dflt_list = nd.as_py(self.dflt)
        if type(dflt_list) in (datetime.datetime,
                               datetime.date, datetime.time):
            # The datetime cannot be serialized with JSON.  Use a 0 int.
            dflt_list = 0
        # In Python 3, the json encoder doesn't accept bytes objects
        if sys.version_info >= (3, 0):
            dflt_list = list_bytes_to_str(dflt_list)
        storagefh.write(json.dumps({
          # str(self.dtype) produces bytes by default in cython.py3.
          # Calling .__str__() is a workaround.
          "dtype": self.dtype.__str__(),
          "bparams": {
            "clevel": self.bparams.clevel,
            "shuffle": self.bparams.shuffle,
            },
          "chunklen": self._chunklen,
          "expectedlen": self.expectedlen,
          "dflt": dflt_list,
          }, ensure_ascii=True).encode('ascii'))
        storagefh.write(b"\n")

  def read_meta(self):
    """Read persistent metadata."""

    # First read the size info
    metadir = os.path.join(self._rootdir, META_DIR)
    shapef = os.path.join(metadir, SIZES_FILE)
    with open(shapef, 'rb') as shapefh:
      sizes = json.loads(shapefh.read().decode('ascii'))
    shape = sizes['shape']
    if type(shape) == list:
      shape = tuple(shape)
    nbytes = sizes["nbytes"]
    cbytes = sizes["cbytes"]

    # Then the rest of metadata
    storagef = os.path.join(metadir, STORAGE_FILE)
    with open(storagef, 'rb') as storagefh:
      data = json.loads(storagefh.read().decode('ascii'))
    dtype_ = np.dtype(data["dtype"])
    chunklen = data["chunklen"]
    bparams = blz.bparams(
      clevel = data["bparams"]["clevel"],
      shuffle = data["bparams"]["shuffle"])
    expectedlen = data["expectedlen"]
    dflt = data["dflt"]
    return (shape, bparams, dtype_, dflt, expectedlen, cbytes, chunklen)

  def append(self, object array):
    """
    append(array)

    Append a numpy `array` to this instance.

    Parameters
    ----------
    array : NumPy-like object
        The array to be appended.  Must be compatible with shape and type of
        the barray.

    """
    cdef int atomsize, itemsize, chunksize, leftover
    cdef int nbytesfirst, chunklen, start, stop
    cdef blz_int_t nbytes, cbytes, bsize, i, nchunks
    cdef object remainder, arrcpy, dflts
    cdef chunk chunk_
    cdef char* data

    if self.mode == "r":
      raise RuntimeError(
        "cannot modify data because mode is '%s'" % self.mode)

    arrcpy = utils.to_ndarray(array, self._dtype)
    if nd.type_of(arrcpy).dtype != self._dtype.dtype:
      raise TypeError, "array dtype does not match with self"

    # Appending a single row should be supported
    if arrcpy.shape == self._shape:
      arrcpy = arrcpy.reshape((1,)+arrcpy.shape)
    if arrcpy.shape[1:] != self._shape:
      raise ValueError, "array trailing dimensions do not match with self"
    data = <char *><Py_uintptr_t>_lowlevel.data_address_of(arrcpy)

    atomsize = self.atomsize
    itemsize = self.itemsize
    chunksize = self._chunksize
    chunks = self.chunks
    leftover = self.leftover
    bsize = np.prod(arrcpy.shape) * itemsize
    cbytes = 0

    # Check if array fits in existing buffer
    if (bsize + leftover) < chunksize:
      # Data fits in lastchunk buffer.  Just copy it
      if arrcpy.strides[0] > 0:
        memcpy(self.lastchunk+leftover, data, bsize)
      else:
        start = cython.cdiv(leftover, atomsize)
        stop = cython.cdiv((leftover+bsize), atomsize)
        self.lastchunkarr[start:stop] = arrcpy
      leftover += bsize
    else:
      # Data does not fit in buffer.  Break it in chunks.

      # First, fill the last buffer completely (if needed)
      if leftover:
        nbytesfirst = chunksize - leftover
        if bsize == itemsize or arrcpy.strides[0] > 0:
          memcpy(self.lastchunk+leftover, data, nbytesfirst)
        else:
          start = cython.cdiv(leftover, atomsize)
          stop = cython.cdiv((leftover+nbytesfirst), atomsize)
          self.lastchunkarr[start:stop] = arrcpy[start:stop]
        # Compress the last chunk and add it to the list
        chunk_ = chunk(self.lastchunkarr, self._dtype, self._bparams,
                       _memory = self._rootdir is None)
        chunks.append(chunk_)
        cbytes = chunk_.cbytes
      else:
        nbytesfirst = 0

      # Then fill other possible chunks
      nbytes = bsize - nbytesfirst
      nchunks = <blz_int_t>cython.cdiv(nbytes, chunksize)
      chunklen = self._chunklen
      # Get a new view skipping the elements that have been already copied
      remainelems = cython.cdiv(nbytesfirst, atomsize)
      if remainelems < len(arrcpy):
        remainder = arrcpy[remainelems:]
      for i from 0 <= i < nchunks:
        chunk_ = chunk(
          remainder[i*chunklen:(i+1)*chunklen], self._dtype, self._bparams,
          _memory = self._rootdir is None)
        chunks.append(chunk_)
        cbytes += chunk_.cbytes

      # Finally, deal with the leftover
      leftover = nbytes % chunksize
      if leftover:
        remainder = remainder[nchunks*chunklen:]
        if remainder.strides[0] > 0:
          data = <char *><Py_uintptr_t>_lowlevel.data_address_of(remainder)
          memcpy(self.lastchunk, data, leftover)
        else:
          self.lastchunkarr[:len(remainder)] = remainder

    # Update some counters
    self.leftover = leftover
    self._cbytes += cbytes
    self._nbytes += bsize
    return

  def trim(self, object nitems):
    """
    trim(nitems)

    Remove the trailing `nitems` from this instance.

    Parameters
    ----------
    nitems : int
        The number of trailing items to be trimmed.  If negative, the object
        is enlarged instead.

    """
    cdef int atomsize, leftover, leftover2
    cdef blz_int_t cbytes, bsize, nchunk2
    cdef chunk chunk_

    if not isinstance(nitems, _inttypes +(float,)):
      raise TypeError, "`nitems` must be an integer"

    # Check that we don't run out of space
    if nitems > self.len:
      raise ValueError, "`nitems` must be less than total length"
    # A negative number of items means that we want to grow the object
    if nitems <= 0:
      self.resize(self.len - nitems)
      return

    atomsize = self.atomsize
    chunks = self.chunks
    leftover = self.leftover
    bsize = nitems * atomsize
    cbytes = 0

    # Check if items belong to the last chunk
    if (leftover - bsize) > 0:
      # Just update leftover counter
      leftover -= bsize
    else:
      # nitems larger than last chunk
      nchunk = cython.cdiv((self.len - nitems), self._chunklen)
      leftover2 = (self.len - nitems) % self._chunklen
      leftover = leftover2 * atomsize

      # Remove complete chunks
      nchunk2 = lnchunk = <blz_int_t>cython.cdiv(self._nbytes,
                                                 self._chunksize)
      while nchunk2 > nchunk:
        chunk_ = chunks.pop()
        cbytes += chunk_.cbytes
        nchunk2 -= 1

      # Finally, deal with the leftover
      if leftover:
        self.lastchunkarr[:leftover2] = chunk_[:leftover2]
        if self._rootdir:
          # Last chunk is removed automatically by the chunks.pop() call, and
          # always is counted as if it is not compressed (although it is in
          # this state on-disk)
          cbytes += chunk_.nbytes

    # Update some counters
    self.leftover = leftover
    self._cbytes -= cbytes
    self._nbytes -= bsize
    # Flush last chunk and update counters on-disk
    self.flush()

  def resize(self, object nitems):
    """
    resize(nitems)

    Resize the instance to have `nitems`.

    Parameters
    ----------
    nitems : int
        The final length of the object.  If `nitems` is larger than the actual
        length, new items will appended using `self.dflt` as filling values.

    """
    cdef object chunk

    if not isinstance(nitems, _inttypes + (float,)):
      raise TypeError, "`nitems` must be an integer"

    if nitems == self.len:
      return
    elif nitems < 0:
      raise ValueError, "`nitems` cannot be negative"

    if nitems > self.len:
      # Create a 0-strided array and append it to self
      # chunk = np.ndarray(nitems-self.len, dtype=self._dtype,
      #                    buffer=self._dflt, strides=(0,))
      # XXX Use a strided-zero nd.array when it would be implemented in dynd
      chunk = utils.nd_empty_easy((nitems - self.len,), self._dtype)
      chunk[:] = self._dflt
      self.append(chunk)
      self.flush()
    else:
      # Just trim the excess of items
      self.trim(self.len-nitems)

  def reshape(self, newshape):
    """
    reshape(newshape)

    Returns a new barray containing the same data with a new shape.

    Parameters
    ----------
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is inferred
        from the length of the array and remaining dimensions.

    Returns
    -------
    reshaped_array : barray
        A copy of the original barray.

    """
    cdef blz_int_t newlen, ilen, isize, osize, newsize, rsize, i
    cdef object ishape, oshape, pos, newdtype, out

    # Enforce newshape as tuple
    if isinstance(newshape, _inttypes):
      newshape = (newshape,)
    newsize = np.prod(newshape)

    ishape = self.shape
    ilen = ishape[0]
    isize = np.prod(ishape)

    # Check for -1 in newshape
    if -1 in newshape:
      if newshape.count(-1) > 1:
        raise ValueError, "only one shape dimension can be -1"
      pos = newshape.index(-1)
      osize = np.prod(newshape[:pos] + newshape[pos+1:])
      if isize == 0:
        newshape = newshape[:pos] + (0,) + newshape[pos+1:]
      else:
        newshape = newshape[:pos] + (isize/osize,) + newshape[pos+1:]
      newsize = np.prod(newshape)

    # Check shape compatibility
    if isize != newsize:
      raise ValueError, "`newshape` is not compatible with the current one"
    # Create the output container
    newdtype = np.dtype((self._dtype.base, newshape[1:]))
    newlen = newshape[0]

    # If shapes are both n-dimensional, convert first to 1-dim shape
    # and then convert again to the final newshape.
    if len(ishape) > 1 and len(newshape) > 1:
      out = self.reshape(-1)
      return out.reshape(newshape)

    if self._rootdir:
      # If persistent, do the copy to a temporary dir
      absdir = os.path.dirname(self._rootdir)
      rootdir = tempfile.mkdtemp(suffix='__temp__', dir=absdir)
    else:
      rootdir = None

    # Create the final container and fill it
    out = barray([], dtype=newdtype, bparams=self.bparams,
                 expectedlen=newlen,
                 rootdir=rootdir, mode='w')
    if newlen < ilen:
      rsize = isize / newlen
      for i from 0 <= i < newlen:
        out.append(self[i*rsize:(i+1)*rsize].reshape(newdtype.shape))
    else:
      for i from 0 <= i < ilen:
        out.append(self[i].reshape(-1))
    out.flush()

    # Finally, rename the temporary data directory to self._rootdir
    if self._rootdir:
      shutil.rmtree(self._rootdir)
      os.rename(rootdir, self._rootdir)
      # Restore the rootdir and mode
      out.rootdir = self._rootdir
      out.mode = self._mode

    return out

  def copy(self, **kwargs):
    """
    copy(**kwargs)

    Return a copy of this object.

    Parameters
    ----------
    kwargs : list of parameters or dictionary
        Any parameter supported by the barray constructor.

    Returns
    -------
    out : barray object
        The copy of this object.

    """
    cdef object chunklen

    # Get defaults for some parameters
    bparams = kwargs.pop('bparams', self._bparams)
    expectedlen = kwargs.pop('expectedlen', self.len)

    # Create a new, empty barray
    ccopy = barray(utils.nd_empty_easy((0,), self._dtype),
                   bparams=bparams,
                   expectedlen=expectedlen,
                   **kwargs)

    # Now copy the barray chunk by chunk
    chunklen = self._chunklen
    for i from 0 <= i < self.len by chunklen:
      ccopy.append(self[i:i+chunklen])
    ccopy.flush()

    return ccopy

  def sum(self, dtype=None):
    """
    sum(dtype=None)

    Return the sum of the array elements.

    Parameters
    ----------
    dtype : NumPy dtype
        The desired type of the output.  If ``None``, the dtype of `self` is
        used.  An exception is when `self` has an integer type with less
        precision than the default platform integer.  In that case, the
        default platform integer is used instead (NumPy convention).


    Return value
    ------------
    out : NumPy scalar with `dtype`

    """
    cdef chunk chunk_
    cdef blz_int_t nchunk, nchunks
    cdef object result

    if dtype is None:
      dtype = self._dtype
      # Check if we have less precision than required for ints
      # (mimick NumPy logic)
      if (dtype.kind in ('bool', 'int') and
          dtype.data_size < IntType.itemsize):
        dtype = IntType
    else:
      if type(dtype) == np.dtype:
        dtype = ndt.type(dtype)
      elif type(dtype) == str:
        dtype = ndt.type(np.dtype(dtype))
    if dtype.kind in ('bytes', 'string'):
      raise TypeError, "cannot perform reduce with flexible type"

    # Get a container for the result
    result = nd.array(0, dtype=dtype)

    nchunks = <blz_int_t>cython.cdiv(self._nbytes, self._chunksize)
    for nchunk from 0 <= nchunk < nchunks:
      chunk_ = self.chunks[nchunk]
      if chunk_.isconstant:
        result += chunk_.constant * self._chunklen
      elif self._dtype.dtype == ndt.bool:
        result += chunk_.true_count
      else:
        #result += chunk_[:].sum(dtype=dtype)
        # XXX workaround until dynd would implement a sum() method
        result += np.asarray(chunk_[:]).sum(dtype=str(dtype))
    if self.leftover:
      leftover = self.len - nchunks * self._chunklen
      #result += self.lastchunkarr[:leftover].sum(dtype=dtype)
      # XXX workaround until dynd would implement a sum() method
      result += np.asarray(self.lastchunkarr[:leftover]).sum(dtype=str(dtype))
      result = result.eval()

    return result

  def __len__(self):
    return self.len

  def __sizeof__(self):
    return self._cbytes

  def free_cachemem(self):

    if type(self.chunks) is not list:
      self.chunks.free_cachemem()
    self.idxcache = -1
    self.blockcache = None

  def __getitem__(self, object key):
    """
    x.__getitem__(key) <==> x[key]

    Returns values based on `key`.  All the functionality of
    ``ndarray.__getitem__()`` is supported (including fancy indexing), plus a
    special support for expressions:

    Parameters
    ----------
    key : string
        It will be interpret as a boolean expression (computed via `eval`) and
        the elements where these values are true will be returned as a NumPy
        array.

    See Also
    --------
    eval

    """

    cdef int chunklen
    cdef blz_int_t startb, stopb
    cdef blz_int_t nchunk, keychunk, nchunks
    cdef blz_int_t nwrow, blen
    cdef object start, stop, step
    cdef object arr

    chunklen = self._chunklen

    # Check for integer
    if (isinstance(key, _inttypes) or
        (hasattr(key, 'eval') and nd.type_of(key).kind == 'int')):
      key = operator.index(key)  # convert into an index
      if key < 0:
        # To support negative values
        key += self.len
      if key >= self.len:
        raise IndexError, "index out of range"
      return self[slice(key, key+1, 1)][0]
    # Slices
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
      if step and step <= 0 :
        raise NotImplementedError("step in slice can only be positive")
    # Multidimensional keys
    elif isinstance(key, tuple):
      if len(key) == 0:
        raise ValueError("empty tuple not supported")
      elif len(key) == 1:
        return self[key[0]]
      # An n-dimensional slice
      # First, retrieve elements in the leading dimension
      arr = self[key[0]]
      # Then, keep only the required elements in other dimensions
      if type(key[0]) == slice:
        arr = arr[(slice(None),) + key[1:]]
      else:
        arr = arr[key[1:]]
      # Force a copy in case returned array is not contiguous
      if not arr.flags.contiguous:
        arr = arr.copy()
      return arr
    # List of integers (case of fancy indexing)
    elif isinstance(key, list):
      # Try to convert to a integer array
      try:
        key = nd.array((operator.index(i) for i in key), dtype='int64')
      except:
        raise IndexError, "key cannot be converted to an array of indices"
      return self[key]
    # A boolean or integer array (case of fancy indexing)
    elif (hasattr(key, "dtype") or hasattr(key, "eval")):
      # Accept arrays that can have dynd types too
      if hasattr(key, "eval"):
        # Quacks like a dynd array
        typeobj = nd.type_of(key).dtype
      else:
        typeobj = ndt.type(key.dtype)
      if typeobj == ndt.bool:
        # A boolean array
        if len(key) != self.len:
          raise IndexError, "boolean array length must match len(self)"
        if isinstance(key, barray):
          #count = key.sum()
          # XXX workaround until dynd would implement a sum() method
          count = np.asarray(key).sum()
        else:
          #count = -1         # for the fromiter
          count = len(self)   # for the iterator
        # XXX This should be replaced by nd.fromiter() as soon as it would be
        # implemented (https://github.com/ContinuumIO/dynd-python/issues/24)
        return nd.array(
          (v for n, v in enumerate(self.where(key)) if n < count),
           dtype=self._dtype)
      elif typeobj.kind == "int":
        # An integer array
        return nd.array((self[i] for i in key), dtype=self._dtype)
      else:
        raise IndexError, \
              "arrays used as indices must be of integer (or boolean) type"
    # All the rest not implemented
    else:
      raise NotImplementedError, "key not supported: %s" % repr(key)

    # From now on, will only deal with [start:stop:step] slices

    # Get the corrected values for start, stop, step
    (start, stop, step) = slice(start, stop, step).indices(self.len)

    # Build a numpy container
    blen = get_len_of_range(start, stop, step)
    arr = utils.nd_empty_easy((blen,), self._dtype)
    if blen == 0:
      # If empty, return immediately
      return arr

    # Fill it from data in chunks
    nwrow = 0
    nchunks = <blz_int_t>cython.cdiv(self._nbytes, self._chunksize)
    if self.leftover > 0:
      nchunks += 1
    for nchunk from 0 <= nchunk < nchunks:
      # Compute start & stop for each block
      startb, stopb, blen = clip_chunk(nchunk, chunklen, start, stop, step)
      if blen == 0:
        continue
      # Get the data chunk and assign it to result array
      if nchunk == nchunks-1 and self.leftover:
        arr[nwrow:nwrow+blen] = self.lastchunkarr[startb:stopb:step]
      else:
        arr[nwrow:nwrow+blen] = self.chunks[nchunk][startb:stopb:step]
      nwrow += blen

    return arr

  def __setitem__(self, object key, object value):
    """
    x.__setitem__(key, value) <==> x[key] = value

    Sets values based on `key`.  All the functionality of
    ``ndarray.__setitem__()`` is supported (including fancy indexing), plus a
    special support for expressions:

    Parameters
    ----------
    key : string
        It will be interpret as a boolean expression (computed via `eval`) and
        the elements where these values are true will be set to `value`.

    See Also
    --------
    eval

    """
    cdef int chunklen
    cdef blz_int_t startb, stopb
    cdef blz_int_t nchunk, keychunk, nchunks
    cdef blz_int_t nwrow, blen, vlen
    cdef chunk chunk_
    cdef object start, stop, step
    cdef object cdata, arr

    if self.mode == "r":
      raise RuntimeError(
        "cannot modify data because mode is '%s'" % self.mode)

    # Check for integer
    if (isinstance(key, _inttypes) or
        (hasattr(key, 'eval') and nd.type_of(key).kind == 'int')):
      key = operator.index(key)  # convert into an index
      if key < 0:
        # To support negative values
        key += self.len
      if key >= self.len:
        raise IndexError, "index out of range"
      (start, stop, step) = key, key+1, 1
    # Slices
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
      if step:
        if step <= 0 :
          raise NotImplementedError("step in slice can only be positive")
    # Multidimensional keys
    elif isinstance(key, tuple):
      if len(key) == 0:
        raise ValueError("empty tuple not supported")
      elif len(key) == 1:
        self[key[0]] = value
        return
      # An n-dimensional slice
      # First, retrieve elements in the leading dimension
      arr = self[key[0]]
      # Then, assing only the requested elements in other dimensions
      if type(key[0]) == slice:
        arr[(slice(None),) + key[1:]] = value
      else:
        arr[key[1:]] = value
      # Finally, update this superset of values in self
      self[key[0]] = arr
      return
    # List of integers (case of fancy indexing)
    elif isinstance(key, list):
      # Try to convert to a integer array
      try:
        #key = nd.array(key, dtype=np.int_)
        key = nd.array((operator.index(i) for i in key), dtype='int64')
      except:
        raise IndexError, "key cannot be converted to an array of indices"
      self[key] = value
      return
    # A boolean or integer array (case of fancy indexing)
    elif (hasattr(key, "dtype") or hasattr(key, "eval")):
      # Accept arrays that can have dynd types too
      if hasattr(key, "eval"):
        # Quacks like a dynd array
        typeobj = nd.type_of(key).dtype
      else:
        typeobj = ndt.type(key.dtype)
      if typeobj == ndt.bool:
        # A boolean array
        if len(key) != self.len:
          raise ValueError, "boolean array length must match len(self)"
        self.bool_update(key, value)
        return
      elif typeobj.kind == "int":
        # An integer array
        value = utils.to_ndarray(value, self._dtype, arrlen=len(key))
        # XXX This could be optimised, but it works like this
        for i, item in enumerate(key):
          self[item] = value[i]
        return
      else:
        raise IndexError, \
              "arrays used as indices must be of integer (or boolean) type"
    # All the rest not implemented
    else:
      raise NotImplementedError, "key not supported: %s" % repr(key)

    # Get the corrected values for start, stop, step
    (start, stop, step) = slice(start, stop, step).indices(self.len)

    # Build a numpy object out of value
    vlen = get_len_of_range(start, stop, step)
    if vlen == 0:
      # If range is empty, return immediately
      return
    value = utils.to_ndarray(value, self._dtype, arrlen=vlen)

    # Fill it from data in chunks
    nwrow = 0
    chunklen = self._chunklen
    nchunks = <blz_int_t>cython.cdiv(self._nbytes, self._chunksize)
    if self.leftover > 0:
      nchunks += 1
    for nchunk from 0 <= nchunk < nchunks:
      # Compute start & stop for each block
      startb, stopb, blen = clip_chunk(nchunk, chunklen, start, stop, step)
      if blen == 0:
        continue
      # Modify the data in chunk
      if nchunk == nchunks-1 and self.leftover:
        self.lastchunkarr[startb:stopb:step] = value[nwrow:nwrow+blen]
      else:
        # Get the data chunk
        chunk_ = self.chunks[nchunk]
        self._cbytes -= chunk_.cbytes
        # Get all the values there
        cdata = chunk_[:]
        # Overwrite it with data from value
        cdata[startb:stopb:step] = value[nwrow:nwrow+blen]
        # Replace the chunk
        chunk_ = chunk(cdata, self._dtype, self._bparams,
                       _memory = self._rootdir is None)
        self.chunks[nchunk] = chunk_
        # Update cbytes counter
        self._cbytes += chunk_.cbytes
      nwrow += blen

    # Safety check
    assert (nwrow == vlen)

  # This is a private function that is specific for `eval`
  def _getrange(self, blz_int_t start, blz_int_t blen, object out):
    cdef int chunklen
    cdef blz_int_t startb, stopb
    cdef blz_int_t nwrow, stop, cblen
    cdef blz_int_t schunk, echunk, nchunk, nchunks
    cdef chunk chunk_
    cdef char* data

    # Check that we are inside limits
    nrows = <blz_int_t>cython.cdiv(self._nbytes, self.atomsize)
    if (start + blen) > nrows:
      blen = nrows - start

    # Fill `out` from data in chunks
    nwrow = 0
    stop = start + blen
    nchunks = <blz_int_t>cython.cdiv(self._nbytes, self._chunksize)
    chunklen = cython.cdiv(self._chunksize, self.atomsize)
    schunk = <blz_int_t>cython.cdiv(start, chunklen)
    echunk = <blz_int_t>cython.cdiv((start+blen), chunklen)
    for nchunk from schunk <= nchunk <= echunk:
      # Compute start & stop for each block
      startb = start % chunklen
      stopb = chunklen
      if (start + startb) + chunklen > stop:
        # XXX I still have to explain why this expression works
        # for chunklen > (start + blen)
        stopb = (stop - start) + startb
        # stopb can never be larger than chunklen
        if stopb > chunklen:
          stopb = chunklen
      cblen = stopb - startb
      if cblen == 0:
        continue
      # Get the data chunk and assign it to result array
      if nchunk == nchunks and self.leftover:
        out[nwrow:nwrow+cblen] = self.lastchunkarr[startb:stopb]
      else:
        chunk_ = self.chunks[nchunk]
        data = <char *><Py_uintptr_t>_lowlevel.data_address_of(out)
        chunk_._getitem(startb, stopb, data+nwrow*self.atomsize)
      nwrow += cblen
      start += cblen

  cdef void bool_update(self, boolarr, value):
    """Update self in positions where `boolarr` is true with `value` array."""
    cdef int chunklen
    cdef blz_int_t startb, stopb
    cdef blz_int_t nchunk, nchunks, nrows
    cdef blz_int_t nwrow, blen, vlen, n
    cdef chunk chunk_
    cdef object cdata, boolb

    #vlen = boolarr.sum()   # number of true values in bool array
    # XXX workaround until dynd would implement a sum() method
    if isinstance(boolarr, barray):
      boolarr = boolarr[:]
    if not hasattr(boolarr, "eval"):
      boolarr = nd.view(boolarr)
    vlen = np.asarray(boolarr).sum()   # number of true values in bool array
    value = utils.to_ndarray(value, self._dtype, arrlen=vlen)

    # Fill it from data in chunks
    nwrow = 0
    chunklen = self._chunklen
    nchunks = <blz_int_t>cython.cdiv(self._nbytes, self._chunksize)
    if self.leftover > 0:
      nchunks += 1
    nrows = <blz_int_t>cython.cdiv(self._nbytes, self.atomsize)
    for nchunk from 0 <= nchunk < nchunks:
      # Compute start & stop for each block
      startb, stopb, _ = clip_chunk(nchunk, chunklen, 0, nrows, 1)
      # Get boolean values for this chunk
      n = nchunk * chunklen
      boolb = boolarr[n+startb:n+stopb]
      #blen = boolb.sum()
      blen = np.asarray(boolb).sum()
      if blen == 0:
        continue
      # Modify the data in chunk
      if nchunk == nchunks-1 and self.leftover:
        #self.lastchunkarr[boolb] = value[nwrow:nwrow+blen]
        # dynd does not support fancy indexing yet...
        for i, b in enumerate(boolb):
          if b:
            self.lastchunkarr[i] = value[nwrow+i]
      else:
        # Get the data chunk
        chunk_ = self.chunks[nchunk]
        self._cbytes -= chunk_.cbytes
        # Get all the values there
        cdata = chunk_[:]
        # Overwrite it with data from value
        #cdata[boolb] = value[nwrow:nwrow+blen]
        # dynd does not support fancy indexing yet...
        for i, b in enumerate(boolb):
          if b:
            cdata[i] = value[nwrow+i]
        # Replace the chunk
        chunk_ = chunk(cdata, self._dtype, self._bparams,
                       _memory = self._rootdir is None)
        self.chunks[nchunk] = chunk_
        # Update cbytes counter
        self._cbytes += chunk_.cbytes
      nwrow += blen

    # Safety check
    assert (nwrow == vlen)

  def __iter__(self):

    if not self.sss_mode:
      self.start = 0
      self.stop = <blz_int_t>cython.cdiv(self._nbytes, self.atomsize)
      self.step = 1
    if not (self.sss_mode or self.where_mode or self.wheretrue_mode):
      self.nhits = 0
      self.limit = _MAXINT
      self.skip = 0
    # Initialize some internal values
    self.startb = 0
    self.nrowsread = self.start
    self._nrow = self.start - self.step
    self._row = -1  # a sentinel
    if self.where_mode and isinstance(self.where_arr, barray):
      self.nrowsinbuf = self.where_arr.chunklen
    else:
      self.nrowsinbuf = self._chunklen

    return self

  def iter(self, start=0, stop=None, step=1, limit=None, skip=0):
    """
    iter(start=0, stop=None, step=1, limit=None, skip=0)

    Iterator with `start`, `stop` and `step` bounds.

    Parameters
    ----------
    start : int
        The starting item.
    stop : int
        The item after which the iterator stops.
    step : int
        The number of items incremented during each iteration.  Cannot be
        negative.
    limit : int
        A maximum number of elements to return.  The default is return
        everything.
    skip : int
        An initial number of elements to skip.  The default is 0.

    Returns
    -------
    out : iterator

    See Also
    --------
    where, wheretrue

    """
    # Check limits
    if step <= 0:
      raise NotImplementedError, "step param can only be positive"
    self.start, self.stop, self.step = \
        slice(start, stop, step).indices(self.len)
    self.reset_sentinels()
    self.sss_mode = True
    if limit is not None:
      self.limit = limit + skip
    self.skip = skip
    return iter(self)

  def wheretrue(self, limit=None, skip=0):
    """
    wheretrue(limit=None, skip=0)

    Iterator that returns indices where this object is true.

    This is currently only useful for boolean barrays that are unidimensional.

    Parameters
    ----------
    limit : int
        A maximum number of elements to return.  The default is return
        everything.
    skip : int
        An initial number of elements to skip.  The default is 0.

    Returns
    -------
    out : iterator

    See Also
    --------
    iter, where

    """
    # Check self
    if self._dtype.dtype != ndt.bool:
      raise ValueError, "`self` is not an array of booleans"
    if self.ndim > 1:
      raise NotImplementedError, "`self` is not unidimensional"
    self.reset_sentinels()
    self.wheretrue_mode = True
    if limit is not None:
      self.limit = limit + skip
    self.skip = skip
    return iter(self)

  def where(self, boolarr, limit=None, skip=0):
    """
    where(boolarr, limit=None, skip=0)

    Iterator that returns values of this object where `boolarr` is true.

    This is currently only useful for boolean barrays that are unidimensional.

    Parameters
    ----------
    boolarr : a barray or NumPy array of boolean type
        The boolean values.
    limit : int
        A maximum number of elements to return.  The default is return
        everything.
    skip : int
        An initial number of elements to skip.  The default is 0.

    Returns
    -------
    out : iterator

    See Also
    --------
    iter, wheretrue

    """
    # Check input
    if self.ndim > 1:
      raise NotImplementedError, "`self` is not unidimensional"

    boolarr = utils.to_ndarray(boolarr, ndt.bool)

    if len(boolarr) != self.len:
      raise ValueError, "`boolarr` must be of the same length than ``self``"
    self.reset_sentinels()
    self.where_mode = True
    self.where_arr = nd.array(boolarr)
    if limit is not None:
      self.limit = limit + skip
    self.skip = skip
    return iter(self)

  def __next__(self):
    cdef char *vbool
    cdef int nhits_buf
    cdef char *iodata, *data
    cdef blz_int_t stop

    self.nextelement = self._nrow + self.step
    while (self.nextelement < self.stop) and (self.nhits < self.limit):
      if self.nextelement >= self.nrowsread:
        # Skip until there is interesting information
        while self.nextelement >= self.nrowsread + self.nrowsinbuf:
          self.nrowsread += self.nrowsinbuf
        # Compute the end for this iteration
        self.stopb = self.stop - self.nrowsread
        if self.stopb > self.nrowsinbuf:
          self.stopb = self.nrowsinbuf
        self._row = self.startb - self.step

        # Skip chunks with zeros if in wheretrue_mode
        if self.wheretrue_mode and self.check_zeros(self):
          self.nrowsread += self.nrowsinbuf
          self.nextelement += self.nrowsinbuf
          continue

        if self.where_mode:
          # Skip chunks with zeros in where_arr
          if self.check_zeros(self.where_arr):
            self.nrowsread += self.nrowsinbuf
            self.nextelement += self.nrowsinbuf
            continue
          # Read a chunk of the boolean array
          # XXX protection until issue #18 of dynd should be fixed
          stop = self.nrowsread+self.nrowsinbuf
          if stop > len(self.where_arr): stop = len(self.where_arr)
          self.where_buf = self.where_arr[
#            self.nrowsread:self.nrowsread+self.nrowsinbuf]
            self.nrowsread:stop]

        # Read a data chunk
        # XXX protection until issue #18 of dynd should be fixed
        stop = self.nrowsread+self.nrowsinbuf
        if stop > len(self): stop = len(self)
        #self.iobuf = self[self.nrowsread:self.nrowsread+self.nrowsinbuf]
        self.iobuf = self[self.nrowsread:stop]
        self.nrowsread += self.nrowsinbuf

        # Check if we can skip this buffer
        if (self.wheretrue_mode or self.where_mode) and self.skip > 0:
          if self.wheretrue_mode:
            #nhits_buf = self.iobuf.sum()
            # XXX workaround until dynd would implement a sum() method
            nhits_buf = np.asarray(self.iobuf).sum()
          else:
            #nhits_buf = self.where_buf.sum()
            # XXX workaround until dynd would implement a sum() method
            nhits_buf = np.asarray(self.where_buf).sum()
          if (self.nhits + nhits_buf) < self.skip:
            self.nhits += nhits_buf
            self.nextelement += self.nrowsinbuf
            continue

      self._row += self.step
      self._nrow = self.nextelement
      if self._row + self.step >= self.stopb:
        # Compute the start row for the next buffer
        self.startb = (self._row + self.step) % self.nrowsinbuf
      self.nextelement = self._nrow + self.step

      # Return a value depending on the mode we are
      iodata = <char *><Py_uintptr_t>_lowlevel.data_address_of(self.iobuf)
      if self.wheretrue_mode:
        vbool = <char *>(iodata + self._row)
        if vbool[0]:
          self.nhits += 1
          if self.nhits <= self.skip:
            continue
          return self._nrow
        else:
          continue
      if self.where_mode:
        data = <char *><Py_uintptr_t>_lowlevel.data_address_of(self.where_buf)
        vbool = <char *>(data + self._row)
        if not vbool[0]:
            continue
      self.nhits += 1
      if self.nhits <= self.skip:
        continue
      # Return the current value in I/O buffer
      if self.itemsize == self.atomsize:
        # return npdefs.PyArray_GETITEM(
        #   self.iobuf, iodata + self._row*self.atomsize)
        return self.iobuf[self._row]
      else:
        return self.iobuf[self._row]

    else:
      # Release buffers
      self.iobuf = utils.nd_empty_easy((0,), dtype=self._dtype)
      self.where_buf = utils.nd_empty_easy((0,), dtype=ndt.bool)
      self.reset_sentinels()
      raise StopIteration        # end of iteration

  cdef reset_sentinels(self):
    """Reset sentinels for iterator."""
    self.sss_mode = False
    self.wheretrue_mode = False
    self.where_mode = False
    self.where_arr = None
    self.nhits = 0
    self.limit = _MAXINT
    self.skip = 0

  cdef int check_zeros(self, object barr):
    """Check for zeros.  Return 1 if all zeros, else return 0."""
    cdef int bsize
    cdef blz_int_t nchunk
    cdef barray carr
    cdef object ndarr
    cdef chunk chunk_

    if isinstance(barr, barray):
      # Check for zero'ed chunks in barrays
      carr = barr
      nchunk = <blz_int_t>cython.cdiv(self.nrowsread, self.nrowsinbuf)
      if nchunk < len(carr.chunks):
        chunk_ = carr.chunks[nchunk]
        if chunk_.isconstant and chunk_.constant in (0, ''):
          return 1
    else:
      # Check for zero'ed chunks in ndarrays
      ndarr = barr
      bsize = self.nrowsinbuf
      if self.nrowsread + bsize > self.len:
        bsize = self.len - self.nrowsread
      data = <char *><Py_uintptr_t>_lowlevel.data_address_of(ndarr)
      if check_zeros(data + self.nrowsread, bsize):
        return 1
    return 0

  def _update_disk_sizes(self):
    """Update the sizes on-disk."""
    sizes = dict()
    if self._rootdir:
      sizes['shape'] = self.shape
      sizes['nbytes'] = self.nbytes
      sizes['cbytes'] = self.cbytes
      rowsf = os.path.join(self.metadir, SIZES_FILE)
      with open(rowsf, 'wb') as rowsfh:
        rowsfh.write(json.dumps(sizes, ensure_ascii=True).encode('ascii'))
        rowsfh.write(b'\n')

  def flush(self):
    """Flush data in internal buffers to disk.

    This call should typically be done after performing modifications
    (__settitem__(), append()) in persistence mode.  If you don't do this, you
    risk loosing part of your modifications.

    """
    cdef chunk chunk_
    cdef blz_int_t nchunks
    cdef int leftover_atoms

    if self._rootdir is None:
      return

    if self.leftover:
      leftover_atoms = cython.cdiv(self.leftover, self.atomsize)
      chunk_ = chunk(self.lastchunkarr[:leftover_atoms], self.dtype,
                     self.bparams,
                     _memory = self._rootdir is None)
      # Flush this chunk to disk
      self.chunks.flush(chunk_)

    # Finally, update the sizes metadata on-disk
    self._update_disk_sizes()

  # XXX This does not work.  Will have to realize how to properly
  # flush buffers before self going away...
  # def __del__(self):
  #   # Make a flush to disk if this object get disposed
  #   self.flush()

  def __str__(self):
    return array2string(self)

  def __repr__(self):
    snbytes = utils.human_readable_size(self._nbytes)
    scbytes = utils.human_readable_size(self._cbytes)
    cratio = self._nbytes / float(self._cbytes)
    header = "barray(%s, %s)\n" % (self.shape, self.dtype)
    header += "  nbytes: %s; cbytes: %s; ratio: %.2f\n" % (
      snbytes, scbytes, cratio)
    header += "  bparams := %r\n" % self.bparams
    if self._rootdir:
      header += "  rootdir := '%s'\n" % self._rootdir
    fullrepr = header + str(self)
    return fullrepr



## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
