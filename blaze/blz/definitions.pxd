########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################

"""Here are some definitions for some C headers dependencies.

"""

#-----------------------------------------------------------------------------

# Some helper routines from the Python API
# PythonHelper.h is used to help make
# python 2 and 3 both work.
cdef extern from "PythonHelper.h":

  # special types
  ctypedef int Py_ssize_t

  # references
  void Py_INCREF(object)
  void Py_DECREF(object)

  # To release global interpreter lock (GIL) for threading
  void Py_BEGIN_ALLOW_THREADS()
  void Py_END_ALLOW_THREADS()

  # Functions for integers
  object PyInt_FromLong(long)
  long PyInt_AsLong(object)
  object PyLong_FromLongLong(long long)
  long long PyLong_AsLongLong(object)

  # Functions for floating points
  object PyFloat_FromDouble(double)

  # Functions for strings
  object PyString_FromString(char *)
  object PyString_FromStringAndSize(char *s, int len)
  char *PyString_AsString(object string)
  size_t PyString_GET_SIZE(object string)

  # Functions for lists
  int PyList_Append(object list, object item)

  # Functions for tuples
  object PyTuple_New(int)
  int PyTuple_SetItem(object, int, object)
  object PyTuple_GetItem(object, int)
  int PyTuple_Size(object tuple)

  # Functions for dicts
  int PyDict_Contains(object p, object key)
  object PyDict_GetItem(object p, object key)

  # Functions for objects
  object PyObject_GetItem(object o, object key)
  int PyObject_SetItem(object o, object key, object v)
  int PyObject_DelItem(object o, object key)
  long PyObject_Length(object o)
  int PyObject_Compare(object o1, object o2)
  int PyObject_AsReadBuffer(object obj, void **buffer, Py_ssize_t *buffer_len)

  # Functions for buffers
  object PyBuffer_FromMemory(void *ptr, Py_ssize_t size)

  ctypedef unsigned int Py_uintptr_t

#-----------------------------------------------------------------------------

# Blosc routines
cdef extern from "blosc.h":

  cdef enum:
    BLOSC_MAX_OVERHEAD,
    BLOSC_VERSION_STRING,
    BLOSC_VERSION_DATE,
    BLOSC_MAX_TYPESIZE

  void blosc_init()
  void blosc_destroy()
  void blosc_get_versions(char *version_str, char *version_date)
  int blosc_set_nthreads(int nthreads)
  int blosc_compress(int clevel, int doshuffle, size_t typesize,
                     size_t nbytes, void *src, void *dest,
                     size_t destsize) nogil
  int blosc_decompress(void *src, void *dest, size_t destsize) nogil
  int blosc_getitem(void *src, int start, int nitems, void *dest) nogil
  void blosc_free_resources()
  void blosc_cbuffer_sizes(void *cbuffer, size_t *nbytes,
                           size_t *cbytes, size_t *blocksize)
  void blosc_cbuffer_metainfo(void *cbuffer, size_t *typesize, int *flags)
  void blosc_cbuffer_versions(void *cbuffer, int *version, int *versionlz)
  void blosc_set_blocksize(size_t blocksize)




## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
