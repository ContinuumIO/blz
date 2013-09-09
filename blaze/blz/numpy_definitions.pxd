# Numpy related definitions used by blz

# API for NumPy objects
cdef extern from "numpy/arrayobject.h":

  # Platform independent types
  ctypedef int npy_intp
  ctypedef signed int npy_int8
  ctypedef unsigned int npy_uint8
  ctypedef signed int npy_int16
  ctypedef unsigned int npy_uint16
  ctypedef signed int npy_int32
  ctypedef unsigned int npy_uint32
  ctypedef signed long long npy_int64
  ctypedef unsigned long long npy_uint64
  ctypedef float npy_float32
  ctypedef double npy_float64

  cdef enum NPY_TYPES:
    NPY_BOOL
    NPY_BYTE
    NPY_UBYTE
    NPY_SHORT
    NPY_USHORT
    NPY_INT
    NPY_UINT
    NPY_LONG
    NPY_ULONG
    NPY_LONGLONG
    NPY_ULONGLONG
    NPY_FLOAT
    NPY_DOUBLE
    NPY_LONGDOUBLE
    NPY_CFLOAT
    NPY_CDOUBLE
    NPY_CLONGDOUBLE
    NPY_OBJECT
    NPY_STRING
    NPY_UNICODE
    NPY_VOID
    NPY_NTYPES
    NPY_NOTYPE

  # Platform independent types
  cdef enum:
    NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64,
    NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64,
    NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128

  # Classes
  ctypedef extern class numpy.dtype [object PyArray_Descr]:
    cdef int type_num, elsize, alignment
    cdef char type, kind, byteorder, hasobject
    cdef object fields, typeobj

  ctypedef extern class numpy.ndarray [object PyArrayObject]:
    cdef char *data
    cdef int nd
    cdef npy_intp *dimensions
    cdef npy_intp *strides
    cdef object base
    cdef dtype descr
    cdef int flags

  # Functions
  object PyArray_GETITEM(object arr, void *itemptr)
  int PyArray_SETITEM(object arr, void *itemptr, object obj)
  dtype PyArray_DescrFromType(int type)
  object PyArray_Scalar(void *data, dtype descr, object base)

  # The NumPy initialization function
  void import_array()

