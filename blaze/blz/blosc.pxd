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

# Blosc routines
cdef extern from "blosc.h":

  cdef enum:
    BLOSC_MAX_OVERHEAD,
    BLOSC_VERSION_STRING,
    BLOSC_VERSION_DATE,
    BLOSC_MAX_TYPESIZE

  void init "blosc_init"()
  void destroy "blosc_destroy"()
  void get_versions "blosc_get_versions"(char *version_str, char *version_date)
  int set_nthreads "blosc_set_nthreads"(int nthreads)
  int compress "blosc_compress"(int clevel, int doshuffle, size_t typesize,
                                size_t nbytes, void *src, void *dest,
                                size_t destsize) nogil
  int decompress "blosc_decompress"(void *src, void *dest,
                                    size_t destsize) nogil
  int getitem "blosc_getitem"(void *src, int start, int nitems,
                              void *dest) nogil
  void free_resources "blosc_free_resources"()
  void cbuffer_sizes "blosc_cbuffer_sizes"(void *cbuffer, size_t *nbytes,
                                           size_t *cbytes, size_t *blocksize)
  void cbuffer_metainfo "blosc_cbuffer_metainfo"(void *cbuffer,
                                                 size_t *typesize, int *flags)
  void cbuffer_versions "blosc_cbuffer_versions"(void *cbuffer,
                                                 int *version, int *versionlz)
  void set_blocksize "blosc_set_blocksize"(size_t blocksize)


## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
