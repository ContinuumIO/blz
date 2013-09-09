========================
 BLZ Developer Document
========================

This document aims to provide some pointers to the developers of BLZ


Physical Structure
==================

BLZ is composed by different components implemented in a mix of Python
and Cython. It also relies on some external components.

BLZ itself is a Python module.

Many of the lower level bits are implemented as a Python extension
module implemented in Cython (blz_ext).

 - definitions.pxd

 - blz_ext.pyx

blz_ext relies in Blosc, that is also included here. Note that source
code is present there, even if it is inside a include directory:

 - blz/include/blosc/*


Classes
=======

class chunk
-----------

Instances of this class contains compressed in-memory data for a single
data chunk.

class chunks
------------

Instances of this class hold and caches barray chunks present in a
directory on-disk.

class barray
------------

This is blz array-like object implemented using blz chunks

