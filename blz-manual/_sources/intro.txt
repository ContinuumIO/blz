------------
Introduction
------------

BLZ at glance
=============

``BLZ`` is a Python package that provides containers (called `barray`
and `btable`) for numerical data that can be compressed either
in-memory and on-disk.  It is based on NumPy, and uses it as the
standard data container to communicate with BLZ objects.

The building blocks of BLZ objects are the so-called ``chunks`` that
are bits of data compressed as a whole, but that can be decompressed
partially in order to improve the fetching of small parts of the
array.  This ``chunked`` nature of the BLZ objects, together with a
buffered I/O, makes appends very cheap and fetches reasonably fast
(although the modification of values can be an expensive operation).

The compression/decompression process is carried out internally by
Blosc, a high-performance compressor that is optimized for binary
data.  That ensures maximum performance for I/O operation.

BLZ makes of use numexpr internally so as to accelerate many vector
and query operations.  numexpr can use optimize the memory usage and
use several cores for doing the computations, so it is blazing fast.
Moreover, with the introduction of a barray/btable disk-based
container (in version 0.5), it can be used for seamlessly performing
out-of-core computations.


barray and btable objects
-------------------------

The main objects in the BLZ package are:

  * `barray`: container for homogeneous & heterogeneous (row-wise)
    data

  * `btable`: container for heterogeneous (column-wise) data

A `barray` is very similar to a NumPy `ndarray` in that it supports
the same types and data access interface.  The main difference between
them is that a `barray` can keep data compressed (both in-memory and
on-disk), allowing to deal with larger datasets with the same amount
of RAM/disk.  Another important difference is the chunked nature of
the `barray` that allows data to be appended much more efficiently.

On his hand, a `btable` is also similar to a NumPy ``structured
array``, and shares the same fundamental properties with its `barray`
brother, namely, compression and chunking.  Another difference is that
data is stored in a column-wise order (and not on a row-wise, like the
``structured array``), allowing for very cheap column walking and
handling.  This is of paramount importance when you need to walk,
add or remove columns in wide (and possibly large) in-memory and
on-disk tables --doing this with regular ``structured arrays`` in
NumPy is exceedingly slow.

Also, column-wise ordering turns out that this gives the `btable` a
huge opportunity to improve compression ratio.  This is because data
tends to expose more similarity in elements that sit in the same
column rather than those in the same row, so compressors can generally
do a much better job.


BLZ main features
-----------------

BLZ objects bring several advantages over plain NumPy objects:

  * Data is compressed: they take less storage space.

  * Efficient shrinks and appends: you can shrink or append more data
    at the end of the objects very efficiently (i.e. copies of the
    whole array are not needed).

  * Persistence comes seamlessly integrated, so you can work with
    on-disk arrays almost in the same way than with in-memory ones
    (bar some special attention to flush data being required).

  * `btable` objects have the data arranged column-wise.  This allows
    for much better performance when working with just a few amount of
    columns in big tables, as well as for improving the compression
    ratio.

  * Numexpr-powered: you can operate with compressed data in a fast
    and convenient way.  Blosc ensures that the additional overhead of
    handling compressed data natively is very low.

  * Advanced query capabilities.  The ability of a `btable` object to
    iterate over the rows whose fields fulfill some conditions (and
    evaluated via numexpr) allows to perform queries very efficiently.



BLZ limitations
---------------

BLZ does not currently come with good support in the next areas:

  * Reduced number of operations, at least when compared with NumPy.
    The supported operations boils down to basically vectorized ones
    (i.e. does that are made element-by-element).  Just use BLZ with
    Blaze when you need more computational functionality.

  * Limited broadcast support.  For example, NumPy lets you operate
    seamlessly with arrays of different shape (as long as they are
    compatible), but you cannot do that with BLZ.  The only object
    that can be broadcasted currently are scalars
    (e.g. ``blz.eval("x+3")``).  Again, Blaze can be used to overcome
    this.

  * Some methods (namely `blz.where()` and `blz.wheretrue()`)
    do not have support for multidimensional arrays.

  * Multidimensional `btable` objects are not supported.  However, as
    the columns of these objects can be fully multidimensional, this
    is not regarded as a grave limitation.
