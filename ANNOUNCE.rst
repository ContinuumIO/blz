Announcing BLZ 0.6
==================

What's new
----------

BLZ has been branched off from the Blaze project
(http://blaze.pydata.org).  BLZ was meant as a persistent format and
library for I/O in Blaze.  BLZ in Blaze was based on previous carray
0.5 and this is why this new version is labeled 0.6.

BLZ supports completely transparent storage on-disk in addition to
memory.  That means that *everything* that can be done with the
in-memory container can be done using the disk instead.

The advantages of a disk-based container is that the addressable space
is much larger than just your available memory.  Also, as BLZ is based
on a chunked and compressed data layout based on the super-fast Blosc
compression library, the data access speed is very good.

The format chosen for the persistence layer is based on the
'bloscpack' library (thanks to Valentin Haenel for his inspiration)
and described in the "Persistent format for BLZ" chapter of the user
manual ('docs/source/persistence-format.rst').  You may want to know
more about BLZ in this blog entry: http://continuum.io/blog/blz-format

In this version, support for Blosc 1.3 has been added, that meaning
that a new `cname` parameter has been added to the `bparams` class, so
that you can select you preferred compressor from 'blosclz', 'lz4',
'lz4hc', 'snappy' and 'zlib'.

Also, many bugs have been fixed, providing a much smoother experience.

CAVEAT: The BLZ/bloscpack format is still evolving, so don't trust on
forward compatibility of the format, at least until 1.0, where the
internal format will be declared frozen.


What it is
----------

BLZ is a chunked container for numerical data.  Chunking allows for
efficient enlarging/shrinking of data container.  In addition, it can
also be compressed for reducing memory/disk needs.  The compression
process is carried out internally by Blosc, a high-performance
compressor that is optimized for binary data.

BLZ can use numexpr internally so as to accelerate many vector and
query operations (although it can use pure NumPy for doing so too)
either from memory or from disk.  In the future, it is planned to use
Numba for blazing fast operation.

BLZ comes with an exhaustive test suite and fully supports both 32-bit
and 64-bit platforms.  Also, it is typically tested on both UNIX and
Windows operating systems.

Resources
---------

Visit the main BLZ site repository at:
http://github.com/ContinuumIO/blz

Home of Blosc compressor:
http://www.blosc.org

User's mail list:
blaze-dev@continuum.io

----

   Enjoy!

.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 70
.. End:
