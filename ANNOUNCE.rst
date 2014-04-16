Announcing BLZ 0.6.2
====================

What's new
----------

BLZ supports completely transparent storage on-disk in addition to
memory.  That means that *everything* that can be done with the
in-memory container can be done using the disk as well.

The advantages of a disk-based container is that the addressable space
is much larger than just your available memory.  Also, as BLZ is based
on a chunked and compressed data layout based on the super-fast Blosc
compression library, the data access speed is very good.

The format chosen for the persistence layer is based on the
'bloscpack' library and described in the "Persistent format for BLZ"
chapter of the user manual ('docs/source/persistence-format.rst').
More about Bloscpack here: https://github.com/esc/bloscpack

You may want to know more about BLZ in this blog entry:
http://continuum.io/blog/blz-format

#XXX version-specific blurb XXX#

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

Read the online docs at:
http://blz.pydata.org/blz-manual/index.html

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
