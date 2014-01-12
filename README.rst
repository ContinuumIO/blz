BLZ: A chunked, compressed, data container (for memory and disk)
================================================================

BLZ is a chunked container for numerical data.  Chunking allows for
efficient enlarging/shrinking of data container.  In addition, it can
also be compressed for reducing memory/disk needs.  The compression
process is carried out internally by Blosc, a high-performance
compressor that is optimized for binary data.

BLZ uses numexpr internally so as to accelerate many vector and query
operations.  numexpr can use optimize the memory usage and use several
cores for doing the computations, so it is blazing fast.  Moreover,
with the introduction of a barray/btable disk-based container (in
version 0.5), it can be used for seamlessly performing out-of-core
computations.

Rational
--------

By using compression, you can deal with more data using the same
amount of memory.  In case you wonder: which is the price to pay in
terms of performance? you should know that nowadays memory access is
the most common bottleneck in many computational scenarios, and CPUs
spend most of its time waiting for data, and having data compressed in
memory can reduce the stress of the memory subsystem.

In other words, the ultimate goal for BLZ is not only reducing the
memory needs of large arrays, but also making operations to go faster
than using a traditional ndarray object from NumPy.  That is already
the case for some special cases now, but will happen more generally in
a short future, when BLZ will be able to take advantage of newer
CPUs integrating more cores and wider vector units.

Requisites
----------

- Python >= 2.6
- NumPy >= 1.7
- Cython >= 0.19
- numexpr >= 2.2 (optional, if not present, plain NumPy will be used)
- Blosc >= 1.3.0 (optional, if not present, a minimal Blosc will be used)

Building
--------

Assuming that you have the requisites and a C compiler installed, do::

    $ python setup.py build_ext --inplace

In case you have Blosc installed as an external library (and disregard
the included Blosc sources) you can link with it in a couple of ways.

Using an environment variable::

    $ BLOSC_DIR=/usr/local     (or "set BLOSC_DIR=\blosc" on Win)
    $ export BLOSC_DIR         (not needed on Win)
    $ python setup.py build_ext --inplace

Using a flag::

    $ python setup.py build_ext --inplace --blosc=/usr/local

Testing
-------

After compiling, you can quickly check that the package is sane by
running:

$ PYTHONPATH=.   (or "set PYTHONPATH=." on Windows)
$ export PYTHONPATH    (not needed on Windows)
$ python blz/tests/test_all.py

Installing
----------

Install it as a typical Python package:

$ python setup.py install

Documentation
-------------

Please refer to the docs/ directory.

Also, you may want to look at the bench/ directory for some examples
of use.

** To be completed **

Resources
---------

Visit the main BLZ site repository at:
http://github.com/ContinuumIO/blz

Home of Blosc compressor:
http://www.blosc.org

User's mail list:
blaze-dev@continuum.io

License
-------

Please see BLZ.txt in LICENSES/ directory.

Share your experience
---------------------

Let us know of any bugs, suggestions, gripes, kudos, etc. you may
have.


Francesc Alted
Continuum Analytics Inc.
