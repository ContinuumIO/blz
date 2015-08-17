BLZ: A chunked, compressed, data container (for memory and disk)
================================================================

Note on this repo
-----------------

This version of the blz code has been deprecated. The project has changed names to bcolz and can be followed at: https://github.com/Blosc/bcolz

Introduction
------------

BLZ is a chunked container for numerical data.  Chunking allows for
efficient enlarging/shrinking of data container.  In addition, it can
also be compressed for reducing memory/disk needs.  The compression
process is carried out internally by Blosc, a high-performance
compressor that is optimized for binary data.

BLZ uses Blosc (http://www.blosc.org) for data compression and numexpr
(https://github.com/pydata/numexpr) transparently so as to accelerate
many vector and query operations.  Blosc can compress binary data very
efficiently, optimizing memory access, while numexpr focus in reducing
the memory usage and use several cores for doing the computations.
Medium term goal is to leverage the advanced computing capabilities in
Blaze (http://blaze.pydata.org) in addition to numexpr.

Finally, the adoption of the Bloscpack persistent format
(https://github.com/esc/bloscpack) allows the main objects in BLZ
(`barray` / `btable`, see below) to be persisted, so it can be used
for performing out-of-core computations transparently.


`btable`: a columnar store
--------------------------

The main objects in BLZ are `barray` and `btable`.  `barray` is meant
for storing multidimensional homogeneous datasets efficiently.
`barray` objects provide the foundations for building `btable`
objects, where each column is made of a single `barray`.  Facilities
are provided for iterating, filtering and querying `btables` in an
efficient way.  You can find more info about `barray` and `btable` in
the tutorial:

http://blz.pydata.org/blz-manual/tutorial.html


Rational
--------

By using compression, you can deal with more data using the same
amount of memory.  In case you wonder: which is the price to pay in
terms of performance? you should know that nowadays memory access is
the most common bottleneck in many computational scenarios, and CPUs
spend most of its time waiting for data.  Hence having data compressed
in memory can reduce the stress of the memory subsystem.

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
- Blosc >= 1.3.0 (optional, the internal Blosc will be used by default)
- unittest2 (only in the case you are running Python 2.6)


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
running::

    $ PYTHONPATH=.   (or "set PYTHONPATH=." on Windows)
    $ export PYTHONPATH    (not needed on Windows)
    $ python -c"import blz; blz.test()"  # add `heavy=True` if desired


Installing
----------

Install it as a typical Python package:

$ python setup.py install


Documentation
-------------

You can find the online manual at:

http://blz.pydata.org/blz-manual/index.html

Also, you may want to look at the bench/ directory for some examples
of use.


Resources
---------

Visit the main BLZ site repository at:
http://github.com/ContinuumIO/blz

Home of Blosc compressor:
http://www.blosc.org

Home of the numexpr project:
https://github.com/pydata/numexpr

User's mail list:
blaze-dev@continuum.io


License
-------

Please see BLZ.txt in LICENSES/ directory.


Share your experience
---------------------

Let us know of any bugs, suggestions, gripes, kudos, etc. you may
have.


Authors
-------

See the AUTHORS.txt file.
