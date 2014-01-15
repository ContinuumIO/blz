------------
Installation
------------

BLZ depends on NumPy (>= 1.7) and, optionally, Numexpr (>= 2.2).
Also, if you are going to install from sources, a C compiler.

Installing from PyPI repository
===============================

Do::

  $ easy_install -U blz

or::

  $ pip install -U blz


Installing from tarball sources
===============================

Go to the BLZ main directory and do the typical distutils dance::

  $ python setup.py build_ext -i
  $ export PYTHONPATH=.   # set PYTHONPATH=.  on Windows
  $ python blz/tests/test_all.py
  $ python setup.py install


Installing from the git repository
==================================

If you have cloned the BLZ repository, you can follow the same
procedure than for the tarball above.

Also, you can generate documentation in both pdf and html formats::

  $ cd docs
  $ make pdf      # PDF output in docs/BLZ-manual.pdf
  $ make html     # HTML output in docs/html/


Testing the installation
========================

You can always test the installation from any directory with::

  $ python -c "import blz; blz.test()"
