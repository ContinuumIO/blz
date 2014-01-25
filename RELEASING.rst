=============
Releasing BLZ
=============

:Author: Francesc Alted
:Contact: francesc@continuum.io
:Date: 2014-01-13


Preliminaries
-------------

- Make sure that ``RELEASE_NOTES.rst`` and ``ANNOUNCE.rst`` are up to
  date with the latest news in the release.

- Check that ``VERSION`` file contains the correct number.

Testing
-------

- After compiling, run:

$ PYTHONPATH=.   (or "set PYTHONPATH=." on Win)
$ export PYTHONPATH=.  (not needed on Win)
$ python -c "import blz; blz.test(heavy=True)"

- Run the test suite in different platforms (at least Linux and
  Windows) and make sure that all tests passes.

Packaging
---------

- Make the tarball with the command:

  $ python setup.py sdist

  Do a quick check that the tarball is sane.

- Make the binary packages for supported Python versions (2.7 and 3.3
  currently).  Check that installer works correctly.

Upload the new version of the manual
------------------------------------

- Produce the html version of the manual:

  $ cd doc
  $ vi source/config.py   # make sure that version and release are updated
  $ make html
  $ cp -r build/html/* ../../blz-gh-pages/blz-manual
  $ cd ../../blz-gh-pages
  $ git commit -m"Uploading a new version of the manual" -a
  $ git push
  $ cd ../blz

Uploading
---------

- Register and upload it also in the PyPi repository::

    $ python setup.py register
    $ python setup.py sdist upload


Tagging
-------

- Create a tag ``X.Y.Z`` from ``master``.  Use the next message::

    $ git tag -a X.Y.Z -m "Tagging version X.Y.Z"

- Push the tag to the github repo::

    $ git push --tags


Announcing
----------

- Send an announcement to the public Blaze mailing list, numpy list
  and python-announce lists.  Use the ``ANNOUNCE.rst`` file as
  skeleton (or possibly as the definitive version).

Post-release actions
--------------------

- Edit ``VERSION`` in master to increment the version to the next
  minor one (i.e. X.Y.Z --> X.Y.(Z+1).dev).

- Edit docs/source/conf.py and make sure that `release` is set to
  X.Y.(Z+1).dev .

- Create new headers for adding new features in ``RELEASE_NOTES.rst``
  and empty the release-specific information in ``ANNOUNCE.rst`` and
  add this place-holder instead:

  #XXX version-specific blurb XXX#


That's all folks!


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 70
.. End:
