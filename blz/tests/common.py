import unittest
import tempfile
import os, os.path
import glob
import shutil
import numpy as np

# Global variables for the tests
verbose = False
heavy = False


def remove_tree(rootdir):
    # Remove every directory starting with rootdir
    for dir_ in glob.glob(rootdir+'*'):
        shutil.rmtree(dir_)

# Useful superclass for disk-based tests
class MayBeDiskTest():

    disk = False

    def setUp(self):
        if self.disk:
            prefix = 'barray-' + self.__class__.__name__
            self.rootdir = tempfile.mkdtemp(prefix=prefix)
            os.rmdir(self.rootdir)  # tests needs this cleared
        else:
            self.rootdir = None

    def tearDown(self):
        if self.disk:
            remove_tree(self.rootdir)
