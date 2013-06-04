import blaze
from blaze.datadescriptor import dd_as_py
import numpy as np
import unittest
from .common import MayBeUriTest


class TestEphemeral(unittest.TestCase):

    def test_create(self):
        # A default array (backed by BLZ)
        a = blaze.array([ '1', '4444', '22' ], dshape='string',
                        caps={'compress': True})
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), ['1', '4444', '22'])



