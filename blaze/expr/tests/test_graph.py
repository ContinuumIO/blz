import unittest

from blaze import dshape, array
from blaze.ops.ufuncs import add, mul

from dynd import nd, ndt

class TestGraph(unittest.TestCase):

    def test_graph(self):
        a = array(nd.range(10, dtype=ndt.int32))
        b = array(nd.range(10, dtype=ndt.float32))
        expr = add(a, mul(a, b))
        graph, ctx = expr.expr
        self.assertEqual(len(ctx.params), 2)
        self.assertFalse(ctx.constraints)
        self.assertEqual(graph.dshape, dshape('10, float64'))


if __name__ == '__main__':
    unittest.main()