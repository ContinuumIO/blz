# Benchmark to compare the times for computing expressions by using
# btable objects.  Numexpr is needed in order to execute this.

import math
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numexpr as ne
import blz
from time import time

N = 1e8       # the number of elements in x
clevel = 9    # the compression level
#sexpr = "(x+1)<0"  # the expression to compute
#sexpr = "(2*x**3+.3*y**2+z+1)<0"  # the expression to compute
#sexpr = "((.25*x + .75)*x - 1.5)*x - 2"  # a computer-friendly polynomial
sexpr = "(((.25*x + .75)*x - 1.5)*x - 2)<0"  # a computer-friendly polynomial

print "Creating inputs..."

bparams = blz.bparams(clevel)

x = np.arange(N)
#x = np.linspace(0,100,N)
cx = blz.barray(x, bparams=bparams)
if 'y' not in sexpr:
    t = blz.btable((cx,), names=['x'])
else:
    y = np.arange(N)
    z = np.arange(N)
    cy = blz.barray(y, bparams=bparams)
    cz = blz.barray(z, bparams=bparams)
    t = blz.btable((cx, cy, cz), names=['x','y','z'])

print "Evaluating '%s' with 10^%d points" % (sexpr, int(math.log10(N)))

t0 = time()
out = eval(sexpr)
print "Time for plain numpy--> %.3f" % (time()-t0,)

t0 = time()
out = ne.evaluate(sexpr)
print "Time for numexpr (numpy)--> %.3f" % (time()-t0,)

# Uncomment the next for disabling threading
#ne.set_num_threads(1)
#blz.blosc_set_nthreads(1)
# Seems that this works better if we dividw the number of cores by 2.
# Maybe due to some contention between Numexpr and Blosc?
#blz.set_nthreads(blz.ncores//2)

for kernel in "python", "numexpr":
    t0 = time()
    #cout = t.eval(sexpr, kernel=kernel, bparams=bparams)
    cout = t.eval(sexpr, bparams=bparams)
    print "Time for btable (%s) --> %.3f" % (kernel, time()-t0,)
    #print "cout-->", repr(cout)

#assert_array_equal(out, cout, "Arrays are not equal")
