# Benchmark for iterators

import numpy as np
import blz
from time import time

N = 1e8       # the number of elements in x
clevel = 5    # the compression level
sexpr = "(x-1) < 10."  # the expression to compute
#sexpr = "((x-1) % 1000) == 0."  # the expression to compute
#sexpr = "(2*x**3+.3*y**2+z+1)<0"  # the expression to compute

bparams = blz.bparams(clevel)

print "Creating inputs with %d elements..." % N

x = np.arange(N)
cx = blz.barray(x, bparams=bparams)
if 'y' not in sexpr:
    ct = blz.btable((cx,), names=['x'])
else:
    y = np.arange(N)
    z = np.arange(N)
    cy = blz.barray(y, bparams=bparams)
    cz = blz.barray(z, bparams=bparams)
    ct = blz.btable((cx, cy, cz), names=['x','y','z'])

print "Evaluating...", sexpr
t0 = time()
cbout = ct.eval(sexpr)
print "Time for evaluation--> %.3f" % (time()-t0,)
print "Converting to numy arrays"
bout = cbout[:]
t = ct[:]

t0 = time()
cbool = blz.barray(bout, bparams=bparams)
print "Time for converting boolean--> %.3f" % (time()-t0,)
print "cbool-->", repr(cbool)

t0 = time()
vals = [v for v in cbool.wheretrue()]
print "Time for wheretrue()--> %.3f" % (time()-t0,)
print "vals-->", len(vals)

print "Starting benchmark now..."
# Retrieve from a ndarray
t0 = time()
vals = [v for v in x[bout]]
print "Time for ndarray--> %.3f" % (time()-t0,)
#print "vals-->", len(vals)

#blz.set_num_threads(blz.ncores//2)

# Retrieve from a barray
t0 = time()
#cvals = [v for v in cx[cbout]]
cvals = [v for v in cx.where(cbout)]
print "Time for barray--> %.3f" % (time()-t0,)
#print "vals-->", len(cvals)
assert vals == cvals

# Retrieve from a structured ndarray
t0 = time()
vals = [tuple(v) for v in t[bout]]
print "Time for structured array--> %.3f" % (time()-t0,)
#print "vals-->", len(vals)

# Retrieve from a btable
t0 = time()
#cvals = [tuple(v) for v in ct[cbout]]
cvals = [v for v in ct.where(cbout)]
print "Time for btable--> %.3f" % (time()-t0,)
#print "vals-->", len(cvals)
assert vals == cvals
