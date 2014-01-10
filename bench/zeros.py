import numpy as np
import blz
from time import time

N = 1e8
dtype = 'i4'

t0 = time()
a = np.zeros(N, dtype=dtype)
print "Time numpy.zeros() --> %.4f" % (time()-t0)

t0 = time()
ac = blz.zeros(N, dtype=dtype)
#ac = blz.barray(a)
print "Time barray.zeros() --> %.4f" % (time()-t0)

print "ac-->", `ac`

#assert(np.all(a == ac))
