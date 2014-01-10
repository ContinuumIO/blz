import numpy as np
import blz
from time import time

N = 1e8
#a = np.arange(N, dtype='f8')
a = np.random.randint(0,10,N).astype('bool')

t0 = time()
sa = a.sum()
print "Time sum() numpy --> %.3f" % (time()-t0)

t0 = time()
ac = blz.barray(a, bparams=blz.bparams(9))
print "Time barry conv --> %.3f" % (time()-t0)
print "ac-->", `ac`

t0 = time()
sac = ac.sum()
#sac = ac.sum(dtype=np.dtype('i8'))
print "Time sum() barray --> %.3f" % (time()-t0)

# t0 = time()
# sac = sum(i for i in ac)
# print "Time sum() carray (iter) --> %.3f" % (time()-t0)

print "sa, sac-->", sa, sac, type(sa), type(sac)
assert(sa == sac)
