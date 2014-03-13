# Benchmark that compares the times for concatenating arrays with
# compressed arrays vs plain numpy arrays.  The 'numpy' and 'concat'
# styles are for regular numpy arrays, while 'blz' is for barrays.
#
# Call this benchmark as:
#
# python bench/concat.py style
#
# where `style` can be any of 'numpy', 'concat' or 'blz'
#
# You can modify other parameters from the command line if you want:
#
# python bench/concat.py style arraysize nchunks nrepeats clevel, cname
#

import sys, math
import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal
import blz
import time

def concat(data):
    tlen = sum(x.shape[0] for x in data)
    alldata = numpy.empty((tlen,))
    pos = 0
    for x in data:
        step = x.shape[0]
        alldata[pos:pos+step] = x
        pos += step

    return alldata

def append(data, clevel, cname):
    alldata = blz.barray(data[0], bparams=blz.bparams(clevel, cname=cname))
    for carr in data[1:]:
        alldata.append(carr)

    return alldata

if len(sys.argv) < 2:
    print "Pass at least one of these styles: 'numpy', 'concat' or 'blz' "
    sys.exit(1)

style = sys.argv[1]
if len(sys.argv) == 2:
    N, K, T, clevel, cname = (100000, 100, 3, 1, 'lz4')
else:
    N,K,T = [int(arg) for arg in sys.argv[2:5]]
    if len(sys.argv) > 5:
        clevel = int(sys.argv[5])
    else:
        clevel = 0
    if len(sys.argv) > 6:
        cname = int(sys.argv[6])
    else:
        cname = 'lz4'

# The next datasets allow for very high compression ratios
a = [numpy.arange(N, dtype='f8') for _ in range(K)]
print("problem size: (%d) x %d = 10^%g" % (N, K, math.log10(N*K)))

t = time.time()
if style == 'numpy':
    for _ in xrange(T):
        r = numpy.concatenate(a, 0)
elif style == 'concat':
    for _ in xrange(T):
        r = concat(a)
elif style == 'blz':
    for _ in xrange(T):
        r = append(a, clevel, cname)
else:
    print "Unrecognized style: %s" % style
    sys.exit()

t = time.time() - t
print('time for concatenation: %.3fs' % (t / T))

if style == 'blz':
    size = r.cbytes
else:
    size = r.size*r.dtype.itemsize
print("size of the final container: %.3f MB" % (size / float(1024*1024)) )
