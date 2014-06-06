__author__ = 'hussa_000'

import blz
import numpy as np

N = 100*1000
ct = blz.fromiter(((i,i*i) for i in xrange(N)), dtype="i4,f8", count=N,rootdir='test4')
new_col = np.linspace(0, 1, 100*1000)
ct.addcol(new_col)
ct.flush()
ct.delcol('f2')
ct.flush()

#trying to open the btable from rootdir produces the error

