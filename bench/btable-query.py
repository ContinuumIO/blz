# Benchmark to compare the times for querying btable objects.  Numexpr
# is needed in order to execute this.  A comparison with SQLite3 and
# PyTables (if installed) is also done.

import sys, math
import os, os.path
import subprocess
import getopt

import sqlite3
import numpy as np
import blz
from time import time

NR = 1e6      # the number of rows
NC = 100      # the number of columns
mv = 1e10     # the mean value for entries (sig digits = 17 - log10(mv))
clevel = 9    # the compression level
show = False  # show statistics
# The query for a btable
squery = "(f2>.9) & ((f8>.3) & (f8<.4))"  # the btable query
# The query for a recarray
nquery = "(t['f2']>.9) & ((t['f8']>.3) & (t['f8']<.4))"  # for a recarray
# A time reference
tref = 0


def show_rss(explain):
    "Show the used time and RSS memory (only works for Linux >= 2.6)."
    global tref
    # Build the command to obtain memory info
    newtref = time()
    print "Time (%20s) --> %.3f" % (explain, newtref-tref),
    tref = newtref
    if show:
        cmd = "cat /proc/%s/status" % os.getpid()
        sout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
        for line in sout:
            if line.startswith("VmRSS:"):
                vmrss = int(line.split()[1]) // 1024
        print "\t(Resident memory: %d MB)" % vmrss
    else:
        print

def enter():
    global tref
    tref = time()

def after_create(mess=""):
    global tref
    if mess: mess = ", "+mess
    show_rss("creation"+mess)

def after_query(mess=""):
    global tref
    if mess: mess = ", "+mess
    show_rss("query"+mess)


def test_numpy():
    enter()
    t = np.fromiter((mv+np.random.rand(NC)-mv for i in xrange(int(NR))),
                    dtype=dt)
    after_create()
    out = np.fromiter(((row['f1'],row['f3']) for row in t[eval(nquery)]),
                      dtype="f8,f8")
    after_query()
    return out


def test_numexpr():
    import numexpr as ne
    enter()
    t = np.fromiter((mv+np.random.rand(NC)-mv for i in xrange(int(NR))),
                    dtype=dt)
    after_create()

    map_field = dict(("f%s"%i, t["f%s"%i]) for i in range(NC))
    out = np.fromiter(((row['f1'],row['f3']) for row in
                       t[ne.evaluate(squery, map_field)]),
                      dtype="f8,f8")
    after_query()
    return out


def test_btable(clevel):
    enter()
    tc = blz.fromiter((mv+np.random.rand(NC)-mv for i in xrange(int(NR))),
                      dtype=dt,
                      bparams=blz.bparams(clevel, cname='lz4'),
                      count=int(NR))
    after_create()

    out = np.fromiter((row for row in tc.where(squery, 'f1,f3')),
                      dtype="f8,f8")
    after_query()
    return out


def test_sqlite():
    enter()
    sqlquery = "(f2>.9) and ((f8>.3) and (f8<.4))"  # the query

    con = sqlite3.connect(":memory:")

    # Create table
    fields = "(%s)" % ",".join(["f%d real"%i for i in range(NC)])
    con.execute("create table bench %s" % fields)

    # Insert a NR rows of data
    vals = "(%s)" % ",".join(["?" for i in range(NC)])
    with con:
        con.executemany("insert into bench values %s" % vals,
                        (mv+np.random.rand(NC)-mv for i in xrange(int(NR))))
    after_create()

    out = np.fromiter(
        (row for row in con.execute(
        "select f1, f3 from bench where %s" % sqlquery)),
        dtype="f8,f8")
    after_query("non-indexed")

    # Create indexes
    con.execute("create index f1idx on bench (f1)")
    con.execute("create index f2idx on bench (f8)")
    after_create("index")

    out = np.fromiter(
        (row for row in con.execute(
        "select f1, f3 from bench where %s" % sqlquery)),
        dtype="f8,f8")
    after_query("indexed")

    return out


if __name__=="__main__":
    global dt

    usage = """usage: %s [-s] [-m method] [-c ncols] [-r nrows] [-z clevel]
            -s show memory statistics (only for Linux)
            -m select the method: "btable" (def.), "numpy", "numexpr", "sqlite"
            -c the number of columns in table (def. 100)
            -r the number of rows in table (def. 1e6)
            -z the compression level (def. 3)
            \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'sc:r:m:z:')
    except:
        sys.stderr.write(usage)
        sys.exit(1)

    method = "btable"
    # Get the options
    for option in opts:
        if option[0] == '-s':
            if "linux" in sys.platform:
                show = True
        elif option[0] == '-m':
            method = option[1]
        elif option[0] == '-c':
            NC = int(option[1])
        elif option[0] == '-r':
            NR = float(option[1])
        elif option[0] == '-z':
            clevel = int(option[1])

    np.random.seed(12)  # so as to get reproducible results
    # The dtype for tables
    #dt = np.dtype("f8,"*NC)             # aligned fields
    dt = np.dtype("f8,"*(NC-1)+"i1")    # unaligned fields

    if method == "numexpr":
        mess = "numexpr (+numpy)"
    elif method == "btable":
        mess = "btable (clevel=%d)" % clevel
    elif method == "sqlite":
        mess = "sqlite (in-memory)"
    else:
        mess = method
    print "########## Checking method: %s ############" % mess

    print "Querying with %g rows and %d cols" % (NR, NC)
    print "Populating database.  Wait please..."

    if method == "btable":
        out = test_btable(clevel)
    elif method == "numpy":
        out = test_numpy()
    elif method == "numexpr":
        out = test_numexpr()
    elif method == "sqlite":
        out = test_sqlite()
    print "Number of selected elements in query:", len(out)
