# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys
import struct
import os, os.path
from unittest import TestCase
from ...py2help import skip

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from dynd import nd, ndt

import blaze.blz as blz
from blaze.blz.blz_ext import chunk
from .common import MayBeDiskTest

is_64bit = (struct.calcsize("P") == 8)

if sys.version_info >= (3, 0):
    xrange = range

# Just memory tests for now

class chunkTest(TestCase):

    def test_getitem_scalar(self):
        a = np.arange(1e3)
        a = nd.view(a)
        dt = nd.type_of(a).element_type
        b = chunk(a, atom=dt, bparams=blz.bparams())
        # print "b[1]->", `b[1]`, type(b[1])
        # XXX The line below does not work on dynd 0.4.post12
        #self.assert_(a[1] == b[1], "Values in key 1 are not equal")
        a1, b1 = np.array(a[1]), np.array(b[1])
        self.assert_(a1 == b1, "Values in key 1 are not equal")

    def test_getitem_ranges(self):
        a = np.arange(1e3)
        a = nd.asarray(a)
        dt = nd.type_of(a).element_type
        b = chunk(a, atom=dt, bparams=blz.bparams())
        #print "b[1:3]->", `b[1:3]`
        assert_array_equal(a[1:3], b[1:3], "Arrays are not equal")

    def test_getitem_ranges_steps(self):
        a = np.arange(1e3)
        a = nd.array(a)
        dt = nd.type_of(a).element_type
        b = chunk(a, atom=dt, bparams=blz.bparams())
        #print "b[1:8:3]->", `b[1:8:3]`
        a1, b1 = nd.as_numpy(a[1:8:3]), nd.as_numpy(b[1:8:3])
        #assert_array_equal(a[1:8:3], b[1:8:3], "Arrays are not equal")
        assert_array_equal(a1, b1, "Arrays are not equal")

    def test_getitem_long_ranges(self):
        a = np.arange(1e4)
        a = nd.view(a)
        dt = nd.type_of(a).element_type
        b = chunk(a, atom=dt, bparams=blz.bparams())
        #print "b[1:8000]->", `b[1:8000]`
        assert_array_equal(a[1:8000], b[1:8000], "Arrays are not equal")


class getitemTest(MayBeDiskTest, TestCase):

    def test01a(self):
        """Testing `__getitem()__` method with only a start"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(1)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01b(self):
        """Testing `__getitem()__` method with only a (negative) start"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(-1)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01c(self):
        """Testing `__getitem()__` method with only a (start,)"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        #print "b[(1,)]->", `b[(1,)]`
        self.assert_(a[(1,)] == b[(1,)], "Values with key (1,) are not equal")

    def test01d(self):
        """Testing `__getitem()__` method with only a (large) start"""
        a = np.arange(1e4)
        b = blz.barray(a, rootdir=self.rootdir)
        sl = -2   # second last element
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02a(self):
        """Testing `__getitem()__` method with ranges"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, 3)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02b(self):
        """Testing `__getitem()__` method with ranges (negative start)"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(-3)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02c(self):
        """Testing `__getitem()__` method with ranges (negative stop)"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, -3)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02d(self):
        """Testing `__getitem()__` method with ranges (negative start, stop)"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(-3, -1)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02e(self):
        """Testing `__getitem()__` method with start > stop"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(4, 3, 30)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03a(self):
        """Testing `__getitem()__` method with ranges and steps (I)"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, 80, 3)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03b(self):
        """Testing `__getitem()__` method with ranges and steps (II)"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, 80, 30)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03c(self):
        """Testing `__getitem()__` method with ranges and steps (III)"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(990, 998, 2)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03d(self):
        """Testing `__getitem()__` method with ranges and steps (IV)"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(4, 80, 3000)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04a(self):
        """Testing `__getitem()__` method with long ranges"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=100, rootdir=self.rootdir)
        sl = slice(1, 8000)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04b(self):
        """Testing `__getitem()__` method with no start"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=100, rootdir=self.rootdir)
        sl = slice(None, 8000)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04c(self):
        """Testing `__getitem()__` method with no stop"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=100, rootdir=self.rootdir)
        sl = slice(8000, None)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04d(self):
        """Testing `__getitem()__` method with no start and no stop"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=100, rootdir=self.rootdir)
        sl = slice(None, None, 2)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test05(self):
        """Testing `__getitem()__` method with negative steps"""
        a = np.arange(1e3)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(None, None, -3)
        #print "b[sl]->", `b[sl]`
        self.assertRaises(NotImplementedError, b.__getitem__, sl)

class getitemDiskTest(getitemTest):
    disk = True


class setitemTest(MayBeDiskTest, TestCase):

    def test00a(self):
        """Testing `__setitem()__` method with only one element"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        b[1] = 10.
        a[1] = 10.
        #print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test00b(self):
        """Testing `__setitem()__` method with only one element (tuple)"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        b[(1,)] = 10.
        a[(1,)] = 10.
        #print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test01(self):
        """Testing `__setitem()__` method with a range"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        b[10:100] = np.arange(1e2 - 10.)
        a[10:100] = np.arange(1e2 - 10.)
        #print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test02(self):
        """Testing `__setitem()__` method with broadcasting"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        b[10:100] = 10.
        a[10:100] = 10.
        #print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test03(self):
        """Testing `__setitem()__` method with the complete range"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=10, rootdir=self.rootdir)
        b[:] = np.arange(10., 1e2 + 10.)
        a[:] = np.arange(10., 1e2 + 10.)
        #print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04a(self):
        """Testing `__setitem()__` method with start:stop:step"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=1, rootdir=self.rootdir)
        sl = slice(10, 100, 3)
        b[sl] = 10.
        a[sl] = 10.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04b(self):
        """Testing `__setitem()__` method with start:stop:step (II)"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=1, rootdir=self.rootdir)
        sl = slice(10, 11, 3)
        b[sl] = 10.
        a[sl] = 10.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04c(self):
        """Testing `__setitem()__` method with start:stop:step (III)"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=1, rootdir=self.rootdir)
        sl = slice(96, 100, 3)
        b[sl] = 10.
        a[sl] = 10.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04d(self):
        """Testing `__setitem()__` method with start:stop:step (IV)"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=1, rootdir=self.rootdir)
        sl = slice(2, 99, 30)
        b[sl] = 10.
        a[sl] = 10.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test05(self):
        """Testing `__setitem()__` method with negative step"""
        a = np.arange(1e2)
        b = blz.barray(a, chunklen=1, rootdir=self.rootdir)
        sl = slice(2, 99, -30)
        self.assertRaises(NotImplementedError, b.__setitem__, sl, 3.)

class setitemDiskTest(setitemTest):
    disk = True


class appendTest(MayBeDiskTest, TestCase):

    def test_basic(self):
        a = np.arange(1000)
        b = blz.barray(a, rootdir=self.rootdir)
        b.append(a)
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test_chunklen_1(self):
        a = np.arange(1000)
        b = blz.barray(a, chunklen=1, rootdir=self.rootdir)
        b.append(a)
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test_chunklen_10k_1(self):
        a = np.arange(1000)
        b = blz.barray(a, chunklen=10*1000, rootdir=self.rootdir)
        b.append(a)
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test_chunklen_10k_2(self):
        a = np.arange(100*1000)
        b = blz.barray(a, chunklen=10*1000, rootdir=self.rootdir)
        b.append(a)
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test_100km1(self):
        a = np.arange(1000*1000)
        b = blz.barray(a, chunklen=100*1000-1, rootdir=self.rootdir)
        b.append(a)
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test_large(self):
        a = np.arange(1e4)
        c = np.arange(2e5)
        b = blz.barray(a, rootdir=self.rootdir)
        b.append(c)
        d = np.concatenate((a, c))
        assert_array_equal(d, b[:], "Arrays are not equal")

class appendDiskTest(appendTest):
    disk = True


class trimTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing `trim()` method"""
        b = blz.arange(1e3, rootdir=self.rootdir)
        b.trim(3)
        a = np.arange(1e3-3)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01(self):
        """Testing `trim()` method (small chunklen)"""
        b = blz.arange(1e2, chunklen=2, rootdir=self.rootdir)
        b.trim(5)
        a = np.arange(1e2-5)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing `trim()` method (large trim)"""
        a = np.arange(2)
        b = blz.arange(1e4, rootdir=self.rootdir)
        b.trim(1e4-2)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test03(self):
        """Testing `trim()` method (complete trim)"""
        a = np.arange(0.)
        b = blz.arange(1e4, rootdir=self.rootdir)
        b.trim(1e4)
        #print "b->", `b`
        self.assert_(len(a) == len(b), "Lengths are not equal")

    def test04(self):
        """Testing `trim()` method (trimming more than available items)"""
        a = np.arange(0.)
        b = blz.arange(1e4, rootdir=self.rootdir)
        #print "b->", `b`
        self.assertRaises(ValueError, b.trim, 1e4+1)

    def test05(self):
        """Testing `trim()` method (trimming zero items)"""
        a = np.arange(1e1)
        b = blz.arange(1e1, rootdir=self.rootdir)
        b.trim(0)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test06(self):
        """Testing `trim()` method (negative number of items)"""
        a = np.arange(2e1)
        b = blz.arange(1e1, rootdir=self.rootdir)
        b.trim(-10)
        a[10:] = 0
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

class trimDiskTest(trimTest):
    disk = True


class resizeTest(MayBeDiskTest):

    def test00a(self):
        """Testing `resize()` method (decrease)"""
        b = blz.arange(self.N, rootdir=self.rootdir)
        b.resize(self.N-3)
        a = np.arange(self.N-3)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test00b(self):
        """Testing `resize()` method (increase)"""
        b = blz.arange(self.N, rootdir=self.rootdir)
        b.resize(self.N+3)
        a = np.arange(self.N+3)
        a[self.N:] = 0
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01a(self):
        """Testing `resize()` method (decrease, large variation)"""
        b = blz.arange(self.N, rootdir=self.rootdir)
        b.resize(3)
        a = np.arange(3)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01b(self):
        """Testing `resize()` method (increase, large variation)"""
        b = blz.arange(self.N, dflt=1, rootdir=self.rootdir)
        b.resize(self.N*3)
        a = np.arange(self.N*3)
        a[self.N:] = 1
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing `resize()` method (zero size)"""
        b = blz.arange(self.N, rootdir=self.rootdir)
        b.resize(0)
        a = np.arange(0)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")


class resize_smallTest(resizeTest, TestCase):
    N = 10

class resize_smallDiskTest(resizeTest, TestCase):
    N = 10
    disk = True

class resize_largeTest(resizeTest, TestCase):
    N = 10000

class resize_largeDiskTest(resizeTest, TestCase):
    N = 10000
    disk = True

class miscTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing __len__()"""
        a = np.arange(111)
        b = blz.barray(a, rootdir=self.rootdir)
        self.assert_(len(a) == len(b), "Arrays do not have the same length")

    def test01(self):
        """Testing __sizeof__() (big arrays)"""
        a = np.arange(2e5)
        b = blz.barray(a, rootdir=self.rootdir)
        #print "size b uncompressed-->", b.nbytes
        #print "size b compressed  -->", b.cbytes
        self.assert_(sys.getsizeof(b) < b.nbytes,
                     "barray does not seem to compress at all")

    def test02(self):
        """Testing __sizeof__() (small arrays)"""
        a = np.arange(111)
        b = blz.barray(a)
        #print "size b uncompressed-->", b.nbytes
        #print "size b compressed  -->", b.cbytes
        self.assert_(sys.getsizeof(b) > b.nbytes,
                     "barray compressed too much??")

class miscDiskTest(miscTest):
    disk = True


class copyTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing copy() without params"""
        a = np.arange(111)
        b = blz.barray(a, rootdir=self.rootdir)
        c = b.copy()
        c.append(np.arange(111, 122))
        self.assert_(len(b) == 111, "copy() does not work well")
        self.assert_(len(c) == 122, "copy() does not work well")
        r = np.arange(122)
        assert_array_equal(c[:], r, "incorrect correct values after copy()")

    def test01(self):
        """Testing copy() with higher compression"""
        a = np.linspace(-1., 1., 1e4)
        b = blz.barray(a, rootdir=self.rootdir)
        c = b.copy(bparams=blz.bparams(clevel=9))
        #print "b.cbytes, c.cbytes:", b.cbytes, c.cbytes
        self.assert_(b.cbytes > c.cbytes, "clevel not changed")

    def test02(self):
        """Testing copy() with lesser compression"""
        a = np.linspace(-1., 1., 1e4)
        b = blz.barray(a, rootdir=self.rootdir)
        c = b.copy(bparams=blz.bparams(clevel=1))
        #print "b.cbytes, c.cbytes:", b.cbytes, c.cbytes
        self.assert_(b.cbytes < c.cbytes, "clevel not changed")

    def test03(self):
        """Testing copy() with no shuffle"""
        a = np.linspace(-1., 1., 1e4)
        b = blz.barray(a, rootdir=self.rootdir)
        c = b.copy(bparams=blz.bparams(shuffle=False))
        #print "b.cbytes, c.cbytes:", b.cbytes, c.cbytes
        self.assert_(b.cbytes < c.cbytes, "shuffle not changed")

class copyDiskTest(copyTest):
    disk = True


class iterTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing `iter()` method"""
        a = np.arange(101)
        b = blz.barray(a, chunklen=2, rootdir=self.rootdir)
        #print "sum iter1->", sum(b)
        #print "sum iter2->", sum((v for v in b))
        self.assert_(sum(a) == sum(b), "Sums are not equal")
        self.assert_(sum((v for v in a)) == sum((v for v in b)),
                     "Sums are not equal")

    def test01a(self):
        """Testing `iter()` method with a positive start"""
        a = np.arange(101)
        b = blz.barray(a, chunklen=2, rootdir=self.rootdir)
        #print "sum iter->", sum(b.iter(3))
        self.assert_(sum(a[3:]) == sum(b.iter(3)), "Sums are not equal")

    def test01b(self):
        """Testing `iter()` method with a negative start"""
        a = np.arange(101)
        b = blz.barray(a, chunklen=2, rootdir=self.rootdir)
        #print "sum iter->", sum(b.iter(-3))
        self.assert_(sum(a[-3:]) == sum(b.iter(-3)), "Sums are not equal")

    def test02a(self):
        """Testing `iter()` method with positive start, stop"""
        a = np.arange(101)
        b = blz.barray(a, chunklen=2, rootdir=self.rootdir)
        #print "sum iter->", sum(b.iter(3, 24))
        self.assert_(sum(a[3:24]) == sum(b.iter(3, 24)), "Sums are not equal")

    def test02b(self):
        """Testing `iter()` method with negative start, stop"""
        a = np.arange(101)
        b = blz.barray(a, chunklen=2, rootdir=self.rootdir)
        #print "sum iter->", sum(b.iter(-24, -3))
        self.assert_(sum(a[-24:-3]) == sum(b.iter(-24, -3)),
                     "Sums are not equal")

    def test02c(self):
        """Testing `iter()` method with positive start, negative stop"""
        a = np.arange(101)
        b = blz.barray(a, chunklen=2, rootdir=self.rootdir)
        #print "sum iter->", sum(b.iter(24, -3))
        self.assert_(sum(a[24:-3]) == sum(b.iter(24, -3)),
                     "Sums are not equal")

    def test03a(self):
        """Testing `iter()` method with only step"""
        a = np.arange(101)
        b = blz.barray(a, chunklen=2, rootdir=self.rootdir)
        #print "sum iter->", sum(b.iter(step=4))
        self.assert_(sum(a[::4]) == sum(b.iter(step=4)),
                     "Sums are not equal")

    def test03b(self):
        """Testing `iter()` method with start, stop, step"""
        a = np.arange(101)
        b = blz.barray(a, chunklen=2, rootdir=self.rootdir)
        #print "sum iter->", sum(b.iter(3, 24, 4))
        self.assert_(sum(a[3:24:4]) == sum(b.iter(3, 24, 4)),
                     "Sums are not equal")

    def test03c(self):
        """Testing `iter()` method with negative step"""
        a = np.arange(101)
        b = blz.barray(a, chunklen=2, rootdir=self.rootdir)
        self.assertRaises(NotImplementedError, b.iter, 0, 1, -3)

    def test04(self):
        """Testing `iter()` method with large zero arrays"""
        a = np.zeros(1e4, dtype='f8')
        b = blz.barray(a, chunklen=100, rootdir=self.rootdir)
        c = blz.fromiter((v for v in b), dtype='f8', count=len(a))
        #print "c ->", repr(c)
        assert_array_equal(a, c[:], "iterator fails on zeros")

    def test05(self):
        """Testing `iter()` method with `limit`"""
        a = np.arange(1e4, dtype='f8')
        b = blz.barray(a, chunklen=100, rootdir=self.rootdir)
        c = blz.fromiter((v for v in b.iter(limit=1010)), dtype='f8',
                        count=1010)
        #print "c ->", repr(c)
        assert_array_equal(a[:1010], c, "iterator fails on zeros")

    def test06(self):
        """Testing `iter()` method with `skip`"""
        a = np.arange(1e4, dtype='f8')
        b = blz.barray(a, chunklen=100, rootdir=self.rootdir)
        c = blz.fromiter((v for v in b.iter(skip=1010)), dtype='f8',
                        count=10000-1010)
        #print "c ->", repr(c)
        assert_array_equal(a[1010:], c, "iterator fails on zeros")

    def test07(self):
        """Testing `iter()` method with `limit` and `skip`"""
        a = np.arange(1e4, dtype='f8')
        b = blz.barray(a, chunklen=100, rootdir=self.rootdir)
        c = blz.fromiter((v for v in b.iter(limit=1010, skip=1010)), dtype='f8',
                        count=1010)
        #print "c ->", repr(c)
        assert_array_equal(a[1010:2020], c, "iterator fails on zeros")

class iterDiskTest(iterTest):
    disk = True


class wheretrueTest(TestCase):

    def test00(self):
        """Testing `wheretrue()` iterator (all true values)"""
        a = np.arange(1, 11) > 0
        b = blz.barray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.wheretrue()]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test01(self):
        """Testing `wheretrue()` iterator (all false values)"""
        a = np.arange(1, 11) < 0
        b = blz.barray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.wheretrue()]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test02(self):
        """Testing `wheretrue()` iterator (all false values, large array)"""
        a = np.arange(1, 1e5) < 0
        b = blz.barray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.wheretrue()]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test03(self):
        """Testing `wheretrue()` iterator (mix of true/false values)"""
        a = np.arange(1, 11) > 5
        b = blz.barray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.wheretrue()]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test04(self):
        """Testing `wheretrue()` iterator with `limit`"""
        a = np.arange(1, 11) > 5
        b = blz.barray(a)
        wt = a.nonzero()[0].tolist()[:3]
        cwt = [i for i in b.wheretrue(limit=3)]
        #print "numpy ->", a.nonzero()[0].tolist()[:3]
        #print "where ->", [i for i in b.wheretrue(limit=3)]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test05(self):
        """Testing `wheretrue()` iterator with `skip`"""
        a = np.arange(1, 11) > 5
        b = blz.barray(a)
        wt = a.nonzero()[0].tolist()[2:]
        cwt = [i for i in b.wheretrue(skip=2)]
        #print "numpy ->", a.nonzero()[0].tolist()[2:]
        #print "where ->", [i for i in b.wheretrue(skip=2)]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test06(self):
        """Testing `wheretrue()` iterator with `limit` and `skip`"""
        a = np.arange(1, 11) > 5
        b = blz.barray(a)
        wt = a.nonzero()[0].tolist()[2:4]
        cwt = [i for i in b.wheretrue(skip=2, limit=2)]
        #print "numpy ->", a.nonzero()[0].tolist()[2:4]
        #print "where ->", [i for i in b.wheretrue(limit=2,skip=2)]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test07(self):
        """Testing `wheretrue()` iterator with `limit` and `skip` (zeros)"""
        a = np.arange(10000) > 5000
        b = blz.barray(a, chunklen=100)
        wt = a.nonzero()[0].tolist()[1020:2040]
        cwt = [i for i in b.wheretrue(skip=1020, limit=1020)]
        # print "numpy ->", a.nonzero()[0].tolist()[1020:2040]
        # print "where ->", [i for i in b.wheretrue(limit=1020,skip=1020)]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")


class whereTest(TestCase):

    def test00(self):
        """Testing `where()` iterator (all true values)"""
        a = np.arange(1, 11)
        b = blz.barray(a)
        wt = [v for v in a if v>0]
        cwt = [v for v in b.where(a>0)]
        #print "numpy ->", [v for v in a if v>0]
        #print "where ->", [v for v in b.where(a>0)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test01(self):
        """Testing `where()` iterator (all false values)"""
        a = np.arange(1, 11)
        b = blz.barray(a)
        wt = [v for v in a if v<0]
        cwt = [v for v in b.where(a<0)]
        #print "numpy ->", [v for v in a if v<0]
        #print "where ->", [v for v in b.where(a<0)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test02a(self):
        """Testing `where()` iterator (mix of true/false values, I)"""
        a = np.arange(1, 11)
        b = blz.barray(a)
        wt = [v for v in a if v<=5]
        cwt = [v for v in b.where(a<=5)]
        #print "numpy ->", [v for v in a if v<=5]
        #print "where ->", [v for v in b.where(a<=5)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test02b(self):
        """Testing `where()` iterator (mix of true/false values, II)"""
        a = np.arange(1, 11)
        b = blz.barray(a)
        wt = [v for v in a if v<=5 and v>2]
        cwt = [v for v in b.where((a<=5) & (a>2))]
        #print "numpy ->", [v for v in a if v<=5 and v>2]
        #print "where ->", [v for v in b.where((a<=5) & (a>2))]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test02c(self):
        """Testing `where()` iterator (mix of true/false values, III)"""
        a = np.arange(1, 11)
        b = blz.barray(a)
        wt = [v for v in a if v<=5 or v>8]
        cwt = [v for v in b.where((a<=5) | (a>8))]
        #print "numpy ->", [v for v in a if v<=5 or v>8]
        #print "where ->", [v for v in b.where((a<=5) | (a>8))]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test03(self):
        """Testing `where()` iterator (using a boolean array)"""
        a = np.arange(1, 11)
        b = blz.barray(a)
        wt = [v for v in a if v<=5]
        cwt = [v for v in b.where(blz.barray(a<=5))]
        #print "numpy ->", [v for v in a if v<=5]
        #print "where ->", [v for v in b.where(blz.barray(a<=5))]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test04(self):
        """Testing `where()` iterator using `limit`"""
        a = np.arange(1, 11)
        b = blz.barray(a)
        wt = [v for v in a if v<=5][:3]
        cwt = [v for v in b.where(blz.barray(a<=5), limit=3)]
        #print "numpy ->", [v for v in a if v<=5][:3]
        #print "where ->", [v for v in b.where(blz.barray(a<=5), limit=3)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test05(self):
        """Testing `where()` iterator using `skip`"""
        a = np.arange(1, 11)
        b = blz.barray(a)
        wt = [v for v in a if v<=5][2:]
        cwt = [v for v in b.where(blz.barray(a<=5), skip=2)]
        #print "numpy ->", [v for v in a if v<=5][2:]
        #print "where ->", [v for v in b.where(blz.barray(a<=5), skip=2)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test06(self):
        """Testing `where()` iterator using `limit` and `skip`"""
        a = np.arange(1, 11)
        b = blz.barray(a)
        wt = [v for v in a if v<=5][1:4]
        cwt = [v for v in b.where(blz.barray(a<=5), limit=3, skip=1)]
        #print "numpy ->", [v for v in a if v<=5][1:4]
        #print "where ->", [v for v in b.where(blz.barray(a<=5),
        #                                      limit=3, skip=1)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test07(self):
        """Testing `where()` iterator using `limit` and `skip` (zeros)"""
        a = np.arange(10000)
        b = blz.barray(a,)
        wt = [v for v in a if v<=5000][1010:2020]
        cwt = [v for v in b.where(blz.barray(a<=5000, chunklen=100),
                                  limit=1010, skip=1010)]
        # print "numpy ->", [v for v in a if v>=5000][1010:2020]
        # print "where ->", [v for v in b.where(blz.barray(a>=5000,chunklen=100),
        #                                       limit=1010, skip=1010)]
        self.assert_(wt == cwt, "where() does not work correctly")


class fancy_indexing_getitemTest(TestCase):

    def test00(self):
        """Testing fancy indexing (short list)"""
        a = np.arange(1,111)
        b = blz.barray(a)
        c = b[[3,1]]
        r = a[[3,1]]
        assert_array_equal(c[:], r, "fancy indexing does not work correctly")

    def test01(self):
        """Testing fancy indexing (large list, numpy)"""
        a = np.arange(1,1e4)
        b = blz.barray(a)
        idx = np.random.randint(1000, size=1000)
        c = b[idx]
        r = a[idx]
        assert_array_equal(c[:], r, "fancy indexing does not work correctly")

    def test02(self):
        """Testing fancy indexing (empty list)"""
        a = np.arange(101)
        b = blz.barray(a)
        c = b[[]]
        r = a[[]]
        assert_array_equal(c[:], r, "fancy indexing does not work correctly")

    def test03(self):
        """Testing fancy indexing (list of floats)"""
        a = np.arange(1,101)
        b = blz.barray(a)
        idx = [1.1, 3.3]
        self.assertRaises(IndexError, b.__getitem__, idx)

    def test04(self):
        """Testing fancy indexing (list of floats, numpy)"""
        a = np.arange(1,101)
        b = blz.barray(a)
        idx = np.array([1.1, 3.3], dtype='f8')
        self.assertRaises(IndexError, b.__getitem__, idx)

    def test05(self):
        """Testing `where()` iterator (using bool in fancy indexing)"""
        a = np.arange(1, 110)
        b = blz.barray(a, chunklen=10)
        wt = a[a<5]
        cwt = b[a<5]
        #print "numpy ->", a[a<5]
        #print "where ->", b[a<5]
        assert_array_equal(wt, cwt[:], "where() does not work correctly")

    def test06(self):
        """Testing `where()` iterator (using array bool in fancy indexing)"""
        a = np.arange(1, 110)
        b = blz.barray(a, chunklen=10)
        wt = a[(a<5)|(a>9)]
        cwt = b[blz.barray((a<5)|(a>9))]
        #print "numpy ->", a[(a<5)|(a>9)]
        #print "where ->", b[blz.barray((a<5)|(a>9))]
        assert_array_equal(wt, cwt[:], "where() does not work correctly")


class fancy_indexing_setitemTest(TestCase):

    def test00(self):
        """Testing fancy indexing with __setitem__ (small values)"""
        a = np.arange(1,111)
        b = blz.barray(a, chunklen=10)
        sl = [3, 1]
        b[sl] = (10, 20)
        a[sl] = (10, 20)
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test01(self):
        """Testing fancy indexing with __setitem__ (large values)"""
        a = np.arange(1,1e3)
        b = blz.barray(a, chunklen=10)
        sl = [0, 300, 998]
        b[sl] = (5, 10, 20)
        a[sl] = (5, 10, 20)
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test02(self):
        """Testing fancy indexing with __setitem__ (large list)"""
        a = np.arange(0,1000)
        b = blz.barray(a, chunklen=10)
        sl = np.random.randint(0, 1000, size=3*30)
        vals = np.random.randint(1, 1000, size=3*30)
        b[sl] = vals
        a[sl] = vals
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test03(self):
        """Testing fancy indexing with __setitem__ (bool array)"""
        a = np.arange(1,1e2)
        b = blz.barray(a, chunklen=10)
        sl = a > 5
        b[sl] = 3.
        a[sl] = 3.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test04(self):
        """Testing fancy indexing with __setitem__ (bool barray)"""
        a = np.arange(1,1e2)
        b = blz.barray(a, chunklen=10)
        bc = (a > 5) & (a < 40)
        sl = blz.barray(bc)
        b[sl] = 3.
        a[bc] = 3.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test05(self):
        """Testing fancy indexing with __setitem__ (bool, value not scalar)"""
        a = np.arange(1,1e2)
        b = blz.barray(a, chunklen=10)
        sl = a < 5
        b[sl] = range(6, 10)
        a[sl] = range(6, 10)
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")


class fromiterTest(TestCase):

    def test00(self):
        """Testing fromiter (short iter)"""
        a = np.arange(1,111)
        b = blz.fromiter(iter(a), dtype='i4', count=len(a))
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test01a(self):
        """Testing fromiter (long iter)"""
        N = 1e4
        a = (i for i in xrange(int(N)))
        b = blz.fromiter(a, dtype='f8', count=int(N))
        c = np.arange(N)
        assert_array_equal(b[:], c, "fromiter does not work correctly")

    def test01b(self):
        """Testing fromiter (long iter, chunk is multiple of iter length)"""
        N = 1e4
        a = (i for i in xrange(int(N)))
        b = blz.fromiter(a, dtype='f8', chunklen=1000, count=int(N))
        c = np.arange(N)
        assert_array_equal(b[:], c, "fromiter does not work correctly")

    def test02(self):
        """Testing fromiter (empty iter)"""
        a = np.array([], dtype="f8")
        b = blz.fromiter(iter(a), dtype='f8', count=-1)
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test03(self):
        """Testing fromiter (dtype conversion)"""
        a = np.arange(101, dtype="f8")
        b = blz.fromiter(iter(a), dtype='f4', count=len(a))
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test04a(self):
        """Testing fromiter method with large iterator"""
        N = 10*1000
        a = np.fromiter((i*2 for i in xrange(N)), dtype='f8')
        b = blz.fromiter((i*2 for i in xrange(N)), dtype='f8', count=len(a))
        assert_array_equal(b[:], a, "iterator with a hint fails")

    def test04b(self):
        """Testing fromiter method with large iterator with a hint"""
        N = 10*1000
        a = np.fromiter((i*2 for i in xrange(N)), dtype='f8', count=N)
        b = blz.fromiter((i*2 for i in xrange(N)), dtype='f8', count=N)
        assert_array_equal(b[:], a, "iterator with a hint fails")


class computeMethodsTest(TestCase):

    def test_sum(self):
        a = np.arange(1e5)
        sa = a.sum()
        ac = blz.barray(a)
        sac = np.array(ac.sum())
        self.assert_(sa.dtype == sac.dtype, "sum() is not working correctly.")
        self.assert_(sa == sac, "sum() is not working correctly.")

    def test_sum_i8(self):
        # this tests also that it works with an explicit dtype
        a = np.arange(1e5)
        sa = a.sum(dtype='i8')
        ac = blz.barray(a)
        sac = ac.sum(dtype='i8')
        sac = np.array(sac)
        #print "numpy sum-->", sa
        #print "barray sum-->", sac
        self.assert_(sa.dtype == sac.dtype, "sum() is not working correctly.")
        self.assert_(sa == sac, "sum() is not working correctly.")

    @skip("strings not supported yet with dynd in the core")
    def test_sum_strings(self):
        ac = blz.zeros(10, 'S3')
        self.assertRaises(TypeError, ac.sum)


class arangeTemplate():
    def test_only_stop(self):
        a = np.arange(self.N)
        ac = blz.arange(self.N)
        self.assert_(np.all(a == nd.as_numpy(ac[:])))

    def test_start_stop(self):
        a = np.arange(3, self.N)
        ac = blz.arange(3, self.N)
        self.assert_(np.all(a == nd.as_numpy(ac[:])))

    def test_start_stop_step(self):
        a = np.arange(3, self.N, 4)
        ac = blz.arange(3, self.N, 4)
        self.assert_(np.all(a == nd.as_numpy(ac[:])))

    def test_with_dtype(self):
        a = np.arange(self.N, dtype="i1")
        ac = blz.arange(self.N, dtype="i1")
        self.assert_(np.all(a == nd.as_numpy(ac[:])))

class arange_smallTest(arangeTemplate, TestCase):
    N = 10

class arange_bigTest(arangeTemplate, TestCase):
    N = 1e4


class constructorTest(MayBeDiskTest):

    def test_constructor_int32(self):
        a = np.arange(self.N)
        ac = blz.barray(a, dtype='i4', rootdir=self.rootdir)
        self.assert_(str(ac.dtype) == 'int32')
        a = a.astype('i4')
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(np.all(a == ac[:]))

    def test_zeros(self):
        a = np.zeros(self.N)
        ac = blz.zeros(self.N, rootdir=self.rootdir)
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(np.all(a == ac[:]))

    def test_zeros_i4(self):
        a = np.zeros(self.N, dtype='i4')
        ac = blz.zeros(self.N, dtype='i4', rootdir=self.rootdir)
        #print "dtypes-->", a.dtype, ac.dtype
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(np.all(a == ac[:]))

    @skip("Strings not supported yet")
    def test_zeros_S5(self):
        a = np.zeros(self.N, dtype='S5')
        ac = blz.zeros(self.N, dtype='S5', rootdir=self.rootdir)
        #print "ac-->", `ac`
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(np.all(a == ac[:]))

    def test_ones(self):
        a = np.ones(self.N)
        ac = blz.ones(self.N, rootdir=self.rootdir)
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(np.all(a == ac[:]))

    def test_ones_i4(self):
        a = np.ones(self.N, dtype='i4')
        ac = blz.ones(self.N, dtype='i4', rootdir=self.rootdir)
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(np.all(a == ac[:]))

    @skip("Strings not supported yet")
    def test_ones_S3(self):
        a = np.ones(self.N, dtype='S3')
        ac = blz.ones(self.N, dtype='S3', rootdir=self.rootdir)
        #print "a-->", a, ac
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(np.all(a == ac[:]))

class constructorSmallTest(constructorTest, TestCase):
    N = 10

class constructorSmallDiskTest(constructorTest, TestCase):
    N = 10
    disk = True

class constructorBigTest(constructorTest, TestCase):
    N = 50000

class constructorBigDiskTest(constructorTest, TestCase):
    N = 50000
    disk = True


class dtypesTest(TestCase):

    def test00(self):
        """Testing barray constructor with a float32 `dtype`."""
        a = np.arange(10)
        ac = blz.barray(a, dtype='f4')
        self.assert_(ac.dtype == ndt.float32)
        a = a.astype('f4')
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(np.all(a == ac[:]))

    def test01(self):
        """Testing barray constructor with a `dtype` with an empty input."""
        a = np.array([], dtype='i4')
        ac = blz.barray([], dtype='f4')
        self.assert_(ac.dtype == ndt.float32)
        a = a.astype('f4')
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(np.all(a == ac[:]))

    def test02(self):
        """Testing barray constructor with a plain compound `dtype`."""
        dtype = np.dtype("f4,f8")
        a = np.ones(30000, dtype=dtype)
        dtype_dy = ndt.make_cstruct([ndt.float32, ndt.float64], ['f0', 'f1'])
        ac = blz.barray(a, dtype=dtype_dy)
        self.assert_(str(ac.dtype) == str(dtype_dy))
        #print "ac-->", `ac`
        #assert_array_equal(a, nd.as_numpy(ac[:]), "Arrays are not equal")
        # XXX workaround for allowing a comparison
        b = nd.as_numpy(ac[:])
        self.assert_(all([np.all(a[x] == b[x]) for x in a.dtype.names]))

    def test03(self):
        """Testing barray constructor with a nested compound `dtype`."""
        dtype = np.dtype([('f1', [('f1', 'i2'), ('f2', 'i4')])])
        a = np.ones(3000, dtype=dtype)
        ac = blz.barray(a, dtype=dtype)
        self.assert_(ac.dtype == dtype)
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(a.dtype == ac.dtype)
        #print "ac-->", `ac`
        assert_array_equal(a, ac[:], "Arrays are not equal")

    def test04(self):
        """Testing barray constructor with a string `dtype`."""
        a = np.array(["ale", "e", "aco"], dtype="S4")
        ac = blz.barray(a, dtype='S4')
        self.assert_(ac.dtype == np.dtype('S4'))
        self.assert_(str(a.dtype) == str(ac.dtype))
        self.assert_(a.dtype == ac.dtype)
        #print "ac-->", `ac`
        assert_array_equal(a, ac, "Arrays are not equal")

    def test05(self):
        """Testing barray constructor with a unicode `dtype`."""
        a = np.array([u"aŀle", u"eñe", u"açò"], dtype="U4")
        ac = blz.barray(a, dtype='U4')
        self.assert_(ac.dtype == np.dtype('U4'))
        self.assert_(a.dtype == ac.dtype)
        self.assert_(str(a.dtype) == str(ac.dtype))
        #print "ac-->", `ac`
        assert_array_equal(a, ac, "Arrays are not equal")

    def test07(self):
        """Checking barray constructor from another barray.

        Test introduced after it was seen failing (blaze issue #30)
        """
        types = [np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64,
                 np.float16, np.float32, np.float64,
                 np.complex64, np.complex128]
        if hasattr(np, 'float128'):
            types.extend([np.float128, np.complex256])
        shapes = [(10,), (10,10), (10,10,10)]
        for shape in shapes:
            for t in types:
                a = blz.zeros(shape, t)
                b = blz.barray(a)
                #self.assertEqual(a.dtype, b.dtype)
                self.assert_(str(a.dtype) == str(b.dtype))
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.shape, shape)


class persistenceTest(MayBeDiskTest, TestCase):

    disk = True

    def test01a(self):
        """Creating a barray in "r" mode."""

        N = 10000
        self.assertRaises(RuntimeError, blz.zeros,
                          N, dtype="i1", rootdir=self.rootdir, mode='r')

    def test01b(self):
        """Creating a barray in "w" mode."""

        N = 50000
        cn = blz.zeros(N, dtype="i1", rootdir=self.rootdir)
        self.assert_(len(cn) == N)

        cn = blz.zeros(N-2, dtype="i1", rootdir=self.rootdir, mode='w')
        self.assert_(len(cn) == N-2)

        # Now check some accesses (no errors should be raised)
        cn.append([1,1])
        self.assert_(len(cn) == N)
        cn[1] = 2
        self.assert_(cn[1] == 2)

    def test01c(self):
        """Creating a barray in "a" mode."""

        N = 30003
        cn = blz.zeros(N, dtype="i1", rootdir=self.rootdir)
        self.assert_(len(cn) == N)

        self.assertRaises(RuntimeError, blz.zeros,
                          N-2, dtype="i1", rootdir=self.rootdir, mode='a')

    def test02a(self):
        """Opening a barray in "r" mode."""

        N = 10001
        cn = blz.zeros(N, dtype="i1", rootdir=self.rootdir)
        self.assert_(len(cn) == N)

        cn = blz.barray(rootdir=self.rootdir, mode='r')
        self.assert_(len(cn) == N)

        # Now check some accesses
        self.assertRaises(RuntimeError, cn.__setitem__, 1, 1)
        self.assertRaises(RuntimeError, cn.append, 1)

    def test02b(self):
        """Opening a barray in "w" mode."""

        N = 100001
        cn = blz.zeros(N, dtype="i1", rootdir=self.rootdir)
        self.assert_(len(cn) == N)

        cn = blz.barray(rootdir=self.rootdir, mode='w')
        self.assert_(len(cn) == 0)

        # Now check some accesses (no errors should be raised)
        cn.append([1,1])
        self.assert_(len(cn) == 2)
        cn[1] = 2
        self.assert_(cn[1] == 2)

    def test02c(self):
        """Opening a barray in "a" mode."""

        N = 1000-1
        cn = blz.zeros(N, dtype="i1", rootdir=self.rootdir)
        self.assert_(len(cn) == N)

        cn = blz.barray(rootdir=self.rootdir, mode='a')
        self.assert_(len(cn) == N)

        # Now check some accesses (no errors should be raised)
        cn.append([1,1])
        self.assert_(len(cn) == N+2)
        cn[1] = 2
        self.assert_(cn[1] == 2)
        cn[N+1] = 3
        self.assert_(cn[N+1] == 3)


class iterchunksTest(TestCase):

    @skip('coredumps')
    def test_iterchunks(self):
        N = int(1e4)
        a = blz.fromiter(xrange(N), dtype=np.float64, count=N)
        l, s = 0, 0
        for block in blz.iterblocks(a):
            l += len(block)
            s += block.sum()
        self.assert_(l == N)
        self.assert_(s == (N - 1) * (N / 2))  # as per Gauss summation formula

    @skip('coredumps')
    def test_iterchunks_blen(self):
        N, blen = int(1e4), 100
        a = blz.fromiter(xrange(N), dtype=np.float64, count=N)
        l, s = 0, 0
        for block in blz.iterblocks(a, blen):
            self.assert_(len(block) == blen)
            l += len(block)
            s += block.sum()
        self.assert_(l == N)

    @skip('coredumps')
    def test_iterchunks_blen_start(self):
        N, blen = int(1e4), 100
        a = blz.fromiter(xrange(N), dtype=np.float64, count=N)
        l, s = 0, 0
        for block in blz.iterblocks(a, blen, blen-1):
            l += len(block)
            s += block.sum()
        self.assert_(l == (N - (blen - 1)))
        self.assert_(s == np.arange(blen-1, N).sum())

    @skip('coredumps')
    def test_iterchunks_full(self):
        N, blen = int(1e4), 100
        a = blz.fromiter(xrange(N), dtype=np.float64, count=N)
        l, s = 0, 0
        for block in blz.iterblocks(a, blen, blen-1, 3*blen+2):
            l += len(block)
            s += block.sum()
        self.assert_(l == 2*blen + 3)
        self.assert_(s == np.arange(blen-1, 3*blen+2).sum())


## Local Variables:
## mode: python
## coding: utf-8
## tab-width: 4
## fill-column: 66
## End:
