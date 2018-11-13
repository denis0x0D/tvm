from nose.tools import raises
import tvm
import numpy as np

@raises(Exception)
def test_out_of_bounds_llvm(index_a, index_b):
    n = tvm.var("n")
    A = tvm.placeholder ((n,), name='A')
    B = tvm.placeholder ((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i + index_a] + B[i + index_b], name='C')
    s = tvm.create_schedule (C.op)
    tgt = "llvm"
    tgt_host = "llvm"
    stmt = tvm.lower (s, [A, B, C], simple_mode=True)
    print (stmt)
    fadd = tvm.build (s, [A, B, C], tgt, target_host=tgt_host, name="myadd")
    ctx = tvm.context(tgt, 0)
    a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=1024).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(1024, dtype=C.dtype), ctx)
    fadd (a, b, c)

def test_in_bounds_llvm():
    n = tvm.var("n")
    A = tvm.placeholder ((n,), name='A')
    B = tvm.placeholder ((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
    s = tvm.create_schedule (C.op)
    tgt = "llvm"
    tgt_host = "llvm"
    stmt = tvm.lower (s, [A, B, C], simple_mode=True)
    print (stmt)
    fadd = tvm.build (s, [A, B, C], tgt, target_host=tgt_host, name="myadd")
    ctx = tvm.context(tgt, 0)
    a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=1024).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(1024, dtype=C.dtype), ctx)
    fadd (a, b, c)

@raises(Exception)
def test_out_of_bounds_vectorize_llvm(nn, index_a, index_b):
    n = tvm.convert(nn)
    a = tvm.placeholder((n), name='a')
    b = tvm.placeholder((n), name='b')
    c = tvm.compute((n,), lambda i: a[i + index_a] + b[i + index_b], name='c')
    s = tvm.create_schedule(c.op)
    xo, xi = s[c].split(c.op.axis[0], factor=8)
    s[c].parallel(xo)
    s[c].vectorize(xi)
    tgt = "llvm"
    tgt_host = "llvm"
    stmt = tvm.lower (s, [a, b, c], simple_mode=True)
    print (stmt)
    f = tvm.build(s, [a, b, c], tgt, target_host=tgt_host, name="myaddvec")
    ctx = tvm.cpu(0)
    n = nn
    a = tvm.nd.array(np.random.uniform(size=(n)).astype(a.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(n)).astype(a.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=c.dtype), ctx)
    f(a, b, c)

def test_in_bounds_vectorize_llvm(n, lanes):
    A = tvm.placeholder((n,), name='A', dtype="float32x%d" % lanes)
    B = tvm.compute((n,), lambda i: A[i], name='B')
    C = tvm.compute((n,), lambda i: B[i] + tvm.const(1, A.dtype), name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], nparts=2)
    _, xi = s[C].split(xi, factor=2)
    s[C].parallel(xo)
    s[C].vectorize(xi)
    s[B].compute_at(s[C], xo)
    xo, xi = s[B].split(B.op.axis[0], factor=2)
    s[B].vectorize(xi)
    # build and invoke the kernel.
    stmt = tvm.lower (s, [A, C], "llvm", simple_mode=True)
    print (stmt)
    f = tvm.build(s, [A, C], "llvm")
    ctx = tvm.cpu(0)
    # launch the kernel.
    a = tvm.nd.empty((n,), A.dtype).copyfrom(
        np.random.uniform(size=(n, lanes)))
    c = tvm.nd.empty((n,), C.dtype, ctx)
    f(a, c)
    tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)

if __name__ == "__main__":
    with tvm.build_config(instrument_bound_checkers=True):
        test_in_bounds_llvm()
        # upper bound
        test_out_of_bounds_llvm(1, 0)
        test_out_of_bounds_llvm(0, 1)
        test_out_of_bounds_llvm(1, 1)
        test_out_of_bounds_llvm(10000, 0)
        test_out_of_bounds_llvm(0, 10000)
        test_out_of_bounds_llvm(10000, 10000)
        # lower bound
        test_out_of_bounds_llvm(-1, 0)
        test_out_of_bounds_llvm(0, -1)
        test_out_of_bounds_llvm(-1, -1)
        test_out_of_bounds_llvm(-10000, 0)
        test_out_of_bounds_llvm(0, -10000)
        test_out_of_bounds_llvm(-10000, -10000)
        # vectorization upper bound
        test_out_of_bounds_vectorize_llvm(1024, 1000, 0)
        test_out_of_bounds_vectorize_llvm(1024, 0, 10000)
        # vectorization lower bound
        test_out_of_bounds_vectorize_llvm(1024, -1000, 0)
        test_out_of_bounds_vectorize_llvm(1024, 0, -10000)
        # placeholder has lanes > 1
        test_in_bounds_vectorize_llvm(512, 2)
