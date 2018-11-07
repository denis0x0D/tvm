import tvm
import numpy as np

def test_out_of_bounds(index_a, index_b):
    n = tvm.var("n")
    A = tvm.placeholder ((n,), name='A')
    B = tvm.placeholder ((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i + index_a] + B[i + index_b], name='C')
    s = tvm.create_schedule (C.op)
    tgt = "llvm"
    tgt_host = "llvm"
    fadd = tvm.build (s, [A, B, C], tgt, target_host=tgt_host, name="myadd")
    ctx = tvm.context(tgt, 0)
    a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=1024).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(1024, dtype=C.dtype), ctx)
    try:
        fadd (a, b, c)
    except:
        assert True
    else:
        assert False, "Test should handle the exception"

def test_in_bounds():
    n = tvm.var("n")
    A = tvm.placeholder ((n,), name='A')
    B = tvm.placeholder ((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
    s = tvm.create_schedule (C.op)
    tgt = "llvm"
    tgt_host = "llvm"
    fadd = tvm.build (s, [A, B, C], tgt, target_host=tgt_host, name="myadd")
    ctx = tvm.context(tgt, 0)
    a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=1024).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(1024, dtype=C.dtype), ctx)
    try:
        fadd (a, b, c)
    except:
        assert False, "Test should finish correctly"
    else:
        assert True

if __name__=='__main__':
    test_in_bounds()
    test_out_of_bounds(1, 0)
    test_out_of_bounds(0, 1)
    test_out_of_bounds(1, 1)
    test_out_of_bounds(10000, 0)
    test_out_of_bounds(0, 10000)
    test_out_of_bounds(10000, 10000)

