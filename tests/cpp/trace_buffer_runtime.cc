#include <iostream>
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

using namespace std;
using namespace tvm;

int main(void) {

  tvm::runtime::Module mod =
    tvm::runtime::Module::LoadFromFile("libtest.so");

  tvm::runtime::PackedFunc f = mod.GetFunction("vecadd");
  CHECK(f != nullptr);

  DLTensor* a;
  DLTensor* b;
  DLTensor* c;
  DLTensor* d;
  DLTensor *z;
  int ndim = 2;
  int dtype_code = kDLInt;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[2] = {1024, 1024};

  /* Prepare the input data */
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
    device_type, device_id, &a);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
    device_type, device_id, &b);

  /* Prepare the placeholder for output data */
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
    device_type, device_id, &c);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type,
                device_id, &d);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type,
                device_id, &z);
  using dtype = int32_t;

  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      static_cast<dtype *>(a->data)[i * shape[0] + j] = 1;
      static_cast<dtype *>(b->data)[i * shape[0] + j] = 1;
    }
  }

  /* Call the function */
  f(a, b, c, d, z);
  std::cout << "result " << std::endl;
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      std::cout << static_cast<dtype *>(z->data)[i * shape[0] + j] << " ";
    }
  }

  cout << endl;
  return 0;
}
