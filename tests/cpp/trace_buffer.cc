#include <random>
#include <iomanip>
#include <array>
#include <exception>

#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <tvm/build_module.h>
#include <topi/broadcast.h>
#include <tvm/trace.h>

using namespace std;
#define TYPE Int
#define bits 32

int main()
{
  auto n = tvm::var("n");
  tvm::Array<tvm::Expr> shape = {n, n};
  tvm::Tensor A = tvm::placeholder(shape, tvm::TYPE(bits), "A");
  tvm::Tensor B = tvm::placeholder(shape, tvm::TYPE(bits), "B");
  /*
    tvm::Tensor C = tvm::compute(shape, tvm::FCompute([=](auto i) {
                                   return tvm::trace("A ", i, A(i)) + B(i);
                                 }));
                                 */
  tvm::Buffer xbuffer = tvm::decl_buffer(shape, tvm::TYPE(bits));
  auto C = tvm::compute(
      shape, [&](tvm::Var i, tvm::Var j) { return A[i][j] + B[i][j]; });

  auto Z = tvm::compute(
      shape, [&](tvm::Var i, tvm::Var j) { return C[i][j] + B[i][j]; });

  auto D = tvm::compute(shape, tvm::FCompute([=](auto i) {
                          return tvm::trace("trace buffer", xbuffer);
                        }));

  /* Prepare a function `vecadd` with no optimizations */
  tvm::Schedule s = tvm::create_schedule({C->op, Z->op, D->op});
  tvm::BuildConfig config = tvm::build_config();
  std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
  binds.insert(pair<tvm::Tensor, tvm::Buffer>(A, xbuffer));
  auto args = tvm::Array<tvm::Tensor>({A, B, C, D, Z});
  auto lowered = tvm::lower(s, args, "vecadd", binds, config);

/* Output IR dump to stderr */
  //cerr << lowered[0]->body << endl;

  auto target = tvm::Target::create("llvm");
  auto target_host = tvm::Target::create("llvm");
  tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);

  /* Output LLVM assembly to stdout */
  cout << mod->GetSource("asm") << endl;
  return 0;
}

