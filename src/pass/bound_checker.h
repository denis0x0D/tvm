/*!
 *  Copyright (c) 2018 by Contributors
 * \file bound_checker.h
 * \brief Helper utility for BoundChecker.
 */
#ifndef TVM_PASS_BOUND_CHECKER_H_
#define TVM_PASS_BOUND_CHECKER_H_

#include <unordered_map>
#include <mutex>

namespace tvm {
namespace ir {
struct BoundCheckerManager {
  std::unordered_map<const Variable *, Array<Expr>> mem_to_buffer;
  std::mutex mutex;
  static BoundCheckerManager *Global() {
    static BoundCheckerManager *manager = new BoundCheckerManager();
    return manager;
  }
};
}  // namespace ir
}  // namespace tvm

#endif  // TVM_PASS_BOUND_CHECKER_H_
