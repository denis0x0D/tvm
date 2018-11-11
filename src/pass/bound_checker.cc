/*!
 *  Copyright (c) 2018 by Contributors
 * \file bounds_checker.cc
 */
// Instrument checker for out of bounds access.

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <vector>
#include <unordered_map>
#include <utility>
#include "bound_checker.h"

namespace tvm {
namespace ir {

class BoundChecker : public IRMutator {
 public:
  BoundChecker() {}

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    if (UpdateIsNeeded(op->buffer_var)) {
      Update(op->buffer_var, op->extents, op->type);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &ex) final {
    if (proceed_store && op->is_intrinsic(intrinsic::tvm_if_then_else)) {
      unsafe_rewrited = true;
    }
    return IRMutator::Mutate_(op, ex);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    bound_collector.clear();
    proceed_store = true;
    unsafe_rewrited = false;
    IRMutator::Mutate_(op, s);
    proceed_store = false;
    // Store should has at least one load.
    if (CanInstrument(op->index, op->buffer_var) && bound_collector.size()) {
      Collect(op->index, op->buffer_var);
      Expr condition = MakeCondition();
      if (!condition.as<StringImm>()) {
        Stmt nop = Evaluate::make(1);
        Stmt then_case =
            Store::make(op->buffer_var, op->value, op->index, op->predicate);
        Stmt else_case =
            AssertStmt::make(condition, StringImm::make(error_message), nop);
        Stmt body = IfThenElse::make(condition, then_case, else_case);
        return body;
      }
    }
    return s;
  }

  Expr Mutate_(const Load *op, const Expr &ex) final {
    if (CanInstrument(op->index, op->buffer_var)) {
      Collect(op->index, op->buffer_var);
    }
    return IRMutator::Mutate_(op, ex);
  }

 private:
  bool UpdateIsNeeded(const VarExpr &buffer_var) const {
    std::lock_guard<std::mutex> lock(BoundCheckerManager::Global()->mutex);
    return (
        buffer_var.defined() &&
        BoundCheckerManager::Global()->mem_to_buffer.count(buffer_var.get()));
  }

  void Update(const VarExpr &buffer_var, const Array<Expr> &new_shape,
              const Type &type) {
    std::lock_guard<std::mutex> lock(BoundCheckerManager::Global()->mutex);
    // Make sure we catch the actual shape. The type could has lanes > 1.
    Array<Expr> actual_shape;
    for (size_t i = 0; i < new_shape.size(); ++i)
      // Cast to unsigned to avoid integer overlow at frist.
      actual_shape.push_back(Mul::make(make_const(UInt(64), type.lanes()),
                                       Cast::make(UInt(64), new_shape[i])));
    BoundCheckerManager::Global()->mem_to_buffer[buffer_var.get()] =
        actual_shape;
  }

  bool IndexIsValid(const Expr &index) const {
    if (!index.defined())
      return false;

    if (const Ramp *ramp_index = index.as<Ramp>()) {
      return ramp_index->base.defined() &&
             ramp_index->base.type().is_scalar() &&
             ramp_index->stride.defined() &&
             ramp_index->stride.type().is_scalar() && (ramp_index->lanes > 0);
    }
    return true;
  }

  bool CanInstrument(const Expr &index, const VarExpr &buffer_var) const {
    std::lock_guard<std::mutex> lock(BoundCheckerManager::Global()->mutex);
    return buffer_var.defined() &&
           BoundCheckerManager::Global()->mem_to_buffer.count(
               buffer_var.get()) &&
           IndexIsValid(index) && !unsafe_rewrited;
  }

  void Collect(Expr index, VarExpr buffer_var) {
    std::lock_guard<std::mutex> lock(BoundCheckerManager::Global()->mutex);
    bound_collector.push_back(std::make_pair(
        index, BoundCheckerManager::Global()->mem_to_buffer[buffer_var.get()]));
  }

  Expr MakeCondition() {
    Expr condition;
    for (size_t i = 0; i < bound_collector.size(); ++i) {
      std::pair<Expr, Array<Expr>> buffer_to_mem = bound_collector[i];

      Expr upper_bound;
      if (buffer_to_mem.second.size()) {
        Array<Expr> shape = buffer_to_mem.second;
        upper_bound = shape[0];
        for (size_t j = 1; j < shape.size(); ++j) {
          upper_bound = Mul::make(upper_bound, shape[j]);
        }
      } else {
        // TODO(denis0x0D): Handle this case.
        return StringImm::make("zero shape");
      }

      Expr index = buffer_to_mem.first;

      if (const Ramp *ramp_index = index.as<Ramp>()) {
        // In case index is base + stride * i.
        // Non inclusive range.
        index = Add::make(
            ramp_index->base,
            Mul::make(ramp_index->stride, make_const(ramp_index->stride.type(),
                                                     ramp_index->lanes - 1)));
      }

      // Try to simplify index and bound.
      index = ir::Simplify(index);
      upper_bound = ir::Simplify(upper_bound);

      // Cast to the same type - signed, to be able to check lower bound.
      index = Cast::make(Int(64), index);
      upper_bound = Cast::make(Int(64), upper_bound);

      // Looks like a lower bound should always be zero after normalization.
      Expr lower_bound = make_zero(index.type());

      Expr current_condition =
          And::make(GE::make(index, lower_bound), LT::make(index, upper_bound));
      condition =
          !i ? current_condition : And::make(condition, current_condition);
    }
    return condition;
  }

  bool proceed_store{false};
  bool unsafe_rewrited{false};
  std::vector<std::pair<Expr, Array<Expr>>> bound_collector;
  const char *const error_message = "OUT OF BOUNDS";
};

Stmt InstrumentBoundCheckers(Stmt stmt) {
  return BoundChecker().Mutate(stmt);
}
}  // namespace ir
}  // namespace tvm
