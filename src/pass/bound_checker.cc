/*!
 *  Copyright (c) 2018 by Contributors
 * \file bounds_checker.cc
 */
// Instrument checker for out of bounds access.

#include <map>
#include <string>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace tvm {
namespace ir {

extern std::unordered_map<const Variable *, Buffer> *MemoryToBuffer();

class BoundChecker : public IRMutator {
 public:
   BoundChecker() {}

   Stmt Mutate_(const Store *op, const Stmt &s) final {
     bound_collector.clear();
     IRMutator::Mutate_(op, s);
     if (CanInstrument(op->index, op->buffer_var)) {
       Collect(op->index, op->buffer_var);
       Expr condition = MakeCondition();
       Stmt nop = Evaluate::make(1);
       Stmt then_case =
           Store::make(op->buffer_var, op->value, op->index, op->predicate);
       Stmt else_case =
           AssertStmt::make(condition, StringImm::make(error_message), nop);
       Stmt body = IfThenElse::make(condition, then_case, else_case);
       return body;
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
   bool CanInstrument(Expr index, VarExpr buffer_var) {
     return index.defined() && buffer_var.defined() &&
            MemoryToBuffer()->count(buffer_var.get());
   }

   void Collect(Expr index, VarExpr buffer_var) {
     bound_collector.push_back(std::make_pair(
         index, (*MemoryToBuffer())[buffer_var.get()]->shape));
   }

   Expr MakeCondition() {
     if (bound_collector.size()) {
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
           // An error.
           return Expr ();
         }

         Expr index = buffer_to_mem.first;
         Expr current_condition = LT::make(index, upper_bound);
         condition =
             !i ? current_condition : And::make(condition, current_condition);
       }
       return condition;
     }
     // An error.
     return Expr();
   }

   std::vector<std::pair<Expr, Array<Expr>>> bound_collector;
   const char *const error_message = "OUT OF BOUNDS";
};

Stmt InstrumentBoundCheckers(Stmt stmt) { return BoundChecker().Mutate(stmt); }

}  // namespace ir
}  // namespace tvm
