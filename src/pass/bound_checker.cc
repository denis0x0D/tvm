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

// FIXME. Should use tvm::Range instead ?
struct IRange {
  IRange(Expr b, Expr e) : begin(b), end(e) {}
  Expr begin;
  Expr end;
};

class BoundCheckersCollector final : public IRVisitor {
public:
  BoundCheckersCollector() {}
  void Visit(const NodeRef &n) final { IRVisitor::Visit(n); }

  void Visit_(const For *op) {
    // Add current range
    ranges.push_back(std::make_shared<IRange>(op->min, op->extent));
    IRVisitor::Visit_(op);
  }

  void Visit_(const Block *op) final {
    size_t current_size = ranges.size();
    IRVisitor::Visit(op->first);
    while (ranges.size() > current_size)
      ranges.pop_back();
    IRVisitor::Visit(op->rest);
  }

  void Visit_(const IfThenElse *op) final {
    size_t current_size = ranges.size();
    IRVisitor::Visit(op->then_case);
    while (ranges.size() > current_size)
      ranges.pop_back();
    IRVisitor::Visit(op->else_case);
  }

  void Visit_(const Store *op) final {
    memory_accesses.insert(std::make_pair(op, ranges));
    indexes.clear();
    IRVisitor::Visit_(op);
    indexes.push_back(op->index);
    memory_accesses_indexes.insert(std::make_pair(op, indexes));
  }

  void Visit_(const Load *op) final {
    indexes.push_back(op->index);
    IRVisitor::Visit_(op);
  }

  ~BoundCheckersCollector() {}

  std::unordered_map<const Node *, std::vector<std::shared_ptr<IRange>>>
      memory_accesses;
  std::vector<std::shared_ptr<IRange>> ranges;

  std::unordered_map<const Node *, std::vector<Expr>> memory_accesses_indexes;
  std::vector<Expr> indexes;
};

class BoundChecker : public IRMutator {
 public:
   BoundChecker(const std::unordered_map<const Node *,
                                         std::vector<std::shared_ptr<IRange>>>
                    &memory_accesses,
                const std::unordered_map<const Node *, std::vector<Expr>>
                    &memory_accesses_indexes)
       : memory_accesses(memory_accesses),
         memory_accesses_indexes(memory_accesses_indexes) {}

   Stmt Mutate_(const Store *op, const Stmt &s) final {
     if (CanInstrument(op)) {
       Expr condition = MakeCondition(op);
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
   
   bool CanInstrument(const Node *op) {
     return memory_accesses.count(op) && memory_accesses_indexes.count(op);
   }

   ~BoundChecker() {}

 private:
   Expr MakeCondition(const Node *node) {
     if (memory_accesses.count(node) && memory_accesses_indexes.count(node)) {
       std::vector<std::shared_ptr<IRange>> ranges = memory_accesses[node];
       std::vector<Expr> indexes = memory_accesses_indexes[node];

       if (!ranges.size() || !indexes.size()) {
         return Expr();
       }

       Expr upper_bound = ranges[0]->end;
       for (size_t i = 1; i < ranges.size(); ++i) {
         upper_bound = Mul::make(upper_bound, ranges[i]->end);
       }

       Expr condition = LT::make(indexes[0], upper_bound);
       for (size_t i = 1; i < indexes.size(); ++i) {
         condition = And::make(condition, LT::make(indexes[i], upper_bound));
       }
       return condition;
     }
     return Expr();
   }

   std::unordered_map<const Node *, std::vector<std::shared_ptr<IRange>>>
       memory_accesses;
   std::unordered_map<const Node *, std::vector<Expr>> memory_accesses_indexes;
   const char *const error_message = "OUT OF BOUNDS";
};

Stmt InstrumentBoundCheckers(Stmt stmt) {
  BoundCheckersCollector collector;
  collector.Visit(stmt);
  BoundChecker bound_checker(collector.memory_accesses,
                             collector.memory_accesses_indexes);
  return bound_checker.Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
