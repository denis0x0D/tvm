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

namespace tvm {
namespace ir {

// FIXME. Should use tvm::Range instead ?
struct IRange {
  IRange(Expr b, Expr e) : begin(b), end(e) {}
  Expr begin;
  Expr end;
};

// Collects all indexes and loop ranges the search of the collected
// ranges is 0(n), this should be optimized.
class BoundCheckersCollector final : public IRVisitor {
public:
  BoundCheckersCollector() {}
  void Visit(const NodeRef &n) final { IRVisitor::Visit(n); }

  void Visit_(const For *op) final {
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

  // I assume the store and load always are leafs in the tree.
  void Visit_(const Load *op) final {
    memory_acceses.insert(std::make_pair(op, ranges));
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store *op) final {
    memory_acceses.insert(std::make_pair(op, ranges));
    IRVisitor::Visit_(op);
  }

  ~BoundCheckersCollector() {}

  std::unordered_map<const Node *, std::vector<std::shared_ptr<IRange>>>
      memory_acceses;
  std::vector<std::shared_ptr<IRange>> ranges;
};

/*
class BoundChecker : public IRMutator {
 public:
   BoundChecker(
       std::unordered_map<const Node *, std::vector<std::shared_ptr<IRange>>>
           &memory_acceses)
       : memory_acceses(memory_acceses) {}

   Stmt Mutate_(const For *op, const Stmt &s) final {
     if (op->body.as<Store>() || op->body.as<Block>()) {
       std::cout << " Instrument for body is store " << std::endl;
       Expr condition = MakeCondition(op);
       Stmt nop = Evaluate::make(1);
       Stmt then_case = op->body;
       Stmt else_case =
           AssertStmt::make(condition, StringImm::make(error_message), nop);
       Stmt body = IfThenElse::make(condition, then_case, else_case);
       return For::make(op->loop_var, op->min, op->extent, op->for_type,
                        op->device_api, body);
     } else {
       return IRMutator::Mutate_(op, op->body);
     }
   }

   ~BoundChecker() {}

 private:
   Expr MakeCondition(const Node *node) {
     if (memory_acceses(node).count()) {
       std::vector<std::shared_ptr<IRange>> ranges = memory_acceses[node];
       if (!ranges.size() || !indexes.size()) {
         std::cout << "size is null" << std::endl;
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
     return Expr(); // should be null
   }

   std::unordered_map<
       std::pair<const Node *, std::vector<std::shared_ptr<IRange>>>>
       memory_acceses;
   const char *const error_message = "OUT OF BOUNDS";
};
*/

Stmt InstrumentBoundCheckers(Stmt stmt) {
  BoundCheckersCollector collector;
  collector.Visit(stmt);
  return stmt;
//  BoundChecker bound_checker(collector.scoped_ranges, collector.scoped_indexes);
 // return bound_checker.Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
