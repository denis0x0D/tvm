/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Trace funciion, which allows to trace data during runtime.
 * \file trace.h
 */

#ifndef TVM_TRACE_H_
#define TVM_TRACE_H_

#include <string>
#include "tvm.h"
#include "../../topi/include/topi/detail/extern.h"

namespace tvm {
static inline std::string get_type(Type type) {
  return type.is_int() ? "i" : "f";
}

#define CHECK_TRACED_TYPE(type)                                                \
  CHECK(type.bits() >= 32 && type.bits() <= 64 &&                              \
        (type.is_int() || type.is_float()))                                    \
      << "Traced value should has type Int or UInt or Float with size 32 or "  \
         "64 bits."

/*!
 * \brief Construct an Expr which allows to trace the data of the specific Tensor.
 *
 * \param str Represents the output message
 *
 * \param indexes An array of indexes
 *
 * \param x Expr to trace
 *
 * \return An expression representing a PackedFunc
 */
inline Expr trace(std::string str, Array<Var> indexes, Expr x) {
  CHECK(x.defined()) << "Traced Expr should be defined";
  Type expr_type = x.type();
  CHECK_TRACED_TYPE(expr_type);
  Array<Expr> call_args{std::string(get_type(expr_type) +
                                    std::to_string(expr_type.bits()) +
                                    "_expr_trace"),
                        ir::StringImm::make(str)};
  for (size_t i = 0; i < indexes.size(); ++i)
    call_args.push_back(indexes[i]);
  call_args.push_back(x);

  return topi::detail::call_packed(call_args);
}

/*!
 * \brief Construct an Expr which allows to trace the data of the specific buffer.
 *
 * \param str Represents the output message
 *
 * \param buffer Buffer to trace
 *
 * \return An expression representing a PackedFunc
 */
inline Expr trace(std::string str, Buffer buffer) {
  Type buffer_type = buffer->dtype;
  CHECK_TRACED_TYPE(buffer_type);
  Array<Expr> call_args{std::string(get_type(buffer_type) +
                                    std::to_string(buffer_type.bits()) +
                                    "_buffer_trace"),
                        ir::StringImm::make(str)};
  for (size_t i = 0; i < buffer->shape.size(); ++i)
    call_args.push_back(buffer->shape[i]);
  call_args.push_back(topi::detail::pack_buffer(buffer));
  // Use call_packed with buffer type to proceed different types.
  return topi::detail::call_packed(buffer_type, call_args);
}
}  // namespace tvm
#endif  // TVM_TRACE_H_
