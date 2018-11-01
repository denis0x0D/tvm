/*!
 *  Copyright (c) 2018 by Contributors
 * \file trace.c
 * \brief Support trace primitive.
 */
#if defined(__linux__)
#include <tvm/runtime/registry.h>
#include <stdarg.h>
#include <string>
#include <unordered_set>
#include <type_traits>
#include "trace_flags.h"

namespace tvm {
namespace runtime {
inline trace::Flags &flags() {
  static trace::Flags flags;
  return flags;
}

inline trace::TraceFile &file() {
  static trace::TraceFile file(flags().process_tracing, flags().trace_log_file);
  return file;
}

static void InitializeFlags() {
  trace::FlagParser parser(&flags());
  parser.ParseString(trace::GetEnv("TVM_TRACE"));
}

static void FinishFile() { file().Close(); }

ATTRIBUTE_CONSTRUCTOR void Init() { InitializeFlags(); }
ATTRIBUTE_DESTRUCTOR void Finish() { FinishFile(); }

static const char *error_message =
    "The size of arguments for traced function can't be less then 2";

static bool Cached(DLTensor *tensor) {
  // Google code style rule for "Static and Global Variables", does not allow to
  // use object without trivial destructor with static storage duration and
  // suggest alternative - to use pointer to that object see:
  // google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
  static std::unordered_set<DLTensor *> *const tensor_cache_ptr =
      new std::unordered_set<DLTensor *>();
  if (tensor_cache_ptr) {
    if (tensor_cache_ptr->count(tensor)) {
      return true;
    } else {
      tensor_cache_ptr->insert(tensor);
      return false;
    }
  }
  return true;
}

template <typename T> static const char *GetOutputFormat() {
  if (std::is_same<T, int32_t>::value) {
    return "%d ";
  } else if (std::is_same<T, int64_t>::value) {
    return "%ld ";
  } else {
    return "%f ";
  }
}

template <typename T>
static void TVMTracePrintExpr(TVMArgs args, TVMRetValue *ret) {
  // The actual size depends on the shape of the processed Tensor.
  CHECK(args.size() > 2) << error_message;
  if (flags().process_tracing) {
    file().Write("%s", args[0].ptr<const char>());
    for (int i = 1; i < args.size() - 1; ++i) {
      // This is strange, but runtime part assumes that, the type of the indexes
      // depends on the type of the last argument. Didn't find any problem
      // in lowered pass and llvm codegen packed functions, so use the type
      // conversion operator to get the atual type.
      file().Write("[%lu]", static_cast<uint64_t>(args[i].operator T()));
    }
    file().WriteAssignSymbol();
    // We define the API of packed function where the traced value should come as
    // the last argument.
    file().Write(GetOutputFormat<T>(), args[args.size() - 1].operator T());
    file().WriteNewLine();
  }
  *ret = 0;
}

template <typename T>
static void TVMTracePrintBuffer(TVMArgs args, TVMRetValue *ret) {
  CHECK(args.size() > 2) << error_message;
  if (flags().process_tracing) {
    DLTensor *traced_tensor = args[args.size() - 1].operator DLTensor *();
    // First - check the cache, don't need to print the Tensor which we already
    // have printed.
    if (traced_tensor && !Cached(traced_tensor)) {
      file().Write("%s", args[0].ptr<const char>());
      uint64_t shape = 1;
      for (int i = 1; i < args.size() - 1; ++i) {
        uint64_t axis_len = args[i].operator uint64_t();
        file().Write("[%lu]", axis_len);
        shape *= axis_len;
      }
      file().WriteSpace();
      for (size_t i = 0; i < shape; ++i) {
        file().Write(GetOutputFormat<T>(),
                     static_cast<T *>(traced_tensor->data)[0]);
      }
      file().WriteNewLine();
    }
  }
  *ret = 0;
}

TVM_REGISTER_GLOBAL("f32_buffer_trace")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    TVMTracePrintBuffer<float>(args, ret);
    });

TVM_REGISTER_GLOBAL("f64_buffer_trace")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    TVMTracePrintBuffer<double>(args, ret);
    });

TVM_REGISTER_GLOBAL("i32_buffer_trace")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    TVMTracePrintBuffer<int32_t>(args, ret);
    });

TVM_REGISTER_GLOBAL("i64_buffer_trace")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    TVMTracePrintBuffer<int64_t>(args, ret);
    });

TVM_REGISTER_GLOBAL("f32_expr_trace")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    TVMTracePrintExpr<double>(args, ret);
    });

TVM_REGISTER_GLOBAL("f64_expr_trace")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    TVMTracePrintExpr<double>(args, ret);
    });

TVM_REGISTER_GLOBAL("i32_expr_trace")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    TVMTracePrintExpr<int64_t>(args, ret);
    });

TVM_REGISTER_GLOBAL("i64_expr_trace")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    TVMTracePrintExpr<int64_t>(args, ret);
    });

}  // namespace runtime
}  // namespace tvm
#endif
