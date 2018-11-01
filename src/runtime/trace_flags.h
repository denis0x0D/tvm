/*!
 *  Copyright (c) 2018 by Contributors
 * \file trace_flags.h
 * \brief Parse flags and so on.
 */
#ifndef TVM_RUNTIME_TRACE_FLAGS_H_
#define TVM_RUNTIME_TRACE_FLAGS_H_

#if defined(__linux__)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <new>

#if __has_attribute(constructor)
#define ATTRIBUTE_CONSTRUCTOR __attribute__((constructor))
#else
#define ATTRIBUTE_CONSTRUCTOR
#endif
#if __has_attribute(destructor)
#define ATTRIBUTE_DESTRUCTOR __attribute__((destructor))
#else
#define ATTRIBUTE_DESTRUCTOR
#endif

namespace trace {
struct Flags {
#define TRACE_FLAG(FlagType, FlagName, FlagDefaultValue, FlagDescription)      \
  FlagType FlagName;
#include "trace_flags.inc"
#undef TRACE_FLAG
  Flags() {
#define TRACE_FLAG(FlagType, FlagName, FlagDefaultValue, FlagDescrption)       \
  FlagName = FlagDefaultValue;
#include "trace_flags.inc"
#undef TRACE_FLAG
  }
  Flags(const Flags &other) = delete;
  Flags &operator=(const Flags &other) = delete;
  Flags(Flags &&other) = delete;
  Flags &operator=(Flags &&other) = delete;
};

// Don't use c++ standard allocators
struct Allocator {
  static void *Allocate(size_t size) {
    void *ptr = malloc(size);
    if (!ptr)
      Abort();
    return ptr;
  }
  static void Deallocate(void *ptr) { free(ptr); }
  static void Abort() { abort(); }
};

class TraceFile {
 public:
  TraceFile(bool process_tracing, const char *file_name)
      : file_to_write(stdout), can_write(true) {
    if (process_tracing && file_name) {
      file_to_write = Open(file_name);
      if (!file_to_write)
        can_write = false;
    }
  }
  TraceFile (const TraceFile &other) = delete;
  TraceFile &operator=(const TraceFile &other) = delete;
  TraceFile (TraceFile &&other) = delete;
  TraceFile &operator=(TraceFile &&other) = delete;

  inline FILE *Open(const char *file_name) { return fopen(file_name, "a"); }

  template <typename T>
  inline void Write(const char *format, T value) {
    if (can_write)
      fprintf(file_to_write, format, value);
  }

  inline void WriteNewLine () {
    if(can_write)
      fprintf(file_to_write, "\n");
  }

  inline void WriteSpace() {
    if (can_write)
      fprintf(file_to_write, " ");
  }

  inline void WriteAssignSymbol() {
    if (can_write)
      fprintf(file_to_write, "=");
  }

  inline void Close() {
    if (can_write)
      fclose(file_to_write);
  }

 private:
  FILE *file_to_write;
  bool can_write;
};

struct BaseHandler {
  virtual bool Handle(const char *flag) = 0;
  virtual ~BaseHandler() {}
};

template <typename T> struct FlagHandler : BaseHandler {
  explicit FlagHandler(T *flag_value) : flag_value(flag_value) {}
  FlagHandler(const FlagHandler &other) = delete;
  FlagHandler &operator=(const FlagHandler &other) = delete;
  FlagHandler &operator=(FlagHandler &&other) = delete;
  FlagHandler(FlagHandler &&other) = delete;
  T *flag_value;
  inline bool Handle(const char *flag);
};

template <> inline bool FlagHandler<bool>::Handle(const char *value) {
  if (!value)
    return false;
  size_t len = strlen(value);
  if (len == 1)
    *flag_value = value[0] == '1' ? true : false;
  else if (len == 4 || len == 5)
    *flag_value = strcmp(value, "true") == 0 ? true : false;
  return true;
}

template <> bool FlagHandler<const char *>::Handle(const char *value) {
  if (!value)
    return false;
  // We do it only once, while initializing, so the allocated space will be
  // freed at the end of the process
  *flag_value = strdup(value);
  return true;
}

struct FlagRef {
  FlagRef()
      : flag_name(nullptr), flag_description(nullptr), flag_handler(nullptr) {}
  const char *flag_name;
  const char *flag_description;
  BaseHandler *flag_handler;
};

class FlagParser {
 public:
  explicit FlagParser(Flags *flags) : flags_count(0) {
    // Register flags, so we can update them while reading env variable.
#define TRACE_FLAG(FlagType, FlagName, FlagDefaultValue, FlagDescription)      \
  RegisterFlagHandler(#FlagName, FlagDescription, &flags->FlagName);
#include "trace_flags.inc"
#undef TRACE_FLAG
  }
  FlagParser(const FlagParser &other) = delete;
  FlagParser &operator=(const FlagParser &other) = delete;
  FlagParser(FlagParser &&other) = delete;
  FlagParser &operator=(FlagParser &&other) = delete;

  template <typename T>
  void RegisterFlagHandler(const char *flag_name, const char *flag_description,
                           T *flag_value) {
    BaseHandler *handler =
        static_cast<BaseHandler *>(Allocator::Allocate(sizeof(FlagHandler<T>)));
    handler = new (handler) FlagHandler<T>(flag_value);
    flags_ref[flags_count].flag_handler = handler;
    flags_ref[flags_count].flag_description = flag_description;
    flags_ref[flags_count].flag_name = flag_name;
    ++flags_count;
  }

  bool CopyString(char *to, const char *from, size_t start, size_t end) {
    // Check that string has not zero len and
    // does not exceed kMaxFlagValueLen
    if (end <= start || (end - start) >= kMaxFlagValueLen)
      return false;
    memset(to, 0, kMaxFlagValueLen);
    memcpy(to, from + start, end - start);
    return true;
  }

  // Parsing is based on DFA, we have three states:
  // flag, value and error.
  bool ParseString(const char *str) {
    if (!str)
      return false;
    size_t str_len = strlen(str);
    size_t it = 0;
    if (str_len > kMaxStringLen)
      return false;

    State current_state = State::kFlag;
    size_t start = 0;

    while (it < str_len && current_state != State::kError) {
      switch (current_state) {
      case State::kFlag:
        if (str[it] == '=') {
          if (!CopyString(flag_name, str, start, it)) {
            HandleError();
            current_state = State::kError;
            break;
          }
          current_state = State::kValue;
          start = it + 1;
        } else if (!IsValid(str[it])) {
          current_state = State::kError;
        }
        break;
      case State::kValue:
        if (str[it] == ':') {
          if (!CopyString(flag_value, str, start, it)) {
            HandleError();
            current_state = State::kError;
            break;
          }
          // Populate the flag with specified value
          ParseFlag(flag_name, flag_value);
          current_state = State::kFlag;
          start = it + 1;
        } else if (!IsValid(str[it])) {
          current_state = State::kError;
        }
        break;
      default:
        break;
      }
      ++it;
    }
    if (current_state == State::kValue) {
      if (!CopyString(flag_value, str, start, it))
        return false;
      ParseFlag(flag_name, flag_value);
    }
    return current_state != State::kError;
  }

  bool ParseFlag(const char *flag_name, const char *value) {
    bool found = false;
    size_t it = 0;
    // We have the constant amount of flags, so the linear
    // search is ok in our case.
    while (it < flags_count) {
      if (CompareFlag(flag_name, flags_ref[it].flag_name)) {
        found = true;
        break;
      }
      ++it;
    }
    // Skip it if didn't find the flag
    if (found)
      flags_ref[it].flag_handler->Handle(value);
    return found;
  }

  inline bool CompareFlag(const char *flag1, const char *flag2) {
    return strcmp(flag1, flag2) == 0;
  }

  ~FlagParser() {
    for (size_t it = 0; it < flags_count; ++it) {
      flags_ref[it].flag_handler->~BaseHandler();
      Allocator::Deallocate(flags_ref[it].flag_handler);
    }
  }

  inline bool IsValid(char ch) {
    if ((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') ||
        (ch >= 'A' && ch <= 'Z') || ch == '_' || ch == '.' || ch == '/')
      return true;
    return false;
  }

  // Handle an error.
  inline void HandleError() {}

 private:
  static const size_t kMaxFlagsCount = 4;
  static const size_t kMaxStringLen = 5024;
  static const size_t kMaxFlagValueLen = 512;
  FlagRef flags_ref[kMaxStringLen];
  char flag_name[kMaxFlagValueLen];
  char flag_value[kMaxFlagValueLen];
  size_t flags_count;
  enum class State { kFlag = 0, kValue = 1, kError = 2 };
};

static const char *GetEnv(const char *env_variable) {
  return getenv (env_variable);
}
}  // namespace trace
#endif
#endif  // TVM_RUNTIME_TRACE_FLAGS_H_
