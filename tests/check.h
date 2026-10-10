// Dependency free test harness.
//
//   TEST(name) { ... }                          registers a test, name must be an identifier
//   CHECK(cond)                                 records a failure when cond is false
//   CHECK_CLOSE(got, want, n, atol, rtol)       compares n elements, pointers or containers
//
// Failures do not abort the test, so one run reports every broken check.
// The runner lives in tests/test_main.cpp.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace check {

struct Test {
  const char* name;
  void (*fn)();
};

inline std::vector<Test>& registry() {
  static std::vector<Test> tests;
  return tests;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

// Failure messages of the running test. The runner clears them before each test.
inline std::vector<std::string>& failures() {
  static std::vector<std::string> msgs;
  return msgs;
}

inline void fail(const char* file, int line, const std::string& msg) {
  failures().push_back(std::string(file) + ":" + std::to_string(line) + ": " + msg);
}

struct CloseResult {
  bool ok = true;
  size_t n_bad = 0;
  size_t first_bad = 0;  // valid when !ok
  double got_bad = 0, want_bad = 0;
  double max_abs = 0;
  double max_rel = 0;  // over elements with want != 0
};

// Element i passes when |got - want| <= atol + rtol * |want|, the numpy.allclose rule.
// NaN in either array never passes.
template <class G, class W>
CloseResult compare_close(const G* got, const W* want, size_t n, double atol, double rtol) {
  CloseResult r;
  for (size_t i = 0; i < n; ++i) {
    const double g = static_cast<double>(got[i]);
    const double w = static_cast<double>(want[i]);
    const double diff = std::fabs(g - w);
    if (diff > r.max_abs) r.max_abs = diff;
    if (w != 0 && diff / std::fabs(w) > r.max_rel) r.max_rel = diff / std::fabs(w);
    if (!(diff <= atol + rtol * std::fabs(w))) {
      if (r.ok) {
        r.ok = false;
        r.first_bad = i;
        r.got_bad = g;
        r.want_bad = w;
      }
      ++r.n_bad;
    }
  }
  return r;
}

// Lets CHECK_CLOSE take raw pointers, arrays or anything with .data().
template <class T>
const T* data_of(const T* p) { return p; }
template <class C>
auto data_of(const C& c) -> decltype(c.data()) { return c.data(); }

template <class G, class W>
void check_close(const G& got, const W& want, size_t n, double atol, double rtol,
                 const char* expr, const char* file, int line) {
  const CloseResult r = compare_close(data_of(got), data_of(want), n, atol, rtol);
  if (r.ok) return;
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "CHECK_CLOSE(%s) %zu of %zu bad, first at [%zu] got %.9g want %.9g, "
                "max abs %.3e, max rel %.3e (atol %.1e, rtol %.1e)",
                expr, r.n_bad, n, r.first_bad, r.got_bad, r.want_bad, r.max_abs, r.max_rel,
                atol, rtol);
  fail(file, line, buf);
}

}  // namespace check

#define TEST(name)                                              \
  static void name();                                           \
  static const ::check::Registrar name##_registrar(#name, name); \
  static void name()

#define CHECK(cond) \
  do {              \
    if (!(cond)) ::check::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define CHECK_CLOSE(got, want, n, atol, rtol) \
  ::check::check_close((got), (want), (n), (atol), (rtol), #got ", " #want, __FILE__, __LINE__)
