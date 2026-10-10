// Test runner. Usage: polysome_tests [substring]
// Runs every registered test (or those whose name contains the substring) in name order,
// prints one line per test and a total, and exits non zero on any failure.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>

#include "check.h"

// Sanity test for the harness itself.
TEST(harness_compare_close) {
  const float want[4] = {1.0f, -2.0f, 0.0f, 4.0f};
  const float good[4] = {1.0f, -2.00001f, 0.0f, 4.0002f};
  const float bad[4] = {1.0f, -2.0f, 0.1f, 4.0f};

  const check::CloseResult r = check::compare_close(bad, want, 4, 1e-5, 1e-4);
  CHECK(!r.ok);
  CHECK(r.n_bad == 1);
  CHECK(r.first_bad == 2);

  const float nan[1] = {std::nanf("")};
  CHECK(!check::compare_close(nan, want, 1, 1e-5, 1e-4).ok);

  CHECK_CLOSE(good, want, 4, 1e-5, 1e-4);
  CHECK_CLOSE(std::vector<float>(good, good + 4), want, 4, 1e-5, 1e-4);
}

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;

  std::vector<check::Test> tests = check::registry();
  std::sort(tests.begin(), tests.end(),
            [](const check::Test& a, const check::Test& b) { return std::strcmp(a.name, b.name) < 0; });

  int passed = 0, failed = 0;
  for (const check::Test& t : tests) {
    if (filter && !std::strstr(t.name, filter)) continue;

    check::failures().clear();
    const auto t0 = std::chrono::steady_clock::now();
    try {
      t.fn();
    } catch (const std::exception& e) {
      check::fail(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
    } catch (...) {
      check::fail(__FILE__, __LINE__, "uncaught unknown exception");
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const bool ok = check::failures().empty();
    std::printf("%s  %-48s %9.2f ms\n", ok ? "PASS" : "FAIL", t.name, ms);
    for (const std::string& msg : check::failures()) std::printf("      %s\n", msg.c_str());
    ok ? ++passed : ++failed;
  }

  std::printf("%d passed, %d failed\n", passed, failed);
  if (passed + failed == 0) {
    std::printf("no tests matched\n");
    return 1;
  }
  return failed == 0 ? 0 : 1;
}
