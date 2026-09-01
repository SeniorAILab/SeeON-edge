#include "workspace_pool.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

struct Slot {
  int id = 0;
};

// Concurrent admissions must never issue the same workspace twice.
void test_try_acquire_is_exclusive() {
  constexpr int kPoolSize = 4;
  seeon::trt::BoundedPool<Slot> pool;
  std::vector<Slot> slots(kPoolSize);
  std::array<std::atomic<bool>, kPoolSize> in_use{};
  for (int index = 0; index < kPoolSize; ++index) {
    slots[static_cast<std::size_t>(index)].id = index;
    pool.add(&slots[static_cast<std::size_t>(index)]);
  }

  std::atomic<bool> double_owned{false};
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::array<std::thread, kPoolSize> threads;
  for (auto& thread : threads) {
    thread = std::thread{[&] {
      ready.fetch_add(1);
      while (!start.load()) std::this_thread::yield();
      Slot* slot = pool.try_acquire();
      if (slot == nullptr) return;
      const seeon::trt::PoolLease<Slot> lease{pool, *slot};
      auto& flag = in_use[static_cast<std::size_t>(slot->id)];
      if (flag.exchange(true)) double_owned = true;
      std::this_thread::sleep_for(std::chrono::microseconds{50});
      flag.store(false);
    }};
  }
  while (ready.load() != kPoolSize) std::this_thread::yield();
  start.store(true);
  for (auto& thread : threads) thread.join();

  check(!double_owned, "no workspace was ever owned by two threads at once");
  check(pool.available() == kPoolSize, "all acquired workspaces returned");
}

// Each registered workspace must support simultaneous ownership. This proves
// that the pool does not serialize successful non-blocking admissions.
void test_every_workspace_is_usable_concurrently() {
  constexpr int kPoolSize = 4;
  seeon::trt::BoundedPool<Slot> pool;
  std::vector<Slot> slots(kPoolSize);
  for (int index = 0; index < kPoolSize; ++index) {
    slots[static_cast<std::size_t>(index)].id = index;
    pool.add(&slots[static_cast<std::size_t>(index)]);
  }

  std::mutex mutex;
  std::condition_variable arrived;
  int waiting = 0;
  std::set<int> held_simultaneously;
  std::array<std::thread, kPoolSize> threads;
  for (auto& thread : threads) {
    thread = std::thread{[&] {
      Slot* slot = pool.try_acquire();
      if (slot == nullptr) return;
      const seeon::trt::PoolLease<Slot> lease{pool, *slot};
      std::unique_lock lock{mutex};
      held_simultaneously.insert(slot->id);
      if (++waiting == kPoolSize) {
        arrived.notify_all();
      } else {
        arrived.wait_for(lock, std::chrono::seconds{10},
                         [&] { return waiting == kPoolSize; });
      }
    }};
  }
  for (auto& thread : threads) thread.join();

  check(held_simultaneously.size() == static_cast<std::size_t>(kPoolSize),
        "all four workspaces were held at the same instant");
  check(pool.available() == kPoolSize, "all four returned after the barrier");
}

// Exhaustion is an immediate busy drop. A losing caller neither waits nor
// releases a workspace it did not acquire; releasing the winner enables reuse.
void test_capacity_one_drops_busy_and_reacquires_after_release() {
  seeon::trt::BoundedPool<Slot> pool;
  Slot only;
  pool.add(&only);

  std::array<Slot*, 2> acquired{};
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::array<std::thread, 2> callers;
  for (std::size_t index = 0; index < callers.size(); ++index) {
    callers[index] = std::thread{[&, index] {
      ready.fetch_add(1);
      while (!start.load()) std::this_thread::yield();
      acquired[index] = pool.try_acquire();
    }};
  }
  while (ready.load() != static_cast<int>(callers.size())) std::this_thread::yield();
  start.store(true);
  for (auto& caller : callers) caller.join();

  const int successful = (acquired[0] != nullptr) + (acquired[1] != nullptr);
  check(successful == 1, "exactly one contending caller acquired the workspace");
  check(pool.available() == 0, "busy drop did not release a workspace");
  Slot* winner = acquired[0] != nullptr ? acquired[0] : acquired[1];
  pool.release(winner);

  Slot* reacquired = pool.try_acquire();
  check(reacquired == &only, "release enabled immediate reacquisition");
  if (reacquired != nullptr) {
    const seeon::trt::PoolLease<Slot> lease{pool, *reacquired};
  }
  check(pool.available() == 1, "reacquired workspace returned through its lease");
}

// Early inference exits must return the workspace rather than leaking capacity.
void test_lease_returns_on_early_exit() {
  seeon::trt::BoundedPool<Slot> pool;
  std::vector<Slot> slots(2);
  for (auto& slot : slots) pool.add(&slot);

  const auto failing_inference = [&]() -> bool {
    Slot* slot = pool.try_acquire();
    if (slot == nullptr) return false;
    const seeon::trt::PoolLease<Slot> lease{pool, *slot};
    return false;  // engine_enqueue_failed, engine_output_copy_failed, ...
  };
  for (int index = 0; index < 100; ++index) {
    check(!failing_inference(), "failing inference reports failure");
  }
  check(pool.available() == 2, "no capacity leaked across 100 failures");
}

// Exception unwinding must also return the workspace.
void test_lease_returns_on_throw() {
  seeon::trt::BoundedPool<Slot> pool;
  Slot only;
  pool.add(&only);

  for (int index = 0; index < 10; ++index) {
    try {
      Slot* slot = pool.try_acquire();
      check(slot != nullptr, "workspace available before throwing inference");
      if (slot == nullptr) continue;
      const seeon::trt::PoolLease<Slot> lease{pool, *slot};
      throw std::runtime_error{"engine failure"};
    } catch (const std::runtime_error&) {
    }
  }
  check(pool.available() == 1, "workspace returned after unwinding");
}

// Every registered element must be reachable without consulting observability
// to decide admission.
void test_all_elements_are_reachable() {
  seeon::trt::BoundedPool<Slot> pool;
  std::vector<Slot> slots(4);
  for (int index = 0; index < 4; ++index) {
    slots[static_cast<std::size_t>(index)].id = index;
    pool.add(&slots[static_cast<std::size_t>(index)]);
  }

  std::vector<Slot*> held;
  held.reserve(4);
  for (int index = 0; index < 4; ++index) {
    Slot* slot = pool.try_acquire();
    check(slot != nullptr, "registered workspace was issued");
    if (slot != nullptr) held.push_back(slot);
  }
  std::set<int> seen;
  for (Slot* slot : held) seen.insert(slot->id);
  check(seen.size() == 4, "all four distinct workspaces were issued");
  check(pool.available() == 0, "pool is empty when all are held");
  for (Slot* slot : held) pool.release(slot);
  check(pool.available() == 4, "pool refills");
}

}  // namespace

int main() {
  test_try_acquire_is_exclusive();
  test_every_workspace_is_usable_concurrently();
  test_capacity_one_drops_busy_and_reacquires_after_release();
  test_lease_returns_on_early_exit();
  test_lease_returns_on_throw();
  test_all_elements_are_reachable();
  if (failures != 0) {
    std::fprintf(stderr, "%d workspace-pool check(s) failed\n", failures);
    return 1;
  }
  std::fprintf(stderr, "workspace-pool checks passed\n");
  return 0;
}
