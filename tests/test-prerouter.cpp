#include "llama-prerouter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    std::string error;
    if (llama_prerouter::load("missing.safetensors", 40, 2048, 256, 8, error) ||
            error.find("K=4") == std::string::npos) {
        std::fprintf(stderr, "incompatible expert count was accepted\n");
        return 1;
    }
    error.clear();
    if (llama_prerouter::load("missing.safetensors", 40, 2048, 256, 4, error) ||
            error.find("cannot open") == std::string::npos) {
        std::fprintf(stderr, "missing checkpoint was accepted\n");
        return 1;
    }
    if (llama_prerouter::load_shared("missing.safetensors", 40, 2048, 256, 8, error) ||
            error.find("K=4") == std::string::npos) {
        std::fprintf(stderr, "shared loader accepted an incompatible expert count\n");
        return 1;
    }
    if (llama_prerouter::load_shared("missing.safetensors", 40, 2048, 256, 4, error) ||
            error.find("cannot open") == std::string::npos) {
        std::fprintf(stderr, "shared loader accepted a missing checkpoint\n");
        return 1;
    }
    if (argc < 2) {
        return 0;
    }

    std::vector<float> hidden(2048, 0.0f);
    const std::vector<int32_t> route = { 1, 3, 7, 11 };
    std::vector<int32_t> independent_prediction;
    {
        auto router = llama_prerouter::load(argv[1], 40, 2048, 256, 4, error);
        if (!router || !router->has_owner(6) || !router->has_owner(38) || router->has_owner(5)) {
            std::fprintf(stderr, "checkpoint loading failed: %s\n", error.c_str());
            return 1;
        }

        independent_prediction = router->predict(6, hidden.data(), hidden.size(), route, {});
        if (independent_prediction.size() != 4) {
            std::fprintf(stderr, "first-token prediction failed\n");
            return 1;
        }
        auto sorted_prediction = independent_prediction;
        std::sort(sorted_prediction.begin(), sorted_prediction.end());
        if (std::unique(sorted_prediction.begin(), sorted_prediction.end()) != sorted_prediction.end() ||
                sorted_prediction.front() < 0 || sorted_prediction.back() >= 256) {
            std::fprintf(stderr, "prediction contains invalid expert IDs\n");
            return 1;
        }

        const auto begin = std::chrono::steady_clock::now();
        for (int owner = 6; owner <= 38; ++owner) {
            if (router->predict(owner, hidden.data(), hidden.size(), route, route).size() != 4) {
                std::fprintf(stderr, "prediction failed for owner %d\n", owner);
                return 1;
            }
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - begin).count();
        std::printf("checkpoint loaded; 33 heads predicted in %lld ms\n", (long long) elapsed);
    }

    constexpr size_t n_threads = 8;
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::shared_ptr<const llama_prerouter>> concurrent_routers(n_threads);
    std::vector<std::string> concurrent_errors(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t i = 0; i < n_threads; ++i) {
        threads.emplace_back([&, i]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            concurrent_routers[i] = llama_prerouter::load_shared(
                    argv[1], 40, 2048, 256, 4, concurrent_errors[i]);
        });
    }
    while (ready.load(std::memory_order_acquire) != n_threads) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto & thread : threads) {
        thread.join();
    }
    for (size_t i = 0; i < n_threads; ++i) {
        if (!concurrent_routers[i] || concurrent_routers[i].get() != concurrent_routers[0].get()) {
            std::fprintf(stderr, "concurrent shared load %zu did not reuse the initial instance: %s\n",
                    i, concurrent_errors[i].c_str());
            return 1;
        }
    }

    auto repeated_router = llama_prerouter::load_shared(argv[1], 40, 2048, 256, 4, error);
    if (!repeated_router || repeated_router.get() != concurrent_routers[0].get()) {
        std::fprintf(stderr, "repeated shared load did not reuse the same checkpoint instance: %s\n",
                error.c_str());
        return 1;
    }
    if (concurrent_routers[0]->predict(6, hidden.data(), hidden.size(), route, {}) != independent_prediction) {
        std::fprintf(stderr, "shared prediction differs from independent load\n");
        return 1;
    }
    if (llama_prerouter::load_shared(argv[1], 40, 2048, 256, 8, error) ||
            error.find("K=4") == std::string::npos) {
        std::fprintf(stderr, "incompatible shared load reused a cached checkpoint\n");
        return 1;
    }
    auto after_invalid_router = llama_prerouter::load_shared(argv[1], 40, 2048, 256, 4, error);
    if (!after_invalid_router || after_invalid_router.get() != concurrent_routers[0].get()) {
        std::fprintf(stderr, "valid shared cache entry was not preserved after invalid counts\n");
        return 1;
    }

    std::weak_ptr<const llama_prerouter> lifetime = concurrent_routers[0];
    repeated_router.reset();
    after_invalid_router.reset();
    concurrent_routers.clear();
    if (!lifetime.expired()) {
        std::fprintf(stderr, "shared cache retained a checkpoint after concurrent consumers released it\n");
        return 1;
    }
    return 0;
}
