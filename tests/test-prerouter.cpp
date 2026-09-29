#include "llama-prerouter.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
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
    if (argc < 2) {
        return 0;
    }

    auto router = llama_prerouter::load(argv[1], 40, 2048, 256, 4, error);
    if (!router || !router->has_owner(6) || !router->has_owner(38) || router->has_owner(5)) {
        std::fprintf(stderr, "checkpoint loading failed: %s\n", error.c_str());
        return 1;
    }

    std::vector<float> hidden(2048, 0.0f);
    const std::vector<int32_t> route = { 1, 3, 7, 11 };
    auto prediction = router->predict(6, hidden.data(), hidden.size(), route, {});
    if (prediction.size() != 4) {
        std::fprintf(stderr, "first-token prediction failed\n");
        return 1;
    }
    std::sort(prediction.begin(), prediction.end());
    if (std::unique(prediction.begin(), prediction.end()) != prediction.end() ||
            prediction.front() < 0 || prediction.back() >= 256) {
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
    return 0;
}
