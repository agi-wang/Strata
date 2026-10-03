#include "strata/core/conversation_wire.hpp"
#include <stdexcept>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace strata::core;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (false)
int main() {
    const std::string path = (std::filesystem::temp_directory_path() /
        ("strata-wire-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string();
    SavedConversation src;
    src.geometry[0] = 42; src.layer_hi = 3; src.cvec = false;
    src.live.ids = {1, 2, 3}; src.live.gdn = {4, 5}; src.live.imgs = {{1, 100}};
    src.checkpoints.push_back(src.live);
    src.kv.resize(1); src.kv[0].format = 1; src.kv[0].cells = 3;
    src.kv[0].k.resize(ConversationBuffer::segment_bytes + 17, 7);
    src.kv[0].v = {1, 8, 9};
    std::string err;
    const size_t cap = 32 * 1024 * 1024;
    CHECK(conversation_wire_file(path, src, "model-A", false, cap, err));
    SavedConversation dst;
    CHECK(conversation_wire_file(path, dst, "model-A", true, cap, err));
    CHECK(dst.geometry == src.geometry && dst.cvec == src.cvec);
    CHECK(dst.live.ids == src.live.ids && dst.live.imgs == src.live.imgs);
    CHECK(dst.checkpoints[0].gdn == src.live.gdn);
    CHECK(dst.kv[0].k == src.kv[0].k && dst.kv[0].v == src.kv[0].v);
    CHECK(!conversation_wire_file(path, dst, "model-B", true, cap, err));
    CHECK(!conversation_wire_file(path, dst, "model-A", true, 1024, err));
    { std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(-9, std::ios::end); char byte = 99; f.write(&byte, 1); }
    CHECK(!conversation_wire_file(path, dst, "model-A", true, cap, err));
    CHECK(conversation_wire_file(path, src, "model-A", false, cap, err));
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    CHECK(!conversation_wire_file(path, dst, "model-A", true, cap, err));
    std::filesystem::remove(path);
    std::cout << "wire roundtrip, segmented KV, model mismatch, size limit, corruption and truncation PASS\n";
}
