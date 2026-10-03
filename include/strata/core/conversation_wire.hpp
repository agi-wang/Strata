#pragma once
#include "strata/core/conversation_cache.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::core {
// Versioned x86 little-endian snapshot. No pointers, allocator state or CUDA addresses.
// File size and every allocation are bounded; the checksum covers all state bytes.
class ConversationWire {
    FILE* f_;
    bool reading_;
    size_t left_;
    uint64_t hash_ = 14695981039346656037ull;
    void raw(void* p, size_t n) {
        if (n > left_) throw std::runtime_error("snapshot exceeds transfer limit or is truncated");
        left_ -= n;
        if ((reading_ ? fread(p, 1, n, f_) : fwrite(p, 1, n, f_)) != n)
            throw std::runtime_error("snapshot I/O failed");
        auto b = static_cast<unsigned char*>(p);
        for (size_t i = 0; i < n; ++i) { hash_ ^= b[i]; hash_ *= 1099511628211ull; }
    }
    template<class T> void scalar(T& x) { raw(&x, sizeof(x)); }
    size_t length(size_t n, size_t max, size_t width) {
        uint64_t count = n; scalar(count);
        if (count > max || count > left_ / width) throw std::runtime_error("invalid snapshot length");
        return static_cast<size_t>(count);
    }
    template<class T> void vector(std::vector<T>& v, size_t max) {
        const size_t n = length(v.size(), max, sizeof(T));
        if (reading_) v.resize(n);
        if (n) raw(v.data(), n * sizeof(T));
    }
    void buffer(ConversationBuffer& b) {
        const size_t n = length(b.size(), 1024ull * 1024 * 1024, 1);
        if (reading_) b.resize(n);
        if (!b.visit(0, n, [&](auto* p, size_t count, size_t) { raw(p, count); return true; }))
            throw std::runtime_error("invalid segmented buffer");
    }
    void checkpoint(ConversationCheckpoint& c) {
        if (!c.stage_parts.empty()) throw std::runtime_error("layer-split transfer unsupported");
        vector(c.ids, 32768);
        size_t n = length(c.imgs.size(), 32768, 16);
        if (reading_) c.imgs.resize(n);
        for (auto& i : c.imgs) { scalar(i.start); scalar(i.hash); }
        vector(c.gdn, 1024ull * 1024 * 1024); vector(c.ple, 1024ull * 1024 * 1024);
        vector(c.tails, 1024ull * 1024 * 1024); vector(c.dead, 1024ull * 1024 * 1024);
        vector(c.block_pos, 1024ull * 1024 * 1024); scalar(c.used);
    }
public:
    ConversationWire(FILE* f, bool reading, size_t limit) : f_(f), reading_(reading), left_(limit) {}
    void image(SavedConversation& s, const std::string& identity) {
        uint64_t magic = 0x31564b4154415253ull; // STRATAKV, schema 1
        scalar(magic);
        if (magic != 0x31564b4154415253ull) throw std::runtime_error("unsupported snapshot schema");
        std::vector<uint8_t> id(identity.begin(), identity.end()); vector(id, 256);
        if (std::string(id.begin(), id.end()) != identity || identity.empty())
            throw std::runtime_error("snapshot model identity mismatch");
        for (auto& v : s.geometry) scalar(v);
        scalar(s.layer_lo); scalar(s.layer_hi);
        uint8_t cvec = s.cvec; scalar(cvec);
        if (cvec > 1) throw std::runtime_error("invalid steering mode");
        s.cvec = cvec;
        checkpoint(s.live);
        size_t n = length(s.checkpoints.size(), 32, 8);
        if (reading_) s.checkpoints.resize(n);
        for (auto& c : s.checkpoints) checkpoint(c);
        n = length(s.kv.size(), 256, 8);
        if (reading_) s.kv.resize(n);
        for (auto& k : s.kv) {
            int32_t format = k.format; scalar(format); k.format = format;
            scalar(k.cells); scalar(k.heads); scalar(k.head_dim); scalar(k.page_size);
            scalar(k.pooled_rows); scalar(k.idx_dim);
            buffer(k.k); buffer(k.v); buffer(k.k_scale); buffer(k.v_scale); buffer(k.pooled);
        }
        const uint64_t expected = hash_;
        uint64_t checksum = expected; scalar(checksum);
        if (checksum != expected) throw std::runtime_error("snapshot checksum mismatch");
        if (reading_ && fgetc(f_) != EOF) throw std::runtime_error("snapshot trailing bytes");
    }
};
inline bool conversation_wire_file(const std::string& path, SavedConversation& image,
                                   const std::string& identity, bool reading, size_t limit, std::string& error) {
    FILE* f = fopen(path.c_str(), reading ? "rb" : "wb");
    if (!f) { error = "cannot open snapshot"; return false; }
    bool ok = false;
    if (reading) {
        if (fseek(f, 0, SEEK_END) != 0) { fclose(f); error = "snapshot seek failed"; return false; }
        const long size = ftell(f);
        if (size < 0 || static_cast<size_t>(size) > limit || fseek(f, 0, SEEK_SET) != 0) {
            fclose(f); error = "snapshot file exceeds limit"; return false;
        }
        limit = static_cast<size_t>(size);
    }
    try { ConversationWire(f, reading, limit).image(image, identity); ok = true; }
    catch (const std::exception& e) { error = e.what(); }
    if (fclose(f) != 0) { error = "snapshot close failed"; ok = false; }
    if (!ok && !reading) std::remove(path.c_str());
    return ok;
}
} // namespace strata::core
