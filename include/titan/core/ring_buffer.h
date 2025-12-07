#pragma once
#include "types.h"
#include <deque>
#include <mutex>
#include <optional>
#include <algorithm>

namespace titan::core {

template <typename T>
class RingTrack {
private:
    std::deque<T> buffer_;
    std::mutex mtx_;
    size_t capacity_;
    
public:
    explicit RingTrack(size_t cap) : capacity_(cap) {}

    void push(const T& item) {
        std::lock_guard<std::mutex> lock(mtx_);
        buffer_.push_back(item);
        if (buffer_.size() > capacity_) buffer_.pop_front();
    }

    std::pair<std::optional<T>, std::optional<T>> getBracket(TimePoint t_query) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (buffer_.empty()) return {std::nullopt, std::nullopt};

        auto it = std::lower_bound(buffer_.begin(), buffer_.end(), t_query, 
            [](const T& a, TimePoint t) { return a.timestamp < t; });

        if (it == buffer_.begin()) return {*it, *it};
        if (it == buffer_.end()) {
            // 需要外推
            return {buffer_.back(), std::nullopt};
        }

        return {*std::prev(it), *it};
    }

    std::vector<T> getRange(TimePoint start, TimePoint end) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::vector<T> res;
        for (const auto& item : buffer_) {
            if (item.timestamp >= start && item.timestamp <= end) res.push_back(item);
        }
        return res;
    }

    std::optional<T> getLatest() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (buffer_.empty()) {
            return std::nullopt;
        }
        return buffer_.back();
    }
    
    // ... 其他辅助函数 ...
};

} // namespace titan::core