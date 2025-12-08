#pragma once
#include "titan/core/types.h"
#include <deque>
#include <vector>
#include <sstream>
#include "nlohmann_json/json.hpp"

namespace titan::memory {

using json = nlohmann::json;

class CognitiveStream {
private:
    std::deque<titan::core::CognitiveEvent> stream_;
    const size_t MAX_HISTORY = 100;
    
    // 用于状态去重，防止连续记录 "Vision is blurry"
    titan::core::FrameQuality last_visual_quality_ = titan::core::FrameQuality::VALID;
    titan::core::ComponentState last_arm_state_ = titan::core::ComponentState::READY;

public:
    // 通用添加接口
    void addEvent(titan::core::EventType type, const std::string& summary, const json& data = {}) {
        titan::core::CognitiveEvent evt;
        evt.timestamp = std::chrono::steady_clock::now();
        evt.type = type;
        evt.summary = summary;
        evt.detailed_data = data;
        
        stream_.push_back(evt);
        if (stream_.size() > MAX_HISTORY) stream_.pop_front();
    }

    // --- [新增] 感知数据融合接口 ---
    
    // 1. 注入视觉感知 (带去重和语义化)
    void addVisualContext(const titan::core::FusedContext& ctx) {
        if (!ctx.vision.has_value()) return;
        const auto& frame = ctx.vision.value();

        // A. 记录画质变化 (状态事件)
        if (frame.quality != last_visual_quality_) {
            std::string status_desc;
            if (frame.quality == titan::core::FrameQuality::BLURRY) status_desc = "Vision became BLURRY (Motion/Focus issue).";
            else if (frame.quality == titan::core::FrameQuality::DARK) status_desc = "Vision became DARK.";
            else if (frame.quality == titan::core::FrameQuality::VALID) status_desc = "Vision recovered to NORMAL.";
            
            addEvent(titan::core::EventType::PERCEPTION_BODY, status_desc, {{"quality", (int)frame.quality}});
            last_visual_quality_ = frame.quality;
        }

        // B. 记录关键检测 (内容事件)
        // 只有当画面清晰且有检测结果时才记录，避免刷屏
        if (frame.quality == titan::core::FrameQuality::VALID && !frame.detections.empty()) {
            json det_list = json::array();
            std::string summary = "Saw objects: ";
            for (const auto& det : frame.detections) {
                summary += det.label + ", ";
                det_list.push_back({{"label", det.label}, {"conf", det.confidence}});
            }
            // 可以在这里加一个时间阈值，比如每秒最多记录一次视觉内容
            addEvent(titan::core::EventType::PERCEPTION_VISUAL, summary, det_list);
        }
    }

    // 2. 注入自身状态 (错误/异常)
    void addSystemStatus(const titan::core::SystemStatus& status) {
        if (status.arm_state != last_arm_state_) {
            std::string desc = "Arm state changed to: ";
            if (status.arm_state == titan::core::ComponentState::STALLED) desc += "STALLED (Error)";
            else if (status.arm_state == titan::core::ComponentState::ACTIVE) desc += "ACTIVE";
            else desc += "IDLE";
            
            addEvent(titan::core::EventType::PERCEPTION_BODY, desc);
            last_arm_state_ = status.arm_state;
        }
    }

    // --- 上下文构建 ---
    std::string buildContextPrompt() {
        std::stringstream ss;
        ss << "### Recent Stream of Consciousness ###\n";
        for (const auto& evt : stream_) {
            // 时间戳转为相对时间 (e.g., T-5s)
            ss << evt.toString() << "\n";
        }
        return ss.str();
    }
    
    std::vector<titan::core::CognitiveEvent> getHistory() const {
        return {stream_.begin(), stream_.end()};
    }

    // 获取完整历史用于复盘学习
    std::vector<titan::core::CognitiveEvent> getEpisodeHistory() const {
        return {stream_.begin(), stream_.end()};
    }
    
    void clear() { stream_.clear(); }
};

} // namespace titan::memory