#pragma once
#include "titan/core/types.h"
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <nlohmann_json/json.hpp> // 用于序列化给 LLM

namespace titan::memory {

using json = nlohmann::json;
using TimePoint = std::chrono::system_clock::time_point;

// 1. 实体事件 (Episodic Event)
// 记录 "Who did What to Whom When"
struct EntityEvent {
    std::string event_id;
    TimePoint timestamp;
    
    std::string description; // 自然语言描述: "Xiao Ming wrote homework"
    std::string action_type; // 动作类型: "write", "move", "speak"
    
    // 语义嵌入 (用于 RAG 检索)
    std::vector<float> embedding; 
    
    // 关联的其他实体 ID (形成图谱连接)
    // e.g., 这里的 target_entity_id 可能是 "Homework_Book" 的 ID
    std::vector<int> related_entity_ids; 

    // 转换为 JSON 供 LLM 阅读
    json toJson() const {
        return {
            {"time", std::chrono::system_clock::to_time_t(timestamp)},
            {"desc", description},
            {"action", action_type}
        };
    }
};

// 2. 实体画像 (Semantic Profile)
// 存储属性和状态
struct EntityProfile {
    // 基础属性 (Facts)
    // key: "name", "role", "age", "location"
    std::map<std::string, std::string> attributes;
    
    // 状态标签 (Tags)
    // e.g., "busy", "tired", "focused"
    std::vector<std::string> current_states;

    // 3. 动态时间线 (The Timeline)
    // 存储该实体参与的历史事件
    std::vector<EntityEvent> history;
    
    // 记忆摘要 (Long-term Summary)
    // 当 history 过长时，由 LLM 压缩成一段话存这就好
    std::string long_term_summary; 

    // 添加事件
    void addEvent(const EntityEvent& evt) {
        history.push_back(evt);
        // 实际工程中这里需要 RingBuffer 机制或存入数据库
    }
};

} // namespace titan::memory