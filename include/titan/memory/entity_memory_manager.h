#pragma once
#include "memory_types.h"
#include "titan/core/math_utils.h" // 假设有 cosine_similarity
#include <iostream>
#include <algorithm>

namespace titan::memory {

class EntityMemoryManager {
private:
    // 全局实体库: Entity ID -> Profile
    std::map<int, EntityProfile> entity_db_;

public:
    // --- 写入接口 ---
    
    // 感知层/认知层观察到事件后调用
    void recordObservation(int entity_id, const std::string& desc, const std::string& action) {
        EntityEvent evt;
        evt.timestamp = std::chrono::system_clock::now();
        evt.description = desc;
        evt.action_type = action;
        // evt.embedding = get_embedding_model(desc); // 调用 Embedding API
        
        entity_db_[entity_id].addEvent(evt);
        
        std::cout << "[Memory] Recorded for Entity " << entity_id << ": " << desc << std::endl;
    }

    // 更新属性 (Facts)
    void updateAttribute(int entity_id, const std::string& key, const std::string& value) {
        entity_db_[entity_id].attributes[key] = value;
    }

    // --- 提取接口 (Retrieval) ---

    // 1. 获取完整上下文 (用于 LLM 决策)
    json getEntityContext(int entity_id, int limit_events = 5) {
        if (entity_db_.find(entity_id) == entity_db_.end()) return {};

        const auto& profile = entity_db_[entity_id];
        json j;
        j["attributes"] = profile.attributes;
        j["states"] = profile.current_states;
        j["summary"] = profile.long_term_summary;
        
        // 获取最近的 N 个事件
        json events_j = json::array();
        int count = 0;
        for (auto it = profile.history.rbegin(); it != profile.history.rend(); ++it) {
            events_j.push_back(it->toJson());
            if (++count >= limit_events) break;
        }
        j["recent_history"] = events_j;
        return j;
    }

    // 2. 语义搜索 (RAG - Retrieval Augmented Generation)
    // "小明最近做过什么关于学习的事？" -> Query: "study learning"
    std::vector<EntityEvent> searchEvents(int entity_id, const std::vector<float>& query_vec) {
        std::vector<EntityEvent> results;
        if (entity_db_.find(entity_id) == entity_db_.end()) return results;

        const auto& hist = entity_db_[entity_id].history;
        
        // 简单的向量相似度搜索
        // for (const auto& evt : hist) {
        //     if (cosine_similarity(evt.embedding, query_vec) > 0.7) {
        //         results.push_back(evt);
        //     }
        // }
        return results;
    }
};

} // namespace titan::memory