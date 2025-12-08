#pragma once
#include "titan/memory/cognitive_stream.h"
#include "nlohmann_json/json.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <cmath>

namespace titan::learning {

struct StrategyEntry {
    int id;
    std::string rule_text;      // "If user shouts, stop immediately."
    std::vector<std::string> tags; // ["safety", "audio", "urgent"]
    
    // 模拟 Embedding 向量
    std::vector<float> embedding; 
    
    int usage_count = 0;        // 使用频率
    double success_rate = 1.0;  // 成功率
};
using json = nlohmann::json;

class StrategyOptimizer {
private:
    std::vector<StrategyEntry> strategy_db_;
    int next_id_ = 1;

    // 模拟：计算两个文本的相似度 (实际应使用 BERT/CLIP/OpenAI Embedding)
    double calculateSimilarity(const std::string& query, const StrategyEntry& entry) {
        // 简化的关键词匹配模拟
        double score = 0.0;
        for (const auto& tag : entry.tags) {
            if (query.find(tag) != std::string::npos) score += 0.5;
        }
        return score;
    }

public:
    // --- 1. RAG 检索接口 ---
    // 根据当前的任务描述和最近的事件流，检索最相关的 K 条策略
    std::string retrieveRelevantStrategies(const std::string& task_desc, const std::string& recent_stream_summary) {
        if (strategy_db_.empty()) return "";

        std::string query_context = task_desc + " " + recent_stream_summary;
        
        // 简单的打分排序
        std::vector<std::pair<double, int>> scores;
        for (int i = 0; i < strategy_db_.size(); ++i) {
            double sim = calculateSimilarity(query_context, strategy_db_[i]);
            // 经常成功使用的策略加分
            double weight = 1.0 + (strategy_db_[i].usage_count * 0.1); 
            scores.push_back({sim * weight, i});
        }

        // Top-K 排序
        std::sort(scores.rbegin(), scores.rend());

        std::stringstream ss;
        ss << "### Relevant Strategies (Retrieved) ###\n";
        int k = std::min((int)strategy_db_.size(), 3); // 只取前 3 条
        for (int i = 0; i < k; ++i) {
            const auto& entry = strategy_db_[scores[i].second];
            if (scores[i].first > 0.1) { // 阈值过滤
                ss << "- " << entry.rule_text << "\n";
            }
        }
        return ss.str();
    }

    // --- 2. 学习与更新接口 ---
    // 在复盘时调用，LLM 返回建议，这里负责合并
    void updateStrategyLibrary(const std::string& llm_suggestion_json) {
        // 假设 LLM 返回的格式是:
        // {
        //    "action": "ADD" | "MODIFY" | "DELETE",
        //    "target_id": 12 (if MODIFY/DELETE),
        //    "new_rule": "...",
        //    "tags": ["..."]
        // }
        
        try {
            auto j = json::parse(llm_suggestion_json);
            std::string action = j["action"];

            if (action == "ADD") {
                StrategyEntry entry;
                entry.id = next_id_++;
                entry.rule_text = j["new_rule"];
                entry.tags = j["tags"].get<std::vector<std::string>>();
                // entry.embedding = compute_embedding(entry.rule_text); 
                strategy_db_.push_back(entry);
                std::cout << "[Strategy] Added new rule: " << entry.rule_text << std::endl;
            } 
            else if (action == "MODIFY") {
                int target_id = j["target_id"];
                for (auto& entry : strategy_db_) {
                    if (entry.id == target_id) {
                        std::cout << "[Strategy] Updated rule " << target_id << ": " << entry.rule_text << " -> " << j["new_rule"] << std::endl;
                        entry.rule_text = j["new_rule"];
                        entry.tags = j["tags"].get<std::vector<std::string>>();
                        break;
                    }
                }
            }
            else if (action == "DELETE") {
                // ... remove logic ...
            }
        } catch (...) {
            std::cerr << "[Strategy] Failed to parse LLM suggestion." << std::endl;
        }
    }

    // --- 3. 反思流程 (System 2) ---
    void reflectOnEpisode(const std::vector<titan::core::CognitiveEvent>& history, bool success) {
        // 1. 将历史转为文本
        std::stringstream log_ss;
        for(const auto& evt : history) log_ss << evt.toString() << "\n";

        // 2. 将当前已有的策略列表传给 LLM (带 ID，方便它引用)
        std::stringstream existing_rules_ss;
        for(const auto& s : strategy_db_) {
            existing_rules_ss << "ID " << s.id << ": " << s.rule_text << "\n";
        }

        // 3. 构建 Prompt
        std::stringstream prompt;
        prompt << "Analyze the interaction log below.\n";
        prompt << "Outcome: " << (success ? "SUCCESS" : "FAILURE") << "\n";
        prompt << "Log:\n" << log_ss.str() << "\n\n";
        prompt << "Existing Strategies:\n" << existing_rules_ss.str() << "\n";
        prompt << "Task: Do we need to ADD a new strategy, MODIFY an existing one, or do NOTHING?\n";
        prompt << "Output JSON format: { \"action\": \"ADD/MODIFY/NONE\", \"target_id\": <id>, \"new_rule\": \"...\", \"tags\": [...] }";

        // 4. 调用 LLM (Mock)
        // std::string response = call_llm(prompt.str());
        
        // 模拟 LLM 发现规则冲突并修改
        if (!success) {
            std::string mock_response = R"({
                "action": "ADD",
                "new_rule": "If vision is BLURRY, stop movement immediately before planning.",
                "tags": ["vision", "safety", "reflex"]
            })";
            updateStrategyLibrary(mock_response);
        }
    }
};

} // namespace titan::learning