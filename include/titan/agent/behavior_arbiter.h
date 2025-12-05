#pragma once
#include <string>
#include <functional>
#include <vector>
#include <iostream>
#include <variant>
#include "titan/core/types.h"

namespace titan::agent {

// 行为提案
struct ActionProposal {
    std::string source;      // "Safety", "Task", "Exploration"
    double priority;         // 0.0 ~ 1.0+
    std::string description; // "Emergency Stop", "Grasp Bottle"
    
    // 具体执行的回调函数
    std::function<void()> execute; 
};

class BehaviorArbiter {
private:
    std::string last_winner_;
    double hysteresis_ = 0.1; // 滞后阈值，防止频繁切换

public:
    // 核心逻辑：赢者通吃 (Winner-Take-All)
    void arbitrate(std::vector<ActionProposal>& proposals) {
        if (proposals.empty()) return;

        // 1. 排序
        std::sort(proposals.begin(), proposals.end(), [](const auto& a, const auto& b) {
            return a.priority > b.priority;
        });

        const auto& winner = proposals[0];

        // 2. 稳定性检查 (Hysteresis)
        // 如果新赢家不是上一个赢家，且优先级优势不明显，则保持上一个 (避免抖动)
        if (winner.source != last_winner_) {
             // 查找上一轮赢家在本次的排位
             // (此处简化逻辑，直接执行最高分)
             std::cout << "[Arbiter] Switching Behavior: " << last_winner_ << " -> " << winner.source << std::endl;
        }

        // 3. 执行
        std::cout << "[Exec] [" << winner.source << "] (" << winner.priority << "): " << winner.description << std::endl;
        winner.execute();
        
        last_winner_ = winner.source;
    }
};

} // namespace titan::agent