#pragma once
#include "titan/core/types.h"
#include <vector>
#include <string>
#include <algorithm>
#include <map>

namespace titan::perception {

using namespace titan::core;

struct AttentionalObject {
    VisualFrame::Detection raw_det;
    double bottom_up_score;  // Saliency (Motion, Surprise, Contrast)
    double top_down_score;   // Task Relevance
    double total_saliency;   // Weighted Sum
};

class AttentionEngine {
private:
    double weight_bu_ = 0.3; // 自下而上权重 (默认较低，容易被忽略)
    double weight_td_ = 0.7; // 自上而下权重 (任务优先)
    
    // 抑制返回 (Inhibition of Return, IOR)
    // 刚看过的东西，短时间内降低显著性，防止死盯着一个点
    std::map<std::string, double> inhibition_map_; 

public:
    void setTaskWeights(double bu, double td) {
        weight_bu_ = bu;
        weight_td_ = td;
    }

    // 核心计算：融合注意力
    // task_desc: 当前任务描述 (e.g., "red bottle")
    // surprise_map: 从 FEP 获得的惊奇度 (ObjectId -> SurpriseValue)
    std::vector<AttentionalObject> computeSaliency(
        const std::vector<VisualFrame::Detection>& detections,
        const std::string& task_keyword,
        const std::map<std::string, double>& surprise_map
    ) {
        std::vector<AttentionalObject> result;
        
        for (const auto& det : detections) {
            AttentionalObject obj;
            obj.raw_det = det;

            // 1. Bottom-Up Calculation
            // 基础分 + 惊奇度 (Surprise) + 运动 (假设从 embedding 差分或光流获得)
            double surprise = 0.0;
            if (surprise_map.count(det.label)) surprise = surprise_map.at(det.label);
            obj.bottom_up_score = det.confidence + (surprise * 2.0); // 惊奇度加权很高

            // 2. Top-Down Calculation
            // 简单的语义匹配 (实际可用 Embedding Cosine Similarity)
            obj.top_down_score = 0.0;
            if (!task_keyword.empty() && det.label.find(task_keyword) != std::string::npos) {
                obj.top_down_score = 1.0;
            }

            // 3. Inhibition (IOR) Decay
            double inhibition = inhibition_map_[det.label];
            inhibition_map_[det.label] *= 0.9; // 每帧衰减

            // 4. Fusion
            obj.total_saliency = (weight_bu_ * obj.bottom_up_score) + 
                                 (weight_td_ * obj.top_down_score) - 
                                 inhibition;
            
            result.push_back(obj);
        }

        // 按显著性降序排列
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            return a.total_saliency > b.total_saliency;
        });

        return result;
    }

    // 当 Agent 决定注视某个物体后调用，增加抑制
    void inhibit(const std::string& label) {
        inhibition_map_[label] += 0.5;
    }
};

} // namespace titan::perception