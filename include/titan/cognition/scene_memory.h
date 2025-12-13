#pragma once
#include "titan/core/types.h"
#include <vector>
#include <map>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace titan::cognition {

using namespace titan::core;

class SceneMemoryEngine {
private:
    std::vector<SceneNode> scenes_;
    int current_scene_id_ = -1;

    // 机器人自身的物理参数 (具身先验)
    const double ROBOT_WIDTH = 0.6; // 60cm 肩宽
    const double AVG_SPEED = 1.2;   // 1.2 m/s

public:
    // --- 1. 具身测量计算 ---
    // 根据深度图或激光雷达计算环境属性
    EnvironmentMetrics measureEnvironment(const cv::Mat& depth_map, const SystemStatus& status) {
        EnvironmentMetrics m;
        
        // A. 电池与续航估算
        m.battery_level = status.battery_voltage / 24.0; // 假设满电 24V
        m.avg_power_consumption = 50.0; // 假设 50W (实际应从 BMS 读取)
        
        // 估算时间 = 电池容量(Wh) * 剩余% / 功耗
        double total_capacity_wh = 500.0; 
        m.estimated_runtime_min = (total_capacity_wh * m.battery_level / m.avg_power_consumption) * 60.0;
        
        // 估算里程 = 时间 * 速度
        m.max_walkable_dist = (m.estimated_runtime_min * 60.0) * AVG_SPEED;

        // B. 空间宽度估算 (模拟)
        // 实际算法：提取深度图中间行的平均距离，转换为宽度
        double avg_depth = 3.0; // 假设前方 3米空旷
        m.estimated_width = 2.5; // 假设通道宽 2.5米 (Mock)
        
        // C. 通过性判定
        m.clearance_ratio = m.estimated_width / ROBOT_WIDTH;

        return m;
    }

    // --- 2. 场景识别与记忆加载 ---
    // 尝试识别当前场景，如果认识，返回 true；如果是新地方，创建新记忆
    bool recognizeOrMemorize(const cv::Mat& image, const EnvironmentMetrics& metrics, int& out_scene_id) {
        // A. 提取视觉特征 (Mock: 缩略图作为指纹)
        cv::Mat descriptor;
        cv::resize(image, descriptor, cv::Size(64, 64));
        cv::cvtColor(descriptor, descriptor, cv::COLOR_BGR2GRAY);

        // B. 搜索已有记忆 (简单的相似度匹配)
        double best_score = 0.0;
        int best_idx = -1;

        for (const auto& scene : scenes_) {
            // 简单的像素相关性 (实际应用请用 NetVLAD / ORB BoW)
            // double score = compare(scene.visual_descriptor, descriptor); 
            // 模拟：假设如果不为空就匹配不上
            if (scenes_.empty()) break; 
        }

        // C. 判定逻辑
        const double MATCH_THRESHOLD = 0.8;
        
        if (best_score > MATCH_THRESHOLD) {
            // -> 场景再认 (Relocalization)
            out_scene_id = scenes_[best_idx].id;
            // 更新该场景的最新状态
            scenes_[best_idx].metrics = metrics; 
            std::cout << "[SceneMemory] Welcome back to Scene " << out_scene_id << std::endl;
            return true; // 已知场景
        } else {
            // -> 新场景构建 (Mapping)
            SceneNode new_node;
            new_node.id = scenes_.size() + 1;
            new_node.visual_descriptor = descriptor.clone();
            new_node.metrics = metrics;
            new_node.created_at = std::chrono::steady_clock::now();
            new_node.semantic_label = "Unknown Area " + std::to_string(new_node.id);
            
            scenes_.push_back(new_node);
            out_scene_id = new_node.id;
            std::cout << "[SceneMemory] Explored new area: Scene " << out_scene_id << std::endl;
            return false; // 新场景
        }
    }
    
    // 加载场景关联的实体 ID
    std::vector<int> getEntitiesInScene(int scene_id) {
        for(const auto& s : scenes_) {
            if (s.id == scene_id) return s.anchor_entity_ids;
        }
        return {};
    }
};

} // namespace