#pragma once

#include "titan/core/types.h"
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>

namespace titan::cognition {

using namespace titan::core;

class ObjectCognitionEngine {
private:
    // 使用 list 方便频繁的删除操作 (Pruning)
    std::list<WorldEntity> entities_;
    int next_track_id_ = 1;

    // --- 配置参数 ---
    const double IOU_THRESHOLD = 0.3;         // IoU > 0.3 视为同一个物体
    const double TIME_TO_LIVE = 2.0;          // 2秒没看到就遗忘
    const double NEW_ENTITY_CONFIDENCE = 0.5; // 新物体置信度阈值

    // 上一次更新的时间，用于计算 dt (delta time)
    std::optional<TimePoint> last_update_time_;

public:
    ObjectCognitionEngine() = default;

    // --- 核心生命周期更新 ---
    void update(const std::vector<VisualDetection>& detections, TimePoint timestamp) {
        double dt = 0.033; // 默认 33ms
        if (last_update_time_.has_value()) {
            dt = std::chrono::duration<double>(timestamp - last_update_time_.value()).count();
            if (dt <= 0) dt = 0.001; // 防止除零
        }
        last_update_time_ = timestamp;

        // 1. [Predict] 预测阶段 (Kalman Filter 简化版)
        // 根据上一帧的速度，猜测这一帧物体在哪里
        for (auto& ent : entities_) {
            // x_new = x_old + v * t
            ent.position += ent.velocity * dt;
            
            // 2D 框也可以做简单预测 (假设大小不变，中心移动)
            // 这里为了简化，暂不预测 2D 框的移动，依赖 IoU 匹配
        }

        // 2. [Match] 匹配阶段 (Data Association)
        // 简单的贪婪匹配 (Greedy Match)，实际生产可用 Hungarian Algorithm
        std::vector<bool> is_det_matched(detections.size(), false);

        for (auto& ent : entities_) {
            int best_idx = -1;
            double best_iou = 0.0;

            for (size_t i = 0; i < detections.size(); ++i) {
                if (is_det_matched[i]) continue; // 已经被匹配过了
                
                // 类别必须一致 (或者相似)
                if (detections[i].label != ent.category) continue;

                double iou = calculateIoU(ent.last_box, detections[i].box);
                if (iou > IOU_THRESHOLD && iou > best_iou) {
                    best_iou = iou;
                    best_idx = i;
                }
            }

            if (best_idx != -1) {
                // -> 匹配成功：更新实体
                updateEntity(ent, detections[best_idx], dt);
                is_det_matched[best_idx] = true;
            } else {
                // -> 匹配失败：实体丢失 (Lost)
                // hit_streak 重置
                ent.hit_streak = 0;
            }
        }

        // 3. [Birth] 新生阶段
        // 未匹配的检测框 -> 变为新实体
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!is_det_matched[i] && detections[i].confidence > NEW_ENTITY_CONFIDENCE) {
                createEntity(detections[i]);
            }
        }

        // 4. [Death] 死亡/修剪阶段
        // 删除太久没看到的物体
        entities_.remove_if([timestamp, this](const WorldEntity& e) {
            double time_since_seen = std::chrono::duration<double>(timestamp - e.last_seen).count();
            
            // 规则 A: 存活很久的老物体，允许消失久一点 (Memory Persistence)
            if (e.age > 100 && time_since_seen < TIME_TO_LIVE * 2.0) return false;

            // 规则 B: 刚出生的物体，如果马上消失，立即删除 (Noise Filtering)
            if (e.age < 5 && time_since_seen > 0.5) return true;

            // 规则 C: 普通超时
            return time_since_seen > TIME_TO_LIVE;
        });
    }

    // --- 查询接口 ---

    // 获取所有实体的指针 (用于遍历)
    std::vector<WorldEntity*> getAllEntitiesPtrs() {
        std::vector<WorldEntity*> ptrs;
        for (auto& e : entities_) ptrs.push_back(&e);
        return ptrs;
    }

    // 根据 ID 获取
    WorldEntity* getEntity(int id) {
        for (auto& e : entities_) {
            if (e.track_id == id) return &e;
        }
        return nullptr;
    }

    // 根据类别搜索 (支持模糊匹配)
    std::vector<WorldEntity*> findByCategory(const std::string& category_keyword) {
        std::vector<WorldEntity*> results;
        for (auto& e : entities_) {
            if (e.category.find(category_keyword) != std::string::npos) {
                results.push_back(&e);
            }
        }
        return results;
    }

private:
    // --- 内部辅助逻辑 ---

    double calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
        cv::Rect inter = box1 & box2;
        cv::Rect union_rect = box1 | box2; // 注意：这是最小包围框，面积近似
        double area_inter = inter.area();
        double area_union = box1.area() + box2.area() - area_inter;
        if (area_union <= 0) return 0.0;
        return area_inter / area_union;
    }

    void createEntity(const VisualDetection& det) {
        WorldEntity ent;
        ent.track_id = next_track_id_++;
        ent.category = det.label;
        ent.last_seen = std::chrono::steady_clock::now(); // 暂存，实际由 update 传入的 timestamp 决定更好
        
        ent.position = det.position_3d;
        ent.velocity = Eigen::Vector3d::Zero();
        
        ent.last_box = det.box;
        if (!det.mask.empty()) ent.last_mask = det.mask.clone();
        
        ent.age = 1;
        ent.hit_streak = 1;

        // [认知] 注入先验知识
        injectCommonSense(ent);

        entities_.push_back(ent);
        // std::cout << "[Cognition] New Entity Created: ID " << ent.track_id << " (" << ent.category << ")" << std::endl;
    }

    void updateEntity(WorldEntity& ent, const VisualDetection& det, double dt) {
        ent.last_seen = std::chrono::steady_clock::now();
        ent.age++;
        ent.hit_streak++;

        // 1. 更新位置和速度 (一阶低通滤波)
        Eigen::Vector3d new_pos = det.position_3d;
        
        // v = (p_new - p_old) / dt
        Eigen::Vector3d measured_vel = (new_pos - ent.position) / dt;
        
        // 简单的滤波: vel = 0.7 * old_vel + 0.3 * measured_vel
        ent.velocity = ent.velocity * 0.7 + measured_vel * 0.3;
        
        // pos = 0.4 * predicted + 0.6 * measured (更相信观测)
        ent.position = ent.position * 0.4 + new_pos * 0.6;

        // 2. 更新视觉外观
        ent.last_box = det.box;
        if (!det.mask.empty()) ent.last_mask = det.mask.clone();

        // 3. 类别校正 (简单的多数投票逻辑可在此扩展)
    }

    // [知识图谱] 简单的常识注入
    // 在实际系统中，这部分应该查询图数据库或 LLM
    void injectCommonSense(WorldEntity& ent) {
        if (ent.category == "cup" || ent.category == "mug") {
            ent.knowledge_graph["graspable"] = {1.0, "true"};
            ent.knowledge_graph["material"] = {0.6, "ceramic"};
            ent.knowledge_graph["fragile"] = {0.8, "true"};
        } 
        else if (ent.category == "bottle") {
            ent.knowledge_graph["graspable"] = {1.0, "true"};
            ent.knowledge_graph["shape"] = {1.0, "cylinder"};
        }
        else if (ent.category == "person") {
            ent.knowledge_graph["graspable"] = {0.0, "false"}; // 不要抓人
            ent.knowledge_graph["is_agent"] = {1.0, "true"};   // 是有主动性的
        }
        else if (ent.category == "apple" || ent.category == "orange") {
            ent.knowledge_graph["edible"] = {1.0, "true"};
            ent.knowledge_graph["graspable"] = {1.0, "true"};
        }
    }
};

} // namespace titan::cognition