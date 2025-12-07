#pragma once
#include "task_types.h"
#include "strategic_planner.h"
#include "behavior_arbiter.h"
#include <algorithm>
#include <vector>

namespace titan::agent {

class MultiTaskExecutive {
private:
    std::vector<TaskContext> task_pool_;
    StrategicPlanner planner_;
    
    // 当前聚焦的任务 ID (Context)
    std::string current_focus_id_;

public:
    // --- 1. 接收指令 ---
    void addInstruction(const std::string& text) {
        // 触发 System 2 思考，传入当前所有任务和新指令
        planner_.triggerOptimization(task_pool_, text);
    }

    // --- 2. 主循环 (每帧调用) ---
    void update(const titan::core::FusedContext& sensor_ctx) {
        // A. 检查 System 2 是否完成了新的战略规划
        std::vector<TaskContext> new_plan;
        if (planner_.checkResult(new_plan)) {
            task_pool_ = new_plan; // 原子替换整个任务池
        }

        // B. 清理已完成的任务
        task_pool_.erase(std::remove_if(task_pool_.begin(), task_pool_.end(), 
            [](const TaskContext& t){ return t.isFinished(); }), task_pool_.end());

        // C. 动态优先级评估 (Micro-Scheduling)
        updateDynamicScores(sensor_ctx);
    }

    // --- 3. 生成当前时刻的行为提案 ---
    ActionProposal getBestProposal(const titan::core::FusedContext& ctx) {
        ActionProposal p;
        p.source = "MultiTaskExecutive";
        p.priority = 0.0;

        if (task_pool_.empty()) return p;

        // 1. 选出最佳任务 (Soft Context Switch)
        auto best_it = std::max_element(task_pool_.begin(), task_pool_.end(), 
            [](const TaskContext& a, const TaskContext& b){
                return a.dynamic_score < b.dynamic_score;
            });
        
        TaskContext& best_task = *best_it;
        SubTask* step = best_task.getCurrentStep();
        
        if (!step) return p;

        // 2. [新增] 预期注入 (Predictive Coding Injection)
        // 如果当前步骤还没生成过预期，生成它
        if (step->status == TaskStatus::PENDING && !step->expectation.has_visual) {
            generateExpectationForStep(*step, ctx);
        }

        // 3. 生成提案
        p.priority = best_task.dynamic_score;
        p.description = "[" + best_task.user_instruction + "] " + step->description;

        // 4. [新增] 绑定带预测验证的执行逻辑
        p.execute = [this, &best_task, step, &ctx]() {
            executeStepWithPrediction(best_task, step, ctx);
        };

        return p;
    }

    // 获取当前最高优先级任务的视觉目标 (给 AttentionEngine 用)
    std::string getTopDownTarget() {
        if (task_pool_.empty()) return "";
        // 简单策略：返回最高分任务的目标
        // 进阶策略：返回所有高分任务目标的并集
        auto best_it = std::max_element(task_pool_.begin(), task_pool_.end(), 
            [](const auto& a, const auto& b){ return a.dynamic_score < b.dynamic_score; });
        SubTask* step = best_it->getCurrentStep();
        return step ? step->target_object : "";
    }

private:
// --- [新增] 生成预期逻辑 ---
    void generateExpectationForStep(SubTask& step, const titan::core::FusedContext& ctx) {
        // 简单逻辑：基于动作类型生成
        if (step.action_verb == "find" || step.action_verb == "grasp") {
            step.expectation.has_visual = true;
            step.expectation.expected_label = step.target_object;
            
            // 假设：如果我在找东西，且之前记得它在桌子上
            // 这里可以接入 Semantic Map (SLAM) 获取历史位置
            // Mock: 假设它应该出现在视野中心附近
            step.expectation.expected_roi = cv::Rect(200, 150, 240, 180); 
        }
        
        if (step.action_verb == "grasp") {
            step.expectation.has_tactile = true;
            // 假设：根据历史经验，抓这个物体通常需要 5N
            step.expectation.expected_force = 5.0;
            step.expectation.force_tolerance = 2.0;
        }
    }

    // --- [新增] 带预测验证的执行逻辑 ---
    void executeStepWithPrediction(TaskContext& task, SubTask* step, const titan::core::FusedContext& ctx) {
        if (task.status == TaskStatus::PENDING) task.status = TaskStatus::RUNNING;

        bool verified = false;
        double surprise = 0.0;

        // A. 视觉预期验证 (Visual Verification)
        if (step->expectation.has_visual && ctx.vision.has_value()) {
            bool found_in_roi = false;
            for (const auto& det : ctx.vision->detections) {
                if (det.label == step->expectation.expected_label) {
                    // 计算 IoU 或包含关系
                    if ((det.box & step->expectation.expected_roi).area() > 0) {
                        found_in_roi = true;
                        break;
                    }
                }
            }
            
            if (found_in_roi) {
                // 符合预期 -> 加速确认
                verified = true;
            } else {
                // 不符合预期 -> 产生惊奇 -> 触发全图搜索或报错
                surprise += 0.5;
                // Fallback: 尝试在全图找
                // verified = fullScan(...);
            }
        }

        // B. 力觉预期验证 (Tactile Verification)
        if (step->expectation.has_tactile) {
            // 假设当前力
            double current_force = 0.0; // ctx.robot.force_sensor...
            double error = std::abs(current_force - step->expectation.expected_force);
            
            if (error > step->expectation.force_tolerance) {
                surprise += 1.0;
                std::cout << "[Executive] Unexpected Force! Error: " << error << std::endl;
                // 这就是"作为调整依据"的地方：
                // 可能触发 FEP 控制器的参数调整，或者直接任务失败
            }
        }

        // C. 更新误差记录 (供 Attention 模块使用)
        step->prediction_error = surprise;

        // D. 模拟任务推进
        // 在真实系统中，verified 为 true 才会推进
        static int timer = 0; 
        if (++timer > 5 || verified) { 
            step->status = TaskStatus::COMPLETED;
            task.current_step_idx++;
            timer = 0;
            if (task.current_step_idx >= task.steps.size()) {
                task.status = TaskStatus::COMPLETED;
            }
        }
    }
    // --- 动态评分逻辑 (Context Awareness) ---
    void updateDynamicScores(const titan::core::FusedContext& ctx) {
        for (auto& task : task_pool_) {
            if (task.isFinished()) { task.dynamic_score = -1.0; continue; }

            // 1. 基础分 (由 LLM 决定)
            double score = (double)task.base_priority;

            // 2. 距离惩罚 (Distance Cost)
            // 如果任务步骤是"去厨房"，但我已经在厨房了，分数+
            // 如果我在阳台，分数-
            // (此处需要结合地图系统，简化模拟)
            
            // 3. 资源约束
            // 如果任务需要"手臂"，但手臂正在忙别的，分数大幅降低
            
            // 4. 饿死提升 (Starvation Boost)
            // 随着等待时间增加，分数微量增加，防止低优先级任务永远不执行
            
            // 5. 状态机逻辑
            if (task.status == TaskStatus::RUNNING) {
                score += 5.0; // 惯性：倾向于把正在做的事做完
            }

            task.dynamic_score = score;
        }
    }

    void executeStepLogic(TaskContext& task, SubTask* step, const titan::core::FusedContext& ctx) {
        // ... (同之前的执行逻辑，检查完成条件，推进 step_idx) ...
        // 只是现在操作的是 task 对象
        if (task.status == TaskStatus::PENDING) task.status = TaskStatus::RUNNING;
        
        // 模拟完成
        static int timer = 0; timer++;
        if (timer > 5) {
            step->status = TaskStatus::COMPLETED;
            task.current_step_idx++;
            timer = 0;
            if (task.current_step_idx >= task.steps.size()) {
                task.status = TaskStatus::COMPLETED;
                std::cout << "[Executive] Task Finished: " << task.user_instruction << std::endl;
            }
        }
    }
};

} // namespace titan::agent