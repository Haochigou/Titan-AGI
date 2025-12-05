#pragma once
#include "task_types.h"
#include "behavior_arbiter.h" // 需要生成 ActionProposal
#include "titan/core/types.h"
#include <iostream>
#include <thread>
#include <future>

namespace titan::agent {

class ExecutiveSystem {
private:
    TaskPlan current_plan_;
    
    // 模拟 LLM 异步推理线程
    std::future<TaskPlan> planning_future_;
    bool is_planning_ = false;

public:
    // --- 1. LLM 规划接口 ---
    void requestPlanning(const std::string& user_goal) {
        std::cout << "[Executive] Requesting LLM Plan for: " << user_goal << std::endl;
        is_planning_ = true;
        
        // 启动异步线程调用 LLM，避免阻塞主循环
        planning_future_ = std::async(std::launch::async, [user_goal]() {
            // [Mock LLM Call] 实际应调用 OpenAI/DeepSeek API
            // 输入: System Prompt + Context + Goal
            // 输出: JSON -> Parse to TaskPlan
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 模拟延迟
            
            TaskPlan plan;
            plan.global_goal = user_goal;
            plan.is_active = true;
            
            // 模拟简单的拆解结果
            if (user_goal.find("coffee") != std::string::npos) {
                plan.steps.push_back({"1", "Find the mug", "mug", "find"});
                plan.steps.push_back({"2", "Grasp the mug", "mug", "grasp"});
                plan.steps.push_back({"3", "Move to machine", "coffee_machine", "move"});
            } else {
                // 通用单步任务
                plan.steps.push_back({"1", "Execute: " + user_goal, "", "general"});
            }
            return plan;
        });
    }

    // --- 2. 主循环更新 ---
    void update() {
        // 检查 LLM 是否返回了结果
        if (is_planning_ && planning_future_.valid()) {
            if (planning_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                current_plan_ = planning_future_.get();
                is_planning_ = false;
                std::cout << "[Executive] Plan generated with " << current_plan_.steps.size() << " steps." << std::endl;
            }
        }
    }

    // --- 3. 获取 Top-Down 注意力焦点 ---
    std::string getCurrentAttentionTarget() {
        auto* step = current_plan_.getCurrentStep();
        if (step && step->status != TaskStatus::COMPLETED) {
            return step->target_object; // "mug"
        }
        return "";
    }

    // --- 4. 生成行为提案 (参与竞争) ---
    // 这是核心：将 Long-term Plan 转化为当前的 Action Proposal
    ActionProposal getProposal(const titan::core::FusedContext& ctx) {
        ActionProposal p;
        p.source = "ExecutivePlan";
        p.priority = 0.0;

        if (is_planning_) {
            // 如果正在规划，生成一个低优先级的"等待"行为，或者是"思考"表现
            p.priority = 1.0;
            p.description = "Thinking/Planning...";
            p.execute = [](){ /* LED blink or Think sound */ };
            return p;
        }

        auto* step = current_plan_.getCurrentStep();
        if (!current_plan_.is_active || !step) return p;

        // 根据子任务状态生成提案
        p.priority = 10.0; // 计划任务通常具有高优先级
        p.description = "Step " + step->id + ": " + step->description;

        // [Execution Logic Injection]
        // 这里定义具体的执行闭包
        p.execute = [this, step, &ctx]() {
            if (step->status == TaskStatus::PENDING) {
                std::cout << "[Exec] Starting Subtask: " << step->description << std::endl;
                step->status = TaskStatus::RUNNING;
            }
            
            // 简单的完成条件检查 (Mock)
            // 实际工程中，这里会调用 FEPController 并检查物理反馈
            static int tick_counter = 0;
            tick_counter++;
            
            // 模拟任务执行过程
            // 如果是 "find"，我们要检查视觉是否看到了目标
            if (step->action_verb == "find") {
                if (ctx.vision.has_value()) {
                    for(auto& det : ctx.vision->detections) {
                        if (det.label == step->target_object) {
                            std::cout << "[Exec] Target found via Vision!" << std::endl;
                            step->is_verified = true;
                        }
                    }
                }
            } else {
                // 其他动作简单延时模拟成功
                if (tick_counter > 10) step->is_verified = true;
            }

            // 任务完成判定
            if (step->is_verified) {
                std::cout << "[Exec] Subtask Complete!" << std::endl;
                step->status = TaskStatus::COMPLETED;
                current_plan_.advance(); // 推进到下一步
                tick_counter = 0;
            }
        };

        return p;
    }

    // 外部反馈接口：如果 FEP 控制器失败，这里处理重试逻辑
    void reportFailure(const std::string& reason) {
        auto* step = current_plan_.getCurrentStep();
        if (step) {
            std::cout << "[Executive] Action Failed: " << reason << ". Retrying..." << std::endl;
            step->retry_count++;
            if (step->retry_count > step->MAX_RETRIES) {
                step->status = TaskStatus::FAILED;
                // 触发重规划 (Re-planning)
                requestPlanning("Recover from failure: " + current_plan_.global_goal);
            }
        }
    }
};

} // namespace titan::agent