#pragma once
#include "titan/core/types.h"
#include "task_types.h"
#include "strategic_planner.h"
#include "behavior_arbiter.h"
#include "titan/learning/strategy_optimizer.h"
#include "titan/memory/cognitive_stream.h"
#include "titan/cognition/object_cognition.h"
#include <algorithm>
#include <vector>

namespace titan::agent {

using namespace titan::core;

struct ActiveTask {
    std::string goal;
    titan::agent::TaskStatus status = titan::agent::TaskStatus::PENDING;
    std::string current_step;
    // ... 其他任务元数据
};

class MultiTaskExecutive {
private:
    std::vector<TaskContext> task_pool_;
    StrategicPlanner planner_;
    
    // 当前聚焦的任务 ID (Context)
    std::string current_focus_id_;
    titan::learning::StrategyOptimizer *strategy_optimizer_{nullptr};
    titan::memory::CognitiveStream *cognitive_stream_{nullptr};

    ActiveTask current_task_;
    std::future<std::string> llm_planning_result_; // 异步 LLM 规划结果
public:
    /**
     * @brief 注入策略优化器，用于在规划时检索经验策略。
     * @param optimizer StrategyOptimizer 实例的指针。
     */
    void injectStrategyOptimizer(titan::learning::StrategyOptimizer* optimizer) {
        strategy_optimizer_ = optimizer;
        std::cout << "[Executive] StrategyOptimizer injected successfully." << std::endl;
    }
    // =========================================================
    // C. 辅助函数实现
    // =========================================================

    /**
     * @brief 触发异步 LLM 规划请求。
     */
    void triggerPlanning(const std::string& reason) {
        if (!strategy_optimizer_ || !cognitive_stream_) {
            std::cerr << "[Executive] Cannot plan: Optimizer or Stream missing." << std::endl;
            return;
        }
        
        // 确保没有正在运行的规划
        if (llm_planning_result_.valid() && 
            llm_planning_result_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            // 避免重复触发，等待当前规划完成
            // std::cout << "[Executive] Planning already in progress. Ignoring trigger." << std::endl;
            return;
        }

        // 1. RAG 策略检索 (RAG)
        std::string recent_context = cognitive_stream_->buildContextPrompt();
        std::string strategies = strategy_optimizer_->retrieveRelevantStrategies(current_task_.goal, recent_context);
        
        // 2. 构建完整 Prompt
        std::string planning_prompt;
        planning_prompt += "TASK: " + current_task_.goal + "\n";
        planning_prompt += "REASON FOR PLAN: " + reason + "\n";
        planning_prompt += strategies;
        planning_prompt += recent_context;
        planning_prompt += "\nINSTRUCTION: Provide the next required action or step for the task.";

        // 3. 异步调用 LLM (关键：使用 std::async 或线程池)
        llm_planning_result_ = std::async(std::launch::async, 
            [planning_prompt]() -> std::string {
                // === [Mock LLM Call] ===
                std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 模拟 LLM 延迟
                if (planning_prompt.find("Anomaly") != std::string::npos) {
                    return "Abort and reset system. Inform user of failure.";
                }
                return "Move to object 'cup' and grasp it.";
                // ========================
            }
        );
        
        std::cout << "[Executive] Triggered new async planning. Reason: " << reason << std::endl;
    }

    // --- Mock 辅助函数 ---
    bool checkStepCompletion(const std::string& step, const titan::core::FusedContext& ctx, titan::cognition::ObjectCognitionEngine& cognition) {
        // 假设：如果当前步骤是"Move to cup"，且 WorldModel 显示机器人距离杯子小于 0.1m，则完成
        // 实际需要复杂的状态机和 FEP Controller 的反馈
        return false; // 总是返回 false，直到规划器给出一个完成状态
    }

    bool checkAnomaly(const titan::core::FusedContext& ctx, titan::cognition::ObjectCognitionEngine& cognition) {
        // 简单检查：如果视觉质量连续 1秒 BLURRY，则视为异常
        if (ctx.vision.has_value() && ctx.vision->quality == titan::core::FrameQuality::BLURRY) {
            // 实际需要状态计数器
            return true; 
        }
        return false;
    }
    /**
     * @brief 注入认知流，用于向 LLM 提供历史上下文。
     */
    void injectMemoryStream(titan::memory::CognitiveStream* stream) {
        cognitive_stream_ = stream;
    }
    // --- 1. 接收指令 ---
    void addInstruction(const std::string& text) {
        // 触发 System 2 思考，传入当前所有任务和新指令
        planner_.triggerOptimization(task_pool_, text);
    }

    // --- 2. 主循环 (每帧调用) ---
    /**
     * @brief 执行 Executive 的心跳更新。
     * * 此函数运行在主线程 (tick())，必须是非阻塞的。
     */
    void update(const FusedContext& ctx, titan::cognition::ObjectCognitionEngine& cognition) {
        if (current_task_.status == TaskStatus::PENDING) {
            // 如果有新指令，但没有规划，立即触发第一次规划
            if (!current_task_.goal.empty()) {
                triggerPlanning("Initial planning for new goal: " + current_task_.goal);
                current_task_.status = TaskStatus::ACTIVE;
            }
            return;
        }

        // =========================================================
        // A. 检查异步 LLM 规划结果 (Non-blocking Check)
        // =========================================================
        if (llm_planning_result_.valid() && 
            llm_planning_result_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            
            try {
                std::string new_plan = llm_planning_result_.get();
                
                // TODO: 解析 new_plan (可能是 CoT, 动作序列, 或状态更新)
                // 记录到认知流
                if (cognitive_stream_) {
                    cognitive_stream_->addEvent(EventType::THOUGHT_CHAIN, 
                        "LLM returned new plan: " + new_plan.substr(0, 50) + "...");
                }

                // [Executive 状态更新]：更新 current_task_.current_step 或动作队列
                // current_task_.action_queue = parseActions(new_plan);
                // current_task_.current_step = current_task_.action_queue.front(); // 简化
                current_task_.current_step = new_plan; // Mock: 整个计划作为当前步骤
                
            } catch (const std::exception& e) {
                std::cerr << "[Executive] Async planning failed: " << e.what() << std::endl;
                // 失败后需要重新规划或降级
                triggerPlanning("Planning failed. Need retry or simplified action.");
            }
        }

        // =========================================================
        // B. 任务推进逻辑 (Task Progression)
        // =========================================================
        
        if (current_task_.status == TaskStatus::ACTIVE) {
            // 1. 检查当前步骤是否完成 (System 1 Feedback)
            bool step_complete = checkStepCompletion(current_task_.current_step, ctx, cognition);

            if (step_complete) {
                // TODO: 推进到下一步
                // if (current_task_.action_queue.empty()) { ... } else { current_task_.action_queue.pop(); }
                
                if (current_task_.current_step.find("Completed") != std::string::npos) { // Mock Completion
                    current_task_.status = TaskStatus::COMPLETED;
                    std::cout << "[Executive] Task completed: " << current_task_.goal << std::endl;
                } else {
                    // 推进后触发新规划或执行下一个预计算的子步骤
                    triggerPlanning("Step '" + current_task_.current_step + "' completed. Need next step.");
                }
            } 
            
            // 2. 检查异常或阻碍 (Anomaly Check)
            // 例如：如果目标对象消失了，或者手臂堵转了
            if (checkAnomaly(ctx, cognition)) {
                triggerPlanning("Anomaly detected: World state violation. Need replanning.");
                current_task_.status = TaskStatus::FAILED; // 暂时挂起，等待新规划
            }
        }
    }
    // --- [新增] 实用函数：3D 坐标转字符串 ---
    /**
     * @brief 将 Eigen::Vector3d 格式化为精确到三位小数的字符串 (e.g., "1.234, 0.500, 0.000")
     */
    std::string vectorToString(const Eigen::Vector3d& vec) {
        std::stringstream ss;
        // 设置浮点数为固定格式，保留三位小数
        ss << std::fixed << std::setprecision(3) 
        << vec.x() << ", " << vec.y() << ", " << vec.z();
        return ss.str();
    }
    // --- 3. 生成当前时刻的行为提案 ---
    // --- [核心集成点]：提案生成与 RAG 策略检索 ---
    ActionProposal getBestProposal(const titan::core::FusedContext& ctx, titan::cognition::ObjectCognitionEngine& cognition) {
        ActionProposal proposal;
        
        // 如果没有活跃任务，返回空提案
        if (current_task_.status != TaskStatus::ACTIVE) {
            proposal.source = "Executive";
            proposal.description = "Idle, awaiting command.";
            proposal.priority = 1.0;
            return proposal;
        }

        // A. 构建 LLM 的上下文 (Context)
        std::string task_context = current_task_.goal + " | Current Step: " + current_task_.current_step;
        
        std::string recent_history;
        if (cognitive_stream_) {
            // 从流中获取历史，但只取最近几秒的高级事件
            recent_history = cognitive_stream_->buildContextPrompt();
        }
        
        // B. [RAG 检索]：从策略库中获取相关经验
        std::string relevant_strategies;
        if (strategy_optimizer_) {
            // 这是注入的关键目的：根据当前任务和历史，检索最相关的策略。
            // 
            relevant_strategies = strategy_optimizer_->retrieveRelevantStrategies(task_context, recent_history);
        }
        
        // C. 构建最终 Prompt
        std::string prompt_for_llm;
        prompt_for_llm += "### LEARNED STRATEGIES ###\n" + relevant_strategies;
        prompt_for_llm += "\n### COGNITIVE STREAM ###\n" + recent_history;
        prompt_for_llm += "\n### TASK & WORLD ###\n" + task_context;
        
        // D. LLM 规划 (通常是异步操作)
        // 假设这里调用了一个同步/异步接口来获取下一步 Action (Plan/Act Loop)
        // std::string next_action = llm_planner_.plan(prompt_for_llm);
        
        // [Mock 结果]
        auto location = getTopDownTargetLocation(cognition);
        std::string next_action{""};
        if (location.has_value()) {
            next_action = "MoveTo(" + vectorToString(location.value()) + ")";
        } 
        
        // E. 生成提案
        proposal.source = "Executive";
        proposal.description = "Executing: " + next_action;
        proposal.priority = 5.0; // 中高优先级
        proposal.execute = [this, next_action]() {
            // 实际执行逻辑：调用 ActionManager
            this->performExecutiveAction(next_action); 
        };
        
        return proposal;
    }
    /**
     * @brief 解析当前任务目标，查询世界模型，返回目标的 3D 坐标。
     * @param cognition ObjectCognitionEngine 的引用，用于查询 WorldEntity。
     * @return 目标的 3D 位置，如果目标未找到，则返回 std::nullopt。
     */
    std::optional<Eigen::Vector3d> getTopDownTargetLocation(
        titan::cognition::ObjectCognitionEngine& cognition) {
        // 1. 任务状态检查
        if (current_task_.status != TaskStatus::ACTIVE || current_task_.goal.empty()) {
            return std::nullopt;
        }

        std::string goal = current_task_.goal;
        std::transform(goal.begin(), goal.end(), goal.begin(), ::tolower); // 转换为小写，便于匹配

        // 2. [语义解析] 简单启发式：提取目标关键词 (Mock LLM Planning)
        std::string target_keyword;
        
        if (goal.find("cup") != std::string::npos || goal.find("mug") != std::string::npos) {
            target_keyword = "cup";
        } else if (goal.find("box") != std::string::npos || goal.find("container") != std::string::npos) {
            target_keyword = "box";
        } else if (goal.find("person") != std::string::npos || goal.find("user") != std::string::npos) {
            target_keyword = "person";
        }
        // ... 更多关键词规则 ...
        
        if (target_keyword.empty()) {
            std::cerr << "[Executive] Could not parse a valid target keyword from the goal." << std::endl;
            return std::nullopt;
        }

        // 3. [世界模型查询] 查找匹配的实体
        auto potential_targets = cognition.findByCategory(target_keyword);

        if (potential_targets.empty()) {
            // 如果 WorldModel 说没找到，返回空
            return std::nullopt;
        }

        // 4. [属性过滤与决策] 找出最佳匹配目标
        // 目标：找到最符合语义和最近的实体。

        titan::core::WorldEntity* best_entity = nullptr;
        double min_distance_sq = std::numeric_limits<double>::max();
        bool found_specific_attribute = false; // 用于处理"红色"、"最近的"等修饰词

        // 假设当前机器人位置为 (0, 0, 0)
        Eigen::Vector3d robot_pos = Eigen::Vector3d::Zero(); 

        for (auto* entity : potential_targets) {
            // A. 处理颜色属性 (例如 "red cup")
            if (goal.find("red") != std::string::npos) {
                auto it = entity->knowledge_graph.find("color");
                if (it != entity->knowledge_graph.end() && it->second.value == "red" && it->second.confidence > 0.7) {
                    best_entity = entity;
                    found_specific_attribute = true;
                    break; // 如果找到明确匹配，立即返回 (贪婪策略)
                }
            }
            
            // B. 如果没有特定属性要求，则选择最近的
            double dist_sq = (entity->position - robot_pos).squaredNorm();
            if (!found_specific_attribute && dist_sq < min_distance_sq) {
                min_distance_sq = dist_sq;
                best_entity = entity;
            }
        }

        // 5. 返回结果
        if (best_entity) {
            // 记录思维事件 (可选)
            if (cognitive_stream_) {
                cognitive_stream_->addEvent(titan::core::EventType::THOUGHT_CHAIN, 
                    "Resolved target: Entity ID " + std::to_string(best_entity->track_id) + " (" + best_entity->category + ")");
            }
            return best_entity->position;
        }

        return std::nullopt;
    }
    void performExecutiveAction(const std::string& action) {
        // 记录思维事件
        if (cognitive_stream_) {
            cognitive_stream_->addEvent(titan::core::EventType::THOUGHT_CHAIN, "Decided next step: " + action);
        }
        // ... 调用 ActionManager 执行 ...
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
/*
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
*/
// --- 动态评分逻辑 (Context Awareness) ---
/*
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
    */
   /*
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
*/
public:
    std::optional<ActiveTask> popFinishedTask() { return std::nullopt; } // Mock
    double getCurrentPredictionError() const { return 0.0; }
    bool hasActiveTask() const { return false; }
    void abortAll() {}
};

} // namespace titan::agent