#pragma once
#include "task_types.h"
#include "titan/memory/entity_memory_manager.h"
#include <future>
#include <sstream>
#include <iostream>
#include <nlohmann_json/json.hpp> // 推荐引入 JSON 库处理 LLM 输出

namespace titan::agent {

using json = nlohmann::json;

class StrategicPlanner {
public:
    struct OptimizationResult {
        std::vector<TaskContext> optimized_tasks;
        std::string reasoning; // LLM 的思考过程
    };
    void injectMemory(titan::memory::EntityMemoryManager* mem) { memory_manager_ = mem; }
    // --- 增强版规划接口 ---
    void planWithMemory(const std::string& user_goal, int target_entity_id) {
        
        // 1. 从记忆中提取上下文
        json entity_ctx = memory_manager_->getEntityContext(target_entity_id);
        
        // 2. 构建包含记忆的 Prompt
        std::string prompt = buildPrompt(user_goal, entity_ctx);
        
        // 3. 调用 LLM
        // ... call LLM ...
    }
private:
    std::future<OptimizationResult> planning_future_;
    bool is_optimizing_ = false;
    titan::memory::EntityMemoryManager* memory_manager_;
public:
    bool isBusy() const { return is_optimizing_; }

    // --- 核心：调用 LLM 进行多任务编排 ---
    void triggerOptimization(const std::vector<TaskContext>& active_tasks, const std::string& new_instruction = "") {
        std::cout << "[Strategy] Triggering LLM for Multi-Task Optimization..." << std::endl;
        is_optimizing_ = true;

        // 异步调用，防止阻塞
        planning_future_ = std::async(std::launch::async, [&]() {
            OptimizationResult result;
            
            // 1. 构建 Prompt
            std::string prompt = buildPrompt(active_tasks, new_instruction);
            
            // 2. 调用 LLM (Mock)
            // std::string llm_output = call_openai_or_deepseek(prompt);
            std::cout << "[Strategy] LLM Thinking: Analyzing " << active_tasks.size() << " tasks + new: " << new_instruction << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1)); // 模拟思考

            // 3. 解析 LLM 输出并重构任务列表
            // 这里模拟 LLM 发现了"去厨房倒水"和"去厨房拿苹果"可以合并
            // 或者是根据新指令调整了优先级
            
            // [模拟逻辑]: 如果有新指令，加入列表；如果是紧急指令，提高优先级
            result.optimized_tasks = active_tasks;
            
            if (!new_instruction.empty()) {
                TaskContext new_task;
                new_task.task_id = "task_" + std::to_string(std::rand());
                new_task.user_instruction = new_instruction;
                new_task.base_priority = PriorityLevel::NORMAL;
                
                // 简单的语义判断 (Mocking LLM logic)
                if (new_instruction.find("fire") != std::string::npos || new_instruction.find("stop") != std::string::npos) {
                    new_task.base_priority = PriorityLevel::CRITICAL;
                    result.reasoning = "Detected safety critical keyword.";
                } else if (new_instruction.find("kitchen") != std::string::npos) {
                    // 跨任务优化示例
                    result.reasoning = "Optimized: Merged kitchen activities.";
                    // 实际应该重新生成 steps，这里仅添加
                }
                
                // 拆解步骤 (should be done by LLM)
                new_task.steps.push_back({"s1", "Execute: " + new_instruction, "target", "act", "expectation"});
                
                result.optimized_tasks.push_back(new_task);
            }
            
            return result;
        });
    }

    // 检查是否有新的规划结果
    bool checkResult(std::vector<TaskContext>& out_tasks) {
        if (is_optimizing_ && planning_future_.valid()) {
            if (planning_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                auto res = planning_future_.get();
                out_tasks = res.optimized_tasks;
                std::cout << "[Strategy] Plan Updated. Reasoning: " << res.reasoning << std::endl;
                is_optimizing_ = false;
                return true;
            }
        }
        return false;
    }

private:
    std::string buildPrompt(const std::vector<TaskContext>& tasks, const std::string& new_cmd) {
        std::stringstream ss;
        ss << "You are a Robot Strategic Planner.\n";
        ss << "Current Active Tasks:\n";
        for(const auto& t : tasks) {
            ss << "- ID: " << t.task_id << " | Desc: " << t.user_instruction 
               << " | Priority: " << (int)t.base_priority << "\n";
        }
        if(!new_cmd.empty()) ss << "New Incoming Task: " << new_cmd << "\n";
        ss << "\nGOAL: \n";
        ss << "1. Assign priority (0-100) to all tasks.\n";
        ss << "2. Decompose new tasks into steps.\n";
        ss << "3. Optimize: If multiple tasks share a location or tool, interleave their steps to save time.\n";
        ss << "4. Output JSON.";
        return ss.str();
    }
    std::string buildPrompt(const std::string& goal, const json& memory) {
        std::stringstream ss;
        ss << "You are a helpful robot assistant.\n";
        ss << "User Goal: " << goal << "\n";
        
        ss << "Target Entity Profile:\n";
        ss << "- Attributes: " << memory["attributes"].dump() << "\n";
        ss << "- Recent Events: " << memory["recent_history"].dump() << "\n";
        
        ss << "Decision Logic:\n";
        ss << "Check the entity's recent history. If the user's goal conflicts with the entity's current state, propose a polite strategy.\n";
        
        return ss.str();
    }
};

} // namespace titan::agent