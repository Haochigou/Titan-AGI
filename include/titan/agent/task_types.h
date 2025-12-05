#pragma once
#include "titan/core/types.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace titan::agent {

enum class TaskStatus { PENDING, RUNNING, COMPLETED, FAILED, RETRYING };

// 任务优先级枚举
enum class PriorityLevel {
    BACKGROUND = 0, // 比如：闲时充电，扫描地图
    NORMAL = 50,    // 比如：倒咖啡
    URGENT = 80,    // 比如：有人敲门
    CRITICAL = 100  // 比如：跌倒检测，电池耗尽
};

// 感知预期 (Perceptual Expectation)
struct Expectation {
    // --- 视觉预期 ---
    bool has_visual = false;
    cv::Rect expected_roi;       // 预期的出现区域 (用于加速 Crop)
    std::string expected_label;  // 预期物体
    double expected_confidence;  // 预期的最低置信度 (低于此则视为异常)
    
    // --- 触觉/力觉预期 ---
    bool has_tactile = false;
    double expected_force;       // 预期接触力
    double force_tolerance;      // 容差范围

    // --- 自身状态预期 ---
    titan::core::RobotState expected_state;

    // --- TODO 声音预期 ---

    // --- TODO 文本预期 ---
    
    // --- 时间预期 ---
    double expected_duration;    // 预期耗时 (用于超时判断)

    // --- 构造函数 ---
    Expectation() = default;
    Expectation(const std::string text) {
        ParseFromString(text);
    }

    // --- TODO 从文本描述中进行解析，后续可能从LLM生成后解析
    bool ParseFromString(const std::string& text) {
        return false;
    }

    std::string ToString() {
        return "";
    }
};

// 单个子任务节点
struct SubTask {
    std::string id;
    std::string description;     // "Find the red cup"
    std::string target_object;   // "red cup" (用于 Top-down 注意力)
    std::string action_verb;     // "pick_up"
    
    TaskStatus status = TaskStatus::PENDING;
    int retry_count = 0;
    
    // [修复关键点 1]
    // 原来的 'const int MAX_RETRIES' 导致无法生成 operator=
    // 改为 static constexpr，让它成为类常量，不占用对象实例内存，也不阻碍赋值
    static constexpr int MAX_RETRIES = 3;
    
    bool is_verified = false; 

    // [修复关键点 2] 补充构造函数

    // 新增：任务携带的预期
    Expectation expectation;
    // 运行时数据：实际观测与预期的偏差 (用于学习)
    double prediction_error = 0.0;

    // 1. 默认构造函数 (用于 std::vector resize 等)
    SubTask() = default;

    // 2. 参数化构造函数 (方便快速创建)
    SubTask(std::string pid, std::string pdesc, std::string ptarget, std::string paction, std::string expectation)
        : id(std::move(pid)), 
          description(std::move(pdesc)), 
          target_object(std::move(ptarget)), 
          action_verb(std::move(paction)),
          status(TaskStatus::PENDING),
          retry_count(0),
          is_verified(false),
          expectation(expectation)
    {
    }
};

// 完整任务计划
struct TaskPlan {
    std::string global_goal;     // "Make coffee"
    std::vector<SubTask> steps;
    long unsigned int current_step_idx = 0;
    bool is_active = false;

    SubTask* getCurrentStep() {
        if (!is_active || current_step_idx >= steps.size()) return nullptr;
        return &steps[current_step_idx];
    }
    
    void advance() {
        if (current_step_idx < steps.size()) {
            steps[current_step_idx].status = TaskStatus::COMPLETED;
            current_step_idx++;
        }
        if (current_step_idx >= steps.size()) {
            is_active = false; // 计划完成
        }
    }
};

// 任务上下文
struct TaskContext {
    std::string task_id;
    std::string user_instruction; // 原始指令
    PriorityLevel base_priority;  // LLM 判定的基准优先级
    double dynamic_score;         // 运行时动态评分 (基准 + 距离 + 状态)
    
    // 任务依赖 (Dependency Graph)
    std::vector<std::string> depends_on_ids; 
    
    // 执行计划
    std::vector<SubTask> steps;
    long unsigned int current_step_idx = 0;
    TaskStatus status = TaskStatus::PENDING;

    // 资源锁 (Resource Locks)
    // 例如 {"arm_left", "camera"}，防止多任务冲突
    std::vector<std::string> required_resources; 

    bool isFinished() const { 
        return status == TaskStatus::COMPLETED || status == TaskStatus::FAILED; 
    }
    
    SubTask* getCurrentStep() {
        if (current_step_idx < steps.size()) return &steps[current_step_idx];
        return nullptr;
    }
};

} // namespace titan::agent