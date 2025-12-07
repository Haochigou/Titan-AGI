#pragma once
#include "titan/core/types.h"
#include "hal/hardware_drivers.h"
#include <mutex>
#include <optional>

namespace titan::control {

class ActionManager {
public:
    enum class ActionStatus { IDLE, RUNNING, SUCCEEDED, FAILED };

private:
    titan::hal::RobotBodyDriver* driver_;
    
    struct CurrentAction {
        std::string name; // e.g., "Grasp"
        ActionStatus status;
        titan::core::TimePoint start_time;
    } current_act_;
    
    std::mutex mtx_;

public:
    explicit ActionManager(titan::hal::RobotBodyDriver* d) : driver_(d) {}

    // 发送指令
    void execute(const Eigen::VectorXd& cmd, const std::string& act_name) {
        std::lock_guard<std::mutex> lock(mtx_);
        driver_->setCommand(cmd);
        
        current_act_.name = act_name;
        current_act_.status = ActionStatus::RUNNING;
        current_act_.start_time = std::chrono::steady_clock::now();
    }

    // 查询当前行为状态 (供 Reasoning 使用)
    ActionStatus getStatus() {
        std::lock_guard<std::mutex> lock(mtx_);
        // 简单模拟：如果执行超过 2秒，认为成功
        auto now = std::chrono::steady_clock::now();
        if (current_act_.status == ActionStatus::RUNNING) {
             if (std::chrono::duration<double>(now - current_act_.start_time).count() > 2.0) {
                 current_act_.status = ActionStatus::SUCCEEDED;
             }
        }
        
        // 如果底层驱动报错 (如 STALLED)，则行为失败
        if (driver_->getState() == titan::core::ComponentState::STALLED) {
            current_act_.status = ActionStatus::FAILED;
        }

        return current_act_.status;
    }

    bool isBusy() {
        return getStatus() == ActionStatus::RUNNING;
    }
};

} // namespace titan::control