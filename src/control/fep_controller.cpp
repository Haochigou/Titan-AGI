#include "titan/control/fep_controller.h"
#include <iostream>
#include <cmath>

namespace titan::control {

FEPController::FEPController() {
    muscle_memory_.load("muscle.bin");
}

FEPController::~FEPController() {
    muscle_memory_.save("muscle.bin");
}

// 1. 核心计算逻辑
FEPController::ControlOutput FEPController::solve(const Eigen::VectorXd& perception_features) {
    auto [mean, variance] = muscle_memory_.predict(perception_features);
    
    ControlOutput out;
    out.is_exploring = false;

    // 基础力计算
    double raw_force = mean;

    // FEP 探索逻辑 (基于不确定性)
    if (variance > 0.5) {
        raw_force += variance * 2.0;
        out.is_exploring = true;
    }

    // [关键实现] 应用稳定性因子
    // 如果 stability_factor_ 变小 (e.g. 0.3)，输出力会变柔和
    // 这相当于降低了 PID 控制器中的 Kp (比例增益)
    out.force = raw_force * stability_factor_;

    // 同时限制最大速度，防止过冲
    // 假设最大物理速度是 1.0 m/s
    out.velocity_limit = 1.0 * stability_factor_;

    return out;
}

void FEPController::learn(const Eigen::VectorXd& features, double actual_best, double pred_val) {
    double surprise = std::abs(actual_best - pred_val);
    muscle_memory_.learn(features, actual_best, surprise);
}

// 2. [快降] 当发现视觉模糊时调用
void FEPController::reduceGainForStability() {
    // 乘法衰减：每次调用降低 50%，直到下限
    // 这种非线性下降能对连续模糊做出极快反应
    stability_factor_ *= 0.5;
    
    if (stability_factor_ < MIN_STABILITY) {
        stability_factor_ = MIN_STABILITY;
    }
    
    // std::cout << "[Control] Stability compromised. Dropping gain to " << stability_factor_ << std::endl;
}

// 3. [慢升] 每帧自动调用
void FEPController::updateInternalState() {
    // 线性恢复：如果没有外部干扰，慢慢回到 1.0
    if (stability_factor_ < 1.0) {
        stability_factor_ += RECOVERY_RATE;
        if (stability_factor_ > 1.0) stability_factor_ = 1.0;
    }
}

} // namespace titan::control