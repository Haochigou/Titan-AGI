#pragma once
#include "titan/memory/sparse_gp_memory.h"
#include <Eigen/Dense>

namespace titan::control {

class FEPController {
public:
    struct ControlOutput {
        double force;
        bool is_exploring;
        double velocity_limit;
    };

    FEPController();
    ~FEPController();

    ControlOutput solve(const Eigen::VectorXd& perception_features);
    void learn(const Eigen::VectorXd& features, double actual_best, double pred_val);

    // [新增] 外部干预接口
    void reduceGainForStability(); 
    
    // [新增] 每一帧调用，用于自动恢复增益
    void updateInternalState();

private:
    titan::memory::SparseGPMemory muscle_memory_;

    // 稳定性因子: 0.1 (极其保守) ~ 1.0 (全速运行)
    double stability_factor_ = 1.0;
    
    // 参数配置
    const double MIN_STABILITY = 0.2;
    const double RECOVERY_RATE = 0.01; // 每帧恢复 1%
};

} // namespace titan::control