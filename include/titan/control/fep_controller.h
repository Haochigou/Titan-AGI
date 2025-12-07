#pragma once
#include "titan/memory/sparse_gp_memory.h"
#include <Eigen/Dense>

namespace titan::control {

class FEPController {
public:
    struct ControlOutput {
        double force;
        bool is_exploring;
    };

    FEPController();
    ~FEPController();

    ControlOutput solve(const Eigen::VectorXd& perception_features);
    void learn(const Eigen::VectorXd& features, double actual_best, double pred_val);

private:
    titan::memory::SparseGPMemory muscle_memory_;
};

} // namespace titan::control