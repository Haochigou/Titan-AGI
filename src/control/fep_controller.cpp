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

FEPController::ControlOutput FEPController::solve(const Eigen::VectorXd& perception_features) {
    auto [mean, variance] = muscle_memory_.predict(perception_features);
    
    ControlOutput out;
    out.force = mean;
    out.is_exploring = false;

    if (variance > 0.5) {
        out.force += variance * 2.0; 
        out.is_exploring = true;
        std::cout << "[FEP] High uncertainty (" << variance << "). Applying safety margin." << std::endl;
    }
    return out;
}

void FEPController::learn(const Eigen::VectorXd& features, double actual_best, double pred_val) {
    double surprise = std::abs(actual_best - pred_val);
    muscle_memory_.learn(features, actual_best, surprise);
}

} // namespace titan::control