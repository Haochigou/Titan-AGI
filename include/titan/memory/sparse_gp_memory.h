#pragma once
#include "titan/core/types.h"
#include <vector>

namespace titan::memory {

class SparseGPMemory {
public:
    SparseGPMemory(); // 构造函数
    
    std::pair<double, double> predict(const Eigen::VectorXd& x);
    void learn(const Eigen::VectorXd& x, double y, double surprise);
    void save(const std::string& path);
    void load(const std::string& path);

private:
    struct MemoryNode {
        Eigen::VectorXd features;
        double outcome;
        double score;
    };
    
    // 私有辅助函数
    double kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2);
    void recompute();
    void pruneRedundant(const Eigen::VectorXd& x_new);

    std::vector<MemoryNode> nodes_;
    Eigen::MatrixXd K_inv_;
    bool dirty_;
    const size_t MAX_NODES = 100;
    double len_scale_;
    double noise_var_;
    double signal_var_;
};

} // namespace titan::memory