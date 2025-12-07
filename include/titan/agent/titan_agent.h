#pragma once
#include "titan/core/types.h" // 只需要基础类型
#include <string>
#include <vector>
#include <memory> // 必须包含，用于 std::unique_ptr

// 注意：这里不需要包含 fep_controller.h, perception_system.h 等
// 因为它们现在被隐藏在 Impl 内部了，这样改动内部逻辑时不需要重新编译依赖 TitanAgent 的外部代码。

namespace titan::agent {

// [关键点 1] 前置声明实现类
// 告诉编译器 "有一个类叫 TitanAgentImpl，具体长什么样你先别管，反正我只存它的指针"
class TitanAgentImpl;

class TitanAgent {    
public:
    TitanAgent();
    // [关键点 2] 析构函数必须在 .cpp 中实现
    // 因为在头文件中 TitanAgentImpl 是不完整的类型，无法在这里通过 default 删除它
    ~TitanAgent();

    // 核心接口保持不变
    void feedSensors(const titan::core::RobotState& rs, const cv::Mat& img, titan::core::TimePoint t_img);
    void feedAudio(const std::vector<int16_t>& pcm);
    
    void tick();
    void onUserCommand(const std::string& text);

private:
    // [关键点 3] 定义指向实现的指针
    // 推荐使用 std::unique_ptr 自动管理内存，防止内存泄漏
    std::unique_ptr<TitanAgentImpl> impl_;
};

} // namespace titan::agent