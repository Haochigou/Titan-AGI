#pragma once
#include "titan/core/types.h"
#include <thread>
#include <atomic>
#include <iostream>
#include <functional>

namespace titan::hal { // Hardware Abstraction Layer

using namespace titan::core;

// 1. 视觉驱动 (独立线程运行)
class CameraDriver {
public:
    using Callback = std::function<void(const cv::Mat&, TimePoint)>;

    CameraDriver(Callback cb) : callback_(cb), running_(true) {
        state_ = ComponentState::INITIALIZING;
        worker_ = std::thread(&CameraDriver::loop, this);
    }

    ~CameraDriver() {
        running_ = false;
        if (worker_.joinable()) worker_.join();
    }

    ComponentState getState() const { return state_; }

private:
    Callback callback_;
    std::thread worker_;
    std::atomic<bool> running_;
    std::atomic<ComponentState> state_;

    void loop() {
        // 模拟初始化耗时
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        state_ = ComponentState::READY;

        while (running_) {
            state_ = ComponentState::ACTIVE; // 正在采集
            auto now = std::chrono::steady_clock::now();
            
            // 模拟采集 (耗时 33ms = 30fps)
            cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3); 
            std::this_thread::sleep_for(std::chrono::milliseconds(33));

            // 模拟相机过热或掉线
            // if (rand() % 1000 == 0) state_ = ComponentState::ERROR;

            if (callback_) callback_(frame, now); // 回调给感知层
        }
        state_ = ComponentState::OFFLINE;
    }
};

// 2. 机械臂/本体驱动 (独立线程，硬实时 1kHz)
class RobotBodyDriver {
public:
    using Callback = std::function<void(const RobotState&)>;

    RobotBodyDriver(Callback cb) : callback_(cb), running_(true) {
        worker_ = std::thread(&RobotBodyDriver::loop, this);
    }
    
    ~RobotBodyDriver() {
        running_ = false;
        if(worker_.joinable()) worker_.join();
    }

    ComponentState getState() const { return state_; }
    
    // 接收控制指令 (非阻塞)
    void setCommand(const Eigen::VectorXd& torques) {
        // 写入底层寄存器...
        // 模拟: 如果力矩过大，设置 STALLED 状态
        if (torques.norm() > 50.0) state_ = ComponentState::STALLED;
    }

private:
    Callback callback_;
    std::thread worker_;
    std::atomic<bool> running_;
    std::atomic<ComponentState> state_{ComponentState::READY};

    void loop() {
        while (running_) {
            state_ = ComponentState::ACTIVE;
            auto now = std::chrono::steady_clock::now();

            RobotState rs;
            rs.timestamp = now;
            // ... 填充模拟数据 ...
            
            if (callback_) callback_(rs);

            // 1ms 周期
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

// 3. 麦克风驱动 (独立线程)
// ... 类似实现 ...

} // namespace titan::hal