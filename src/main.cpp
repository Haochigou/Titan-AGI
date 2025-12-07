#include "titan/agent/titan_agent.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

using namespace titan::core;
using namespace titan::agent;
using namespace Eigen;

// TODO 增加多线程同步信号

int main() {
    std::cout << "=== Titan-AGI System Booting (Full Implementation) ===" << std::endl;
    TitanAgent robot;

    // 1. 模拟传感器数据流线程
    std::thread sensor_thread([&]() {
        auto t_start = std::chrono::steady_clock::now();
        while(true) {
            auto now = std::chrono::steady_clock::now();
            double t = std::chrono::duration<double>(now - t_start).count();

            // 1a. 模拟本体数据 (1kHz)
            RobotState rs;
            rs.timestamp = now;
            rs.joint_pos = VectorXd::Zero(6);
            rs.joint_vel = VectorXd::Constant(6, 0.5 * std::sin(t));
            rs.ee_pos = Vector3d(0.1, 0.5 + 0.1 * std::cos(t), 0.2); // 模拟运动
            rs.ee_rot = Quaterniond::Identity();
            
            // 1b. 模拟视觉数据 (30Hz)
            static auto last_cam = now;
            if (std::chrono::duration<double>(now - last_cam).count() > 0.033) {
                cv::Mat dummy_img = cv::Mat::zeros(480, 640, CV_8UC3);
                // 关键：模拟传输延迟 (曝光在 30ms 之前发生)
                robot.feedSensors(rs, dummy_img, now - std::chrono::milliseconds(30));
                last_cam = now;
            } else {
                robot.feedSensors(rs, cv::Mat(), now); // 只更新本体
            }

            // 1c. 模拟音频数据 (低频)
            robot.feedAudio(std::vector<int16_t>(512, 100));
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    sensor_thread.detach();

    // 2. 运行主控制循环 (100Hz)
    for(int i=0; i<300; ++i) {
        robot.tick(); 
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // 模拟用户指令
        if (i == 50) robot.onUserCommand("Pick up the red block");
        // 模拟中断
        if (i == 150) robot.onUserCommand("Stop it");
    }

    std::cout << "=== System Shutdown. Memories Saved. ===" << std::endl;
    
    return 0;
}