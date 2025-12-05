#pragma once
#include <chrono>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace titan::core {

using TimePoint = std::chrono::steady_clock::time_point;
using namespace Eigen;

// 高频本体状态
struct RobotState {
    TimePoint timestamp;
    VectorXd joint_pos; // 关节位置
    VectorXd joint_vel;
    Vector3d ee_pos; // 地图位置
    Quaterniond ee_rot; // 姿态朝向
    VectorXd imu_acc;
    // 自身运动状态 (IMU / 里程计)
    float velocity;
    float head_yaw;   // 头部水平角度
    float head_pitch; // 头部俯仰角度
};

// 视觉帧
struct VisualFrame {
    TimePoint timestamp;
    cv::Mat image;
    struct Detection {
        std::string label;
        float confidence;
        cv::Rect box;
        VectorXd embedding;
    };
    std::vector<Detection> detections;
    std::string vlm_desc;
};

// [新增] ASR 转录结果
struct AudioTranscript {
    TimePoint timestamp;       // 语音结束的时间点
    std::string text;          // 转录文本
    std::string speaker_id;    // (可选) 说话人ID，这里需要使用声音特征标识
    double confidence;         // 置信度
    bool processed = false;    // 标记是否已被 Agent 消费
};

// 原始音频
struct AudioChunk {
    TimePoint timestamp;
    std::vector<int16_t> pcm_data;
    int sample_rate;
};

struct Action {
    TimePoint start_timestamp;  // 行为开始时间戳
    std::string command;        // 行为命令
    std::string parameters;     // 行为参数
    TimePoint end_timestamp;    // 行为开始时间戳
    std::string report;         // 行为报告，包含执行反馈
};

struct FusedContext {
    TimePoint timestamp;
    RobotState robot;
    std::optional<VisualFrame> vision;
    std::vector<int16_t> audio_window;
    std::optional<AudioTranscript> latest_transcript;
    std::string attention;  // 来自上层的注意力
};
} // namespace titan::core