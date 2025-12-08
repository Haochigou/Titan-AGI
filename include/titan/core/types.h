#pragma once
#include <chrono>
#include <vector>
#include <string>
#include <atomic>
#include <optional>
#include <variant>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include "nlohmann_json/json.hpp"

namespace titan::core {

using json = nlohmann::json;
using TimePoint = std::chrono::steady_clock::time_point;
using namespace Eigen;


enum class EventType {
    // 输入 (Input)
    PERCEPTION_VISUAL,   // 看到了什么
    PERCEPTION_AUDIO,    // 听到了什么 (User Command)
    PERCEPTION_BODY,     // 感觉到了什么 (Status/Error)
    
    // 内部过程 (Internal Process)
    THOUGHT_CHAIN,       // CoT 推理过程
    DECISION_SWITCH,     // 任务切换/仲裁结果
    
    // 输出 (Output / Behavior)
    ACTION_PHYSICAL,     // 机械臂动作
    ACTION_VERBAL        // TTS 语音表达
};

// 认知事件单元
struct CognitiveEvent {
    TimePoint timestamp;
    EventType type;
    
    std::string summary;   // 自然语言描述 (用于 Prompt)
    json detailed_data;    // 结构化数据 (用于回溯分析)
    
    // 序列化为字符串，用于构建 LLM Context
    std::string toString() const {
        std::string prefix;
        switch(type) {
            case EventType::PERCEPTION_VISUAL: prefix = "[Eye]"; break;
            case EventType::PERCEPTION_AUDIO:  prefix = "[Ear]"; break;
            case EventType::THOUGHT_CHAIN:     prefix = "[Think]"; break;
            case EventType::DECISION_SWITCH:   prefix = "[Decide]"; break;
            case EventType::ACTION_PHYSICAL:   prefix = "[Act]"; break;
            case EventType::ACTION_VERBAL:     prefix = "[Say]"; break;
            default: prefix = "[Info]";
        }
        // 格式: [T+1.2s] [Think] Planning to grasp cup...
        return prefix + " " + summary;
    }
};

// 组件健康/运行状态
enum class ComponentState {
    OFFLINE,        // 未连接
    INITIALIZING,   // 初始化中
    READY,          // 正常待机
    ACTIVE,         // 正在工作 (如正在录音、机械臂正在运动)
    STALLED,        // 堵转/卡死 (仅针对电机)
    ERROR,          // 硬件故障
    OCCLUDED        // 遮挡 (仅针对视觉)
};

// 系统全局状态快照
struct SystemStatus {
    ComponentState vision_state = ComponentState::OFFLINE;
    ComponentState audio_state = ComponentState::OFFLINE;
    ComponentState arm_state = ComponentState::OFFLINE;
    
    float battery_voltage = 0.0;
    float cpu_temperature = 0.0;
};

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

enum class FrameQuality {
    VALID,          // 高质量，且有变化
    BLURRY,         // 运动模糊，已丢弃
    STATIC,         // 画面静止，已跳过
    DARK            // 光线过暗 (可选)
};

// 视觉帧
struct VisualFrame {
    TimePoint timestamp;
    cv::Mat image;

    // [新增] 质量与状态标签
    FrameQuality quality = FrameQuality::VALID;
    double blur_score = 0.0;    // 清晰度评分
    double motion_score = 0.0;  // 变化幅度评分

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

// 音频活动检测状态
enum class VADState {
    SILENCE,        // 静音/背景噪声
    SPEECH_START,   // 刚检测到语音开始
    SPEECH_ACTIVE,  // 正在持续说话
    SPEECH_END      // 语音结束 (触发 ASR)
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
    // std::vector<int16_t> audio_window; 原始音频在场景中无法直接使用，还是使用转换后的脚本
    std::optional<AudioTranscript> latest_transcript;
    std::string attention;  // 来自上层的注意力
    SystemStatus system_status;
};
} // namespace titan::core