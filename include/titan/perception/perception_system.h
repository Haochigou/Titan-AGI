#pragma once
#include "titan/core/types.h"
#include "titan/core/ring_buffer.h"
#include "hal/hardware_drivers.h"
#include <opencv2/imgproc.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
namespace titan::perception {

class PerceptionSystem {
private:
    titan::core::RingTrack<titan::core::RobotState> body_track_{2000};
    titan::core::RingTrack<titan::core::VisualFrame> vision_track_{100};
    titan::core::RingTrack<titan::core::AudioChunk> audio_track_{500};

    // [新增] 文本语义轨道
    titan::core::RingTrack<titan::core::AudioTranscript> text_track_{50};

    // 持有驱动的引用或状态指针
    titan::hal::CameraDriver* cam_driver_ = nullptr;
    titan::hal::RobotBodyDriver* body_driver_ = nullptr;

    // [新增] 音频 VAD 状态与缓存
    std::atomic<titan::core::VADState> vad_state_ {titan::core::VADState::SILENCE};
    std::vector<int16_t> asr_audio_buffer_; // 积累的 PCM 数据

    // VAD 参数 (需要根据实际采样率调整)
    const int ENERGY_THRESHOLD = 500;   // 能量阈值 (判断是否足够响亮)
    const int ZCR_THRESHOLD = 1500;     // 过零率阈值 (判断是否为噪声/语音)
    const int MAX_SILENCE_CHUNKS = 10;  // 持续 10 个静音块后，视为语音结束
    int silence_chunk_counter_ = 0;

    // 辅助函数
    bool isSpeechChunk(const std::vector<int16_t>& pcm);
    
    // [修改] ASR 逻辑需要接收累积的 Buffer
    void triggerASRAsync(std::vector<int16_t> pcm_data);

    // --- ASR 异步处理机制 ---
    std::vector<int16_t> audio_buffer_; // 积攒 PCM
    std::mutex audio_mtx_;
    std::condition_variable audio_cv_;
    std::thread asr_thread_;
    std::atomic<bool> running_{true};
    
    // 模拟 ASR 引擎 (实际应调用 Whisper.cpp 接口)
    std::string mockRunASR(const std::vector<int16_t>& pcm);

    void asrWorkerLoop();

    // [新增] 视觉预处理状态
    cv::Mat last_processed_gray_; // 上一帧处理过的灰度图
    
    // [参数配置] 
    // 可以通过认知层动态调整 (例如：紧急情况下降低阈值)
    double blur_threshold_ = 100.0;   // 低于此值视为模糊
    double motion_threshold_ = 5.0;   // 像素变化百分比低于此值视为静止
    int force_process_interval_ = 30; // 每30帧强制处理一次 (心跳机制)
    int skipped_count_ = 0;

    // 辅助算法
    double calculateBlurScore(const cv::Mat& gray);
    double calculateMotionScore(const cv::Mat& current_gray);
public:
    PerceptionSystem();
    ~PerceptionSystem();
    void onImuData(const titan::core::RobotState& rs);
    void onImuJointData(const titan::core::RobotState& s);
    void onCameraFrame(const cv::Mat& img, titan::core::TimePoint t_capture);
    void onAudioMicRaw(const std::vector<int16_t>& pcm, titan::core::TimePoint t_start);
    void onAudioMic(const std::vector<int16_t>& pcm);

    titan::core::FusedContext getContext(titan::core::TimePoint t_query);
    std::vector<int16_t> retrieveRawAudio(double duration_sec);
    void getHistoryContexts(titan::core::TimePoint t_end, double duration, std::vector<titan::core::FusedContext>& out_contexts);
    void reset();
    void process();
    // 注入驱动指针，以便查询状态
    void attachDrivers(titan::hal::CameraDriver* cam, titan::hal::RobotBodyDriver* body) {
        cam_driver_ = cam;
        body_driver_ = body;
    }
    void setVisualSensitivity(double blur_th, double motion_th) {
        blur_threshold_ = blur_th;
        motion_threshold_ = motion_th;
    }
};

} // namespace titan::perception