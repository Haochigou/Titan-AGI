#pragma once
#include "titan/core/types.h"
#include "titan/core/ring_buffer.h"
#include "hal/hardware_drivers.h"
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

    // --- ASR 异步处理机制 ---
    std::vector<int16_t> audio_buffer_; // 积攒 PCM
    std::mutex audio_mtx_;
    std::condition_variable audio_cv_;
    std::thread asr_thread_;
    std::atomic<bool> running_{true};

    // 模拟 ASR 引擎 (实际应调用 Whisper.cpp 接口)
    std::string mockRunASR(const std::vector<int16_t>& pcm);

    void asrWorkerLoop();

public:
    PerceptionSystem();
    ~PerceptionSystem();
    void onImuData(const titan::core::RobotState& rs) {
        // ... push to body_track_ ...
}
    void onImuJointData(const titan::core::RobotState& s);
    void onCameraFrame(const cv::Mat& img, titan::core::TimePoint t_capture);
    void onAudioMicRaw(const std::vector<int16_t>& pcm, titan::core::TimePoint t_start);

    // [修改] 音频输入不再直接存 RingBuffer，而是喂给 ASR 线程
    void onAudioMicASR(const std::vector<int16_t>& pcm);

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
    
};

} // namespace titan::perception