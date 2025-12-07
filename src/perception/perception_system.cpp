#include "titan/perception/perception_system.h"
#include <iostream>
#include <chrono>

namespace titan::perception {

using namespace titan::core;

PerceptionSystem::PerceptionSystem() {
    // 启动 ASR 后台线程
    asr_thread_ = std::thread(&PerceptionSystem::asrWorkerLoop, this);
}

PerceptionSystem::~PerceptionSystem() {
    running_ = false;
    audio_cv_.notify_all();
    if (asr_thread_.joinable()) asr_thread_.join();
}

void PerceptionSystem::onImuJointData(const RobotState& s) { body_track_.push(s); }

void PerceptionSystem::onCameraFrame(const cv::Mat& img, TimePoint t_capture) {
    VisualFrame frame;
    frame.timestamp = t_capture;
    frame.image = img.clone();
    // 模拟 YOLO ...
    vision_track_.push(frame);
}

void PerceptionSystem::onAudioMicRaw(const std::vector<int16_t>& pcm, TimePoint t_start) {
    AudioChunk chunk;
    chunk.timestamp = t_start;
    chunk.pcm_data = pcm;
    audio_track_.push(chunk);
}

// [关键] 接收麦克风数据并积攒
void PerceptionSystem::onAudioMicASR(const std::vector<int16_t>& pcm) {
    std::lock_guard<std::mutex> lock(audio_mtx_);
    audio_buffer_.insert(audio_buffer_.end(), pcm.begin(), pcm.end());
    
    // 简单模拟 VAD：每积攒 0.5秒 (16k采样率 -> 8000点) 触发一次检测
    // 实际应使用 WebRTC VAD 或 Silero VAD
    if (audio_buffer_.size() > 8000) {
        audio_cv_.notify_one();
    }
}

// [关键] ASR 工作线程
void PerceptionSystem::asrWorkerLoop() {
    while (running_) {
        std::vector<int16_t> process_chunk;
        {
            std::unique_lock<std::mutex> lock(audio_mtx_);
            audio_cv_.wait(lock, [this]{ return !running_ || audio_buffer_.size() > 8000; });
            
            if (!running_) break;

            // 取出数据进行处理
            process_chunk = audio_buffer_;
            audio_buffer_.clear(); 
        }

        // 调用 ASR (耗时操作)
        std::string text = mockRunASR(process_chunk);

        if (!text.empty()) {
            AudioTranscript trans;
            trans.timestamp = std::chrono::steady_clock::now();
            trans.text = text;
            trans.confidence = 0.95;
            trans.processed = false;

            std::cout << "[ASR] Transcribed: " << text << std::endl;
            text_track_.push(trans);
        }
    }
}

std::string PerceptionSystem::mockRunASR(const std::vector<int16_t>& pcm) {
    // 模拟延迟
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 简单的 Mock 逻辑：随机返回指令用于测试
    static int counter = 0;
    counter++;
    if (counter % 50 == 0) return "Find the cup";
    if (counter % 120 == 0) return "Stop";
    return ""; // 大部分时间是静音
}

FusedContext PerceptionSystem::getContext(TimePoint t_query) {
    FusedContext ctx;
    ctx.timestamp = t_query;

    // 1. 本体和视觉 (同前)
    auto [prev_r, next_r] = body_track_.getBracket(t_query);
    if (prev_r) ctx.robot = *prev_r; // 简化写法
    
    auto [v_prev, v_next] = vision_track_.getBracket(t_query);
    if (v_prev) ctx.vision = *v_prev;

    // 2. [修改] 获取最新的未处理文本
    auto latest_trans = text_track_.getLatest();
    if (latest_trans && !latest_trans->processed) {
        // 只有当这条指令发生在查询时间附近（或者在过去几秒内）才算有效
        double age = std::chrono::duration<double>(t_query - latest_trans->timestamp).count();
        if (age < 2.0 && age > -0.5) { 
            ctx.latest_transcript = *latest_trans;
        }
    }
    if (cam_driver_) ctx.system_status.vision_state = cam_driver_->getState();
    if (body_driver_) ctx.system_status.arm_state = body_driver_->getState();
        
    // 模拟电池
    ctx.system_status.battery_voltage = 24.5;
    return ctx;
}

void PerceptionSystem::process() {
    // TODO 这里可以添加周期性处理逻辑，比如清理过期数据等
    
}

void PerceptionSystem::reset() {
    // TODO 
}
void PerceptionSystem::onImuData(const RobotState& rs) {
        // ... push to body_track_ ...
}
void PerceptionSystem::getHistoryContexts(TimePoint t_end, double duration, std::vector<FusedContext>& out_contexts) {
    /*
    TimePoint t_start = t_end - std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(duration));
    auto body_records = body_track_.getInRange(t_start, t_end);
    auto vision_records = vision_track_.getInRange(t_start, t_end);
    auto text_records = text_track_.getInRange(t_start, t_end);
    
    // 简单合并逻辑：按时间戳排序
    std::vector<TimePoint> all_timestamps;
    for (const auto& r : body_records) all_timestamps.push_back(r.timestamp);
    for (const auto& r : vision_records) all_timestamps.push_back(r.timestamp);
    for (const auto& r : text_records) all_timestamps.push_back(r.timestamp);
    
    std::sort(all_timestamps.begin(), all_timestamps.end());
    all_timestamps.erase(std::unique(all_timestamps.begin(), all_timestamps.end()), all_timestamps.end());
    
    for (const auto& t : all_timestamps) {
        FusedContext ctx = getContext(t);
        out_contexts.push_back(ctx);
    }
    */
}



} // namespace titan::perception

