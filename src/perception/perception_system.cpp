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
    titan::core::VisualFrame frame;
    frame.timestamp = t_capture;
    // 存入原始图 (供显示或后续回溯)
    frame.image = img.clone(); 

    // [Step 0] 预处理准备：转灰度 + 缩小 (加速计算)
    cv::Mat small_gray;
    cv::cvtColor(img, small_gray, cv::COLOR_BGR2GRAY);
    // 缩放到 320 宽，保持比例，大幅加速计算
    float scale = 320.0f / img.cols;
    cv::resize(small_gray, small_gray, cv::Size(), scale, scale);

    // [Step 1] 模糊检测 (L0 Filter)
    double blur_val = calculateBlurScore(small_gray);
    frame.blur_score = blur_val;

    if (blur_val < blur_threshold_) {
        // 图片太糊了，可能是机器人在快速甩头
        // 此时强行跑 YOLO 只会得到幻觉
        frame.quality = titan::core::FrameQuality::BLURRY;
        
        // 推入缓冲，但没有任何 detection
        vision_track_.push(frame);
        
        // Log 方便调试，实际运行时可去掉
        // std::cout << "[Vision] Skipped BLURRY frame. Score: " << blur_val << std::endl;
        return; 
    }

    // [Step 2] 运动/静止检测 (L1 Filter)
    double motion_val = calculateMotionScore(small_gray);
    frame.motion_score = motion_val;
    skipped_count_++;

    // 触发处理的条件：
    // 1. 画面变化够大 (有东西动了，或者机器人动了)
    // 2. 或者是第一帧
    // 3. 或者是强制心跳帧 (防止长时间静止导致漏掉微小变化)
    bool should_process = (motion_val > motion_threshold_) || 
                          (last_processed_gray_.empty()) ||
                          (skipped_count_ > force_process_interval_);

    if (!should_process) {
        frame.quality = titan::core::FrameQuality::STATIC;
        vision_track_.push(frame);
        return; 
    }

    // [Step 3] 深度感知 (L2 Process - YOLO/VLM)
    // 只有通过了前两关，才消耗算力
    
    // 重置计数器，更新参考帧
    skipped_count_ = 0;
    last_processed_gray_ = small_gray.clone(); 

    // 调用 YOLO
    // frame.detections = yolo_engine_.detect(img); 
    frame.quality = titan::core::FrameQuality::VALID;
    
    vision_track_.push(frame);
    
    // std::cout << "[Vision] Processed VALID frame. Motion: " << motion_val << "%" << std::endl;
}

void PerceptionSystem::onAudioMicRaw(const std::vector<int16_t>& pcm, TimePoint t_start) {
    AudioChunk chunk;
    chunk.timestamp = t_start;
    chunk.pcm_data = pcm;
    audio_track_.push(chunk);
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
    
    // [新增] 实时计算具身指标
    // 假设我们有 SceneMemoryEngine 的引用或辅助计算类
    // ctx.env_metrics = metric_calculator_.compute(vision_frame, system_status);
    
    // 模拟数据填充
    ctx.env_metrics.battery_level = 0.85;
    ctx.env_metrics.estimated_width = 3.2; // 宽敞
    ctx.env_metrics.clearance_ratio = 3.2 / 0.6;
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

// 1. 拉普拉斯方差法检测模糊 (Variance of Laplacian)
// 原理：清晰图片边缘多，拉普拉斯变换后方差大；模糊图片方差小。
double PerceptionSystem::calculateBlurScore(const cv::Mat& gray) {
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    // 方差 = 标准差的平方
    return stddev.val[0] * stddev.val[0];
}

// 2. 帧差法检测运动 (Frame Difference)
double PerceptionSystem::calculateMotionScore(const cv::Mat& curr_gray) {
    if (last_processed_gray_.empty()) return 100.0; // 第一帧必处理

    cv::Mat diff, diff_thresh;
    cv::absdiff(curr_gray, last_processed_gray_, diff);
    
    // 简单的阈值化，统计变化像素占比
    cv::threshold(diff, diff_thresh, 30, 255, cv::THRESH_BINARY);
    
    int non_zero = cv::countNonZero(diff_thresh);
    double percent = (double)non_zero / (double)(curr_gray.total()) * 100.0;
    
    return percent;
}

// 辅助函数 1：基于能量和 ZCR 进行判断
bool PerceptionSystem::isSpeechChunk(const std::vector<int16_t>& pcm) {
    if (pcm.empty()) return false;

    // 1. 能量计算 (Energy): 衡量响度
    long long energy = 0;
    for (int16_t sample : pcm) {
        energy += (long long)sample * sample;
    }
    // 取平均平方根能量
    double rms_energy = std::sqrt((double)energy / pcm.size());

    // 2. 过零率计算 (ZCR): 衡量频率特征
    int zero_crossings = 0;
    for (size_t i = 1; i < pcm.size(); ++i) {
        if ((pcm[i] >= 0 && pcm[i-1] < 0) || (pcm[i] < 0 && pcm[i-1] >= 0)) {
            zero_crossings++;
        }
    }

    // 语音判断逻辑：
    // 1. 必须足够响亮 (过滤掉微弱的背景音)
    // 2. ZCR 必须在合理范围内 (高 ZCR 是白噪声，低 ZCR 是持续的低频音)
    bool is_voiced = (rms_energy > ENERGY_THRESHOLD) && (zero_crossings < ZCR_THRESHOLD);
    
    return is_voiced;
}

// 辅助函数 2：触发 ASR 线程（同之前实现的异步逻辑）
void PerceptionSystem::triggerASRAsync(std::vector<int16_t> pcm_data) {
    // ... 将 pcm_data 放入队列，ASR Worker Thread 异步处理 ...
    std::lock_guard<std::mutex> lock(audio_mtx_);
    audio_buffer_.insert(audio_buffer_.end(), pcm_data.begin(), pcm_data.end());
    
    // 简单模拟 VAD：每积攒 0.5秒 (16k采样率 -> 8000点) 触发一次检测
    // 实际应使用 WebRTC VAD 或 Silero VAD
    if (audio_buffer_.size() > 8000) {
        audio_cv_.notify_one();
    }
    // 下一步由ASR线程完成转换
}

// [核心] 音频输入处理函数
void PerceptionSystem::onAudioMic(const std::vector<int16_t>& pcm) {
    bool is_speech = isSpeechChunk(pcm);
    titan::core::VADState current_state = vad_state_.load();

    if (current_state == titan::core::VADState::SILENCE) {
        if (is_speech) {
            // 状态转换: SILENCE -> SPEECH_ACTIVE
            vad_state_ = titan::core::VADState::SPEECH_ACTIVE;
            asr_audio_buffer_.insert(asr_audio_buffer_.end(), pcm.begin(), pcm.end());
            silence_chunk_counter_ = 0;
            // 
        }
    } 
    else if (current_state == titan::core::VADState::SPEECH_ACTIVE) {
        if (is_speech) {
            // 继续说话
            asr_audio_buffer_.insert(asr_audio_buffer_.end(), pcm.begin(), pcm.end());
            silence_chunk_counter_ = 0;
        } else {
            // 检测到静音尾部
            silence_chunk_counter_++;
            // 依然积累一小段静音，防止用户说话中断
            asr_audio_buffer_.insert(asr_audio_buffer_.end(), pcm.begin(), pcm.end());

            if (silence_chunk_counter_ > MAX_SILENCE_CHUNKS) {
                // 状态转换: SPEECH_ACTIVE -> SPEECH_END (触发 ASR)
                vad_state_ = titan::core::VADState::SPEECH_END;
                
                // 异步触发 ASR
                triggerASRAsync(std::move(asr_audio_buffer_)); 
                asr_audio_buffer_.clear();
                
                // 重置状态
                vad_state_ = titan::core::VADState::SILENCE;
            }
        }
    }
}
} // namespace titan::perception

