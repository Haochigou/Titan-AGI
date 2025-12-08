#include "hal/tts_engine.h"
#include <iostream>
#include <cstdlib>

namespace titan::hal {

TTSEngine::TTSEngine() {
    worker_ = std::thread(&TTSEngine::workerLoop, this);
}

TTSEngine::~TTSEngine() {
    stop();
    running_ = false;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void TTSEngine::speakAsync(const std::string& text) {
    if (text.empty()) return;

    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        speech_queue_.push(text);
    }
    cv_.notify_one();
}

void TTSEngine::stop() {
    // 清空队列
    std::lock_guard<std::mutex> lock(queue_mtx_);
    std::queue<std::string> empty;
    std::swap(speech_queue_, empty);
    
    // TODO: 如果正在播放，需要调用底层 API 终止音频流
    // 例如: system("pkill espeak");
}

bool TTSEngine::isSpeaking() const {
    return is_speaking_.load();
}

void TTSEngine::workerLoop() {
    while (running_) {
        std::string current_text;

        // 1. 等待任务
        {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            cv_.wait(lock, [this] { return !speech_queue_.empty() || !running_; });

            if (!running_) break;

            current_text = speech_queue_.front();
            speech_queue_.pop();
        }

        // 2. 执行合成与播放
        if (!current_text.empty()) {
            is_speaking_ = true;
            synthesizeAndPlay(current_text);
            is_speaking_ = false;
        }
    }
}

// === [Backend Implementation] ===
// 这里是具体调用 TTS 库的地方
void TTSEngine::synthesizeAndPlay(const std::string& text) {
    std::cout << "[TTS] Speaking: " << text << std::endl;

    // 为了防止 Shell 注入，简单的清理一下引号
    std::string safe_text = text; 
    // std::replace(safe_text.begin(), safe_text.end(), '\"', ' ');

    // --- Option A: macOS (开发调试用) ---
    #ifdef __APPLE__
        std::string cmd = "say \"" + safe_text + "\"";
        int ret = std::system(cmd.c_str());
    
    // --- Option B: Linux / Embedded (espeak) ---
    // sudo apt-get install espeak-ng
    #elif __linux__
        // -ven+m3: 男声, -s150: 语速
        std::string cmd = "espeak-ng -ven+m3 -s150 \"" + safe_text + "\"";
        // system() 是阻塞的，正好符合我们的 worker 逻辑
        int ret = std::system(cmd.c_str());

    // --- Option C: Neural TTS (Sherpa-Onnx / Piper) [推荐生产环境使用] ---
    /*
     * sherpa_tts_engine_->Accept(text);
     * while (sherpa_tts_engine_->HasAudio()) {
     * auto audio = sherpa_tts_engine_->GetAudio();
     * audio_device_->Play(audio);
     * }
     */
    #else
        // Windows or other
        std::cout << "[Mock TTS] (System call not impl): " << safe_text << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // 模拟耗时
    #endif
    
    (void)ret; // 防止未使用变量警告
}

} // namespace titan::hal