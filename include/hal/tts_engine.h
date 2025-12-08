#pragma once
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <functional>

namespace titan::hal {

class TTSEngine {
public:
    TTSEngine();
    ~TTSEngine();

    // --- 核心接口 ---
    
    // 异步播放语音
    // 调用后立即返回，语音加入后台队列
    void speakAsync(const std::string& text);

    // 立即停止（打断）
    // 用于用户说 "Stop" 时，机器人立刻闭嘴
    void stop();

    // 机器人是否正在说话？
    // (用于抑制 ASR，避免自言自语)
    bool isSpeaking() const;

private:
    // 任务队列
    std::queue<std::string> speech_queue_;
    std::mutex queue_mtx_;
    std::condition_variable cv_;

    // 工作线程
    std::thread worker_;
    std::atomic<bool> running_{true};
    std::atomic<bool> is_speaking_{false};

    // --- 后端实现 ---
    void workerLoop();
    void synthesizeAndPlay(const std::string& text);
};

} // namespace titan::hal