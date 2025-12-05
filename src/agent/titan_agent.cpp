#include "titan/agent/titan_agent.h"
#include "titan/perception/attention_engine.h" // 新增
#include "titan/perception/perception_system.h" // 新增
#include "titan/agent/behavior_arbiter.h"    // 新增
#include "titan/control/fep_controller.h"
#include "titan/agent/multi_task_executive.h" // 新增
#include <iostream>
#include <map>

namespace titan::agent {

using namespace titan::core;
using namespace titan::perception;
using namespace Eigen;

class TitanAgentImpl { // PImpl 模式或直接实现
public:
    titan::perception::PerceptionSystem perception_;
    titan::control::FEPController controller_;
    titan::perception::AttentionEngine attention_sys_;
    BehaviorArbiter arbiter_;

    // ExecutiveSystem executive_; // 新增执行系统
    MultiTaskExecutive multi_executive_; // 替换旧的 executive_
    titan::memory::EntityMemoryManager memory_mgr_;
    // 当前心智状态
    std::string current_user_task_; // 用户指令
    std::map<std::string, double> surprise_memory_; // 短期惊奇记忆
    bool is_emergency_ = false;

    // --- 行为模块 (Schema) ---

    // 1. 安全模块 (最高优先级)
    ActionProposal proposeSafety() {
        ActionProposal p;
        p.source = "Safety";
        
        if (is_emergency_) {
            p.priority = 100.0; // 绝对优先
            p.description = "HALT execution";
            p.execute = [this]() { 
                // send_stop_command(); 
                std::cout << "!!! SAFETY STOP !!!" << std::endl;
            };
        } else {
            p.priority = 0.0;
        }
        return p;
    }

    // 2. 任务执行模块 (Top-Down Driven)
    ActionProposal proposeTask(const std::vector<AttentionalObject>& salience_map, FusedContext& ctx) {
        ActionProposal p;
        p.source = "Task";
        p.priority = 0.0;

        if (current_user_task_.empty()) return p;

        // 如果注意力地图里有高分 Top-down 物体
        if (!salience_map.empty()) {
            const auto& best_obj = salience_map[0];
            
            // 只有当它是 Top-down 关注的对象时，才产生任务行为
            if (best_obj.top_down_score > 0.8) {
                p.priority = 5.0 + best_obj.total_saliency; // 基础分 + 显著性
                p.description = "Grasp target: " + best_obj.raw_det.label;
                
                p.execute = [this, best_obj, ctx]() {
                    // 调用 FEP 控制器
                    VectorXd feats(2);
                    feats << best_obj.raw_det.box.width, ctx.robot.ee_pos(2);
                    auto ctrl = controller_.solve(feats);
                    
                    // 模拟执行与反馈
                    double ground_truth = feats(0) * 3.0;
                    double surprise = std::abs(ground_truth - ctrl.force);
                    
                    // **关键：更新惊奇度记忆**
                    // 这会反馈给下一帧的 Bottom-up 注意力
                    surprise_memory_[best_obj.raw_det.label] = surprise;
                    
                    controller_.learn(feats, ground_truth, ctrl.force);
                    attention_sys_.inhibit(best_obj.raw_det.label); // 完成动作后抑制，转向下一步
                };
            } else {
                // 有任务但没看到物体 -> 搜索行为
                p.priority = 2.0;
                p.description = "Scanning for " + current_user_task_;
                p.execute = [](){ std::cout << ">>> Scanning environment..." << std::endl; };
            }
        }
        return p;
    }

    // 3. 探索模块 (Bottom-Up Driven / Curiosity)
    ActionProposal proposeExploration(const std::vector<AttentionalObject>& salience_map) {
        ActionProposal p;
        p.source = "Exploration";
        p.priority = 0.0;

        if (!salience_map.empty()) {
            const auto& interesting_obj = salience_map[0];
            
            // 如果有个物体非常显著 (Bottom-up high)，但跟当前任务无关
            if (interesting_obj.bottom_up_score > 0.8 && interesting_obj.top_down_score < 0.2) {
                p.priority = 3.0; // 低于明确的任务，但高于闲置
                p.description = "Look at surprise: " + interesting_obj.raw_det.label;
                
                p.execute = [this, interesting_obj]() {
                    std::cout << "??? What is that? (High Surprise/Motion detected)" << std::endl;
                    attention_sys_.inhibit(interesting_obj.raw_det.label); // 看一眼就够了
                    
                    // 清除惊奇度 (因为我看过了，消除不确定性了)
                    surprise_memory_[interesting_obj.raw_det.label] = 0.0;
                };
            }
        }
        return p;
    }

public:
    void tick() {
        // 1. 获取感知上下文
        auto now = std::chrono::steady_clock::now();
        FusedContext ctx = perception_.getContext(now);

        if (ctx.latest_transcript.has_value()) {
            const auto& trans = ctx.latest_transcript.value();
            std::string cmd = trans.text;
            
            std::cout << "[Agent] Heard Command: " << cmd << std::endl;
            
            // 标记为已处理，防止下一帧重复执行
            // 注意：这里需要一种机制去标记 RingBuffer 里的数据
            // 为简化，我们可以在 onUserCommand 内部做去重，或者在这里直接调用
            this->onUserCommand(cmd); 
        }
        if (ctx.latest_transcript.has_value()) {
            // 假设听到 "小明在哭"
            // 实际这里应该结合 VLM 和 Audio 分析出结构化事件
            int xiaoming_id = 101; 
            memory_mgr_.recordObservation(xiaoming_id, "Xiao Ming is crying", "emotion");
        }
        // 2. 更新多任务状态
        multi_executive_.update(ctx);

        // 3. [关键整合] 获取 Top-down 目标 和 预测误差
        // 此时使用的是 MultiTaskExecutive
        std::string focus_target = multi_executive_.getTopDownTarget();
        
        // 如果当前任务产生了预测误差，将其注入惊奇度记忆
        // 这样 AttentionEngine 下一帧就会自动关注那个"出乎意料"的地方
        // (需要给 MultiTaskExecutive 加一个接口 getCurrentPredictionError)
        // double error = multi_executive_.getCurrentPredictionError();
        // if (error > 0.1 && !focus_target.empty()) {
        //    surprise_memory_[focus_target] += error;
        // }

        // 4. 计算注意力
        std::vector<VisualFrame::Detection> raw_dets;
        if (ctx.vision.has_value()) raw_dets = ctx.vision->detections;
        
        auto saliency = attention_sys_.computeSaliency(raw_dets, focus_target, surprise_memory_);

        // 5. 竞价
        std::vector<ActionProposal> proposals;
        proposals.push_back(proposeSafety());
        proposals.push_back(multi_executive_.getBestProposal(ctx)); // 获取最佳任务提案
        proposals.push_back(proposeExploration(saliency));

        // 6. 仲裁
        arbiter_.arbitrate(proposals);
    }

    void onUserCommand(const std::string& text) {
        // 简单的去重逻辑 (Debounce)
        static std::string last_cmd = "";
        static auto last_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();

        if (text == last_cmd && std::chrono::duration<double>(now - last_time).count() < 2.0) {
            return; // 2秒内忽略重复指令
        }
        last_cmd = text;
        last_time = now;

        // 这里的 text 已经是 ASR 转好的文本了
        if (text == "Stop") {
            // 紧急停止
        } else {
            // 扔给 System 2 (Strategic Planner) 进行语义分析和任务生成
            multi_executive_.addInstruction(text);
        }
    }
    
    // ... feedSensors ...
};

// 2. TitanAgent 的构造与析构桥接

// 构造函数：初始化 unique_ptr
TitanAgent::TitanAgent() : impl_(std::make_unique<TitanAgentImpl>()) {}

// 析构函数：必须在 .cpp 中定义（即使是默认的），因为这里 TitanAgentImpl 才是完整类型
TitanAgent::~TitanAgent() = default;

// 3. 转发调用 (Forwarding calls)

void TitanAgent::tick() {
    impl_->tick();
}

void TitanAgent::onUserCommand(const std::string& text) {
    impl_->onUserCommand(text);
}

void TitanAgent::feedSensors(const titan::core::RobotState& rs, const cv::Mat& img, titan::core::TimePoint t_img) {
    // 假设 Impl 中有 perception_ 成员
    impl_->perception_.onImuJointData(rs);
    if (!img.empty()) impl_->perception_.onCameraFrame(img, t_img);
}

void TitanAgent::feedAudio(const std::vector<int16_t>& pcm) {
    // 假设 Impl 中有 perception_ 成员
    impl_->perception_.onAudioMicASR(pcm);
}
} // namespace titan::agent