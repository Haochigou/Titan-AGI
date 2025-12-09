#include "titan/agent/titan_agent.h"
#include "titan/perception/perception_system.h"
#include "titan/perception/attention_engine.h"
#include "titan/cognition/object_cognition.h"
#include "titan/memory/cognitive_stream.h"
#include "titan/learning/strategy_optimizer.h"
#include "titan/agent/multi_task_executive.h"
#include "titan/agent/behavior_arbiter.h"
#include "titan/control/fep_controller.h"
#include "hal/tts_engine.h"
#include "titan/control/action_manager.h"

#include <iostream>
#include <future>

namespace titan::agent {

using namespace titan::core;
using namespace titan::perception;
using namespace titan::cognition;

class TitanAgentImpl {
public:
    // --- 子系统实例 ---
    PerceptionSystem perception_;
    ObjectCognitionEngine cognition_engine_;
    
    titan::memory::CognitiveStream stream_;
    titan::learning::StrategyOptimizer learner_;
    
    MultiTaskExecutive multi_executive_;
    AttentionEngine attention_sys_;
    BehaviorArbiter arbiter_;
    
    control::FEPController controller_;
    control::ActionManager action_mgr_;
    hal::TTSEngine tts_engine_;

    // --- 构造函数 ---
    TitanAgentImpl() : action_mgr_(nullptr) { 
        // 绑定 StrategyOptimizer 到 Planner，实现策略检索
        multi_executive_.injectStrategyOptimizer(&learner_);
        
        // 绑定 CognitiveStream 到 Planner，实现基于记忆的规划
        multi_executive_.injectMemoryStream(&stream_);
    }

    // --- 核心心跳函数 (The Heartbeat) ---
    void tick() {
        auto now = std::chrono::steady_clock::now();
        
        // =========================================================
        // Phase 1: 感知对齐与注入 (Perception Alignment & Injection)
        // =========================================================
        
        // 1.1 获取时空对齐的上下文 (非阻塞)
        FusedContext ctx = perception_.getContext(now);

        // 1.2 自身状态检查 (Meta-Cognition)
        // [闭环控制] 如果视觉糊了，立即抑制运动增益
        if (ctx.vision.has_value() && ctx.vision->quality == FrameQuality::BLURRY) {
            controller_.reduceGainForStability(); 
        } else {
            controller_.updateInternalState(); // 尝试恢复增益
        }

        // 1.3 认知流注入 (Stream Injection)
        // 将"瞬时信号"转化为"历史事件"
        stream_.addVisualContext(ctx);
        stream_.addSystemStatus(ctx.system_status);

        // 1.4 音频处理 (带自我抑制机制)
        // 全双工关键：如果我在说话，ASR 听到的可能是回声，需要抑制或标记
        if (ctx.latest_transcript.has_value()) {
            std::string user_text = ctx.latest_transcript->text;
            if (tts_engine_.isSpeaking()) {
                // 简单抑制：自己说话时不听指令，或者作为 barge-in 打断信号
                if (user_text == "Stop") {
                    onUserCommand(user_text); // 允许打断
                }
            } else {
                stream_.addEvent(EventType::PERCEPTION_AUDIO, "User said: " + user_text);
                onUserCommand(user_text);
            }
        }

        // =========================================================
        // Phase 2: 世界模型更新 (World Modeling)
        // =========================================================
        
        // 将 2D 检测框升级为 3D 实体 (Object Permanence)
        std::vector<VisualDetection> raw_dets;
        if (ctx.vision.has_value()) {
            // 适配层：将 VisualFrame::Detection 转为 VisualDetection
            for(const auto& d : ctx.vision->detections) {
                VisualDetection vd; 
                vd.label = d.label; vd.box = d.box; vd.confidence = d.confidence;
                // vd.mask = d.mask; // 如果有mask
                raw_dets.push_back(vd);
            }
        }
        cognition_engine_.update(raw_dets, now);

        // =========================================================
        // Phase 3: 战略与任务调度 (Strategic & Executive)
        // =========================================================

        // 3.1 多任务管家更新 (包含 LLM 异步规划结果的检查)
        // 这里会进行任务切换、步骤推进、预期生成
        multi_executive_.update(ctx, cognition_engine_);

        // 3.2 学习闭环 (Learning Loop)
        // 检查是否有任务刚刚完成或失败
        auto finished_task = multi_executive_.popFinishedTask();
        if (finished_task) {
            bool success{false};
            // [异步] 触发反思学习，总结策略
            // 不阻塞 tick，放在后台线程跑
            std::thread([this, task = *finished_task, &success]() {
                success = (task.status == TaskStatus::COMPLETED);
                learner_.reflectOnEpisode(stream_.getHistory(), success);
            }).detach();
            
            // 语音反馈
            performBehavior(EventType::ACTION_VERBAL, 
                success ? "Task complete." : "Task failed, I am learning from this.");
        }

        // =========================================================
        // Phase 4: 注意力与竞价 (Attention & Bidding)
        // =========================================================

        // 4.1 获取 Top-Down 目标
        std::string focus_target = multi_executive_.getTopDownTarget();
        
        // 4.2 获取预测误差 (惊奇度)
        // 如果 Executive 在执行中发现预期不符 (Prediction Error)，注入注意力
        double pred_error = multi_executive_.getCurrentPredictionError();
        std::map<std::string, double> surprise_map; 
        if (!focus_target.empty()) surprise_map[focus_target] = pred_error;

        // 4.3 计算注意力 (Attention Saliency)
        // 融合了：视觉显著性 + 任务目标 + 惊奇度 + 历史抑制
        auto saliency_map = attention_sys_.computeSaliency(raw_dets, focus_target, surprise_map);

        // 4.4 行为竞价 (Generating Proposals)
        std::vector<ActionProposal> proposals;
        
        // A. 安全反射 (Reflex) - 最高优先级
        proposals.push_back(proposeSafety(ctx));

        // B. 任务执行 (Cognitive) - 基于 Executive 状态
        proposals.push_back(multi_executive_.getBestProposal(ctx, cognition_engine_));

        // C. 好奇心探索 (Curiosity) - 基于注意力
        proposals.push_back(proposeExploration(saliency_map));

        // =========================================================
        // Phase 5: 仲裁与并行输出 (Arbitration & Output)
        // =========================================================

        // 5.1 赢家通吃 (Winner-Take-All)
        arbiter_.arbitrate(proposals);

        // 5.2 执行赢家逻辑
        /* 在arbitrate 已经完成执行
        if (winner.has_value()) {
            // execute() 内部会调用 performBehavior 或直接控制 controller
            // 例如：如果是 Task Proposal，它会计算 FEP 并调用 performBehavior
            winner.value().execute();
        }
        */
    }

    // --- 统一行为接口 (Unified Action Interface) ---
    void performBehavior(EventType type, const std::string& content, const json& data = {}) {
        // 1. 记录到流 (Contextualize)
        // 这样 LLM 就能知道"我刚才做了什么"，实现自我意识
        stream_.addEvent(type, content, data);

        // 2. 并行物理输出 (Parallel Output)
        if (type == EventType::ACTION_VERBAL) {
            // [Audio Channel]
            tts_engine_.speakAsync(content); 
        } 
        else if (type == EventType::ACTION_PHYSICAL) {
            // [Motor Channel]
            // 如果 content 是预定义指令
            if (content == "STOP") {
                action_mgr_.execute(Eigen::VectorXd::Zero(6), "STOP");
            }
            // 注意：复杂的连续控制通常由 FEPController 在 winner.execute() 闭包中直接驱动
            // 这里主要处理离散动作
        }
    }

    // --- 辅助提案生成器 ---
    ActionProposal proposeSafety(const FusedContext& ctx) {
        ActionProposal p;
        p.source = "SafetyReflex";
        p.priority = 0.0;

        // 检查：如果手臂堵转，或者外界有急停信号
        if (ctx.system_status.arm_state == ComponentState::STALLED) {
            p.priority = 100.0; // 绝对优先
            p.description = "Emergency Halt: Arm Stalled";
            p.execute = [this]() {
                performBehavior(EventType::ACTION_PHYSICAL, "STOP");
                performBehavior(EventType::ACTION_VERBAL, "My arm is stuck.");
            };
        }
        return p;
    }

    ActionProposal proposeExploration(const std::vector<AttentionalObject>& saliency) {
        ActionProposal p;
        p.source = "Exploration";
        p.priority = 0.0;
        
        // 如果当前没任务，且有个东西特别显眼 (Bottom-up score high)
        if (!multi_executive_.hasActiveTask() && !saliency.empty()) {
            const auto& obj = saliency[0];
            if (obj.bottom_up_score > 0.8) {
                p.priority = 2.0; // 低优先级
                p.description = "Look at " + obj.raw_det.label;
                p.execute = [this, obj]() {
                    // 转头看过去 (Mock)
                    // head_controller_.lookAt(obj.raw_det.box);
                    performBehavior(EventType::ACTION_PHYSICAL, "LookAt:" + obj.raw_det.label);
                };
            }
        }
        return p;
    }

    void onUserCommand(const std::string& text) {
        // 1. 记录
        stream_.addEvent(EventType::PERCEPTION_AUDIO, "User Command: " + text);

        if (text == "Stop") {
            // 硬件急停
            performBehavior(EventType::ACTION_PHYSICAL, "STOP");
            tts_engine_.stop();
            // 清空任务队列
            multi_executive_.abortAll();
        } else {
            // 2. 扔给 System 2 (异步规划)
            multi_executive_.addInstruction(text);
        }
    }
};

// TitanAgent Wrapper to match header
TitanAgent::TitanAgent() : impl_(new TitanAgentImpl()) {}
TitanAgent::~TitanAgent() = default;
void TitanAgent::tick() { impl_->tick(); }
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
    impl_->perception_.onAudioMic(pcm);
}
} // namespace titan::agent