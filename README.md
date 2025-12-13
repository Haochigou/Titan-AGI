# Titan-AGI: Embodied Cognitive Agent Framework

> **Towards System 2 Reasoning in Embodied Robotics.**

Titan-AGI 是一个高性能的具身智能（Embodied AI）框架，专为异构机器人平台设计。它不仅仅是连接 LLM 和控制器的胶水层，而是实现了一个完整的**认知架构（Cognitive Architecture）**。

该框架融合了 **System 1（快速感知与反应）** 和 **System 2（慢速推理与规划）**，赋予机器人**物体恒常性**、**环境场景记忆**以及**具身自我意识**（如体型感知与电量焦虑）。

-----

## 🚀 Key Features (核心特性)

### 1\. 🧠 Object Permanence & Cognition (物体恒常性与认知)

机器人不再只是逐帧检测物体。通过 **`ObjectCognitionEngine`**，Titan-AGI 维护着一个持久化的 3D 实体图谱。

  * **ID Tracking:** 即使物体被短暂遮挡或漏检，系统仍能维持 ID 不变。
  * **State Estimation:** 实时估算物体的 3D 速度，支持对动态目标的预判。
  * **Semantic Injection:** 自动注入常识属性（如 "Cup" -\> `graspable: true`, `fragile: true`），为高层规划提供决策依据。

### 2\. 🗺️ Embodied Scene Memory (具身场景记忆)

引入 **`SceneMemoryEngine`**，使机器人能够“记住”它去过的地方。

  * **Topological Localization:** 识别环境指纹，实现场景再认（Relocalization）。
  * **Embodied Metrics:** 机器人用“身体”丈量世界。它知道环境的通道宽度是否允许自己通过（`clearance_ratio`），并根据电量（`battery_level`）和功耗估算剩余行走里程。
  * **Persistent Mapping:** 自动将关键实体锚定在场景记忆中（例如：“充电桩在客厅”）。

### 3\. ⚡ Async Executive with RAG (异步 RAG 规划器)

**`MultiTaskExecutive`** 实现了非阻塞的异步规划。

  * **Experience RAG:** 利用 **Strategy Optimizer**，在规划前检索类似的历史成功策略，实现经验进化。
  * **Persistent Cognition:** 在没有用户指令时，Agent 不会发呆，而是运行低优先级的“常驻认知任务”，主动扫描环境、优化工作记忆。

### 4\. ⚖️ Competitive Behavior Arbitration (竞价式行为仲裁)

摒弃传统的有限状态机，采用 **`BehaviorArbiter`** 进行基于优先级的行为竞价。

  * **Safety First:** 安全反射（System 1）拥有最高优先级（P=100.0）。
  * **User Command:** 用户指令驱动 Executive 规划（P=5.0）。
  * **Curiosity & Maintenance:** 环境认知与探索作为背景任务（P=1.5）。

-----

## 🏗️ Architecture Overview (架构概览)

Titan-AGI 采用双流（Dual-Stream）混合架构，通过 **World Model** 桥接感知与决策，并使用 **Behavior Arbiter** 统一输出接口。

```mermaid
graph TD
    %% --- 样式定义 ---
    classDef hardware fill:#333,stroke:#000,stroke-width:2px,color:#fff;
    classDef logic fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef memory fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef decision fill:#fce4ec,stroke:#c2185b,stroke-width:2px;

    %% --- 硬件输入 ---
    Sensors(Sensors<br/>Camera / Lidar / Body Status) ::: hardware --> Perception[Perception System] ::: logic

    %% --- 核心：世界模型 ---
    subgraph WorldModel [World Model & Cognition]
        direction TB
        Perception --> |Visual Detections| ObjEng[Object Cognition Engine] ::: memory
        Perception --> |Env Metrics| SceneEng[Scene Memory Engine] ::: memory
        
        ObjEng -- Entities + ID --> Context((Fused Context))
        SceneEng -- Scene Node + Battery --> Context
    end

    %% --- System 1: 快速反应 ---
    subgraph Sys1 [System 1: Fast & Reactive]
        Context --> Safety[Safety Reflex] ::: logic
        Safety --> |Proposal P=100.0| Arbiter
    end

    %% --- System 2: 慢速规划 ---
    subgraph Sys2 [System 2: Slow & Deliberative]
        Context --> Executive[MultiTask Executive] ::: logic
        
        %% 记忆与 RAG
        Stream[(Cognitive Stream)] ::: memory <--> Executive
        Strategy[(Strategy Optimizer)] ::: memory -.-> |RAG: Learned Policy| Executive
        
        Executive --> |Proposal P=5.0/1.5| Arbiter
    end

    %% --- 仲裁与执行 ---
    Arbiter{Behavior Arbiter} ::: decision
    
    Arbiter --> |Winner-Take-All| ActionMgr[Action Manager] ::: logic
    ActionMgr --> Robot(Robot Hardware) ::: hardware

    %% --- 隐式闭环 ---
    Robot -.-> |Physical Feedback| Sensors
```

### 架构解析

1.  **Sensors & Perception**: 原始数据被处理为 `VisualDetection`（视觉）和 `EnvironmentMetrics`（具身度量）。
2.  **World Model (Memory)**:
      * **Object Cognition**: 负责物体恒常性（ID Tracking）和属性注入。
      * **Scene Memory**: 负责环境拓扑识别和自我状态（电量/尺寸）评估。
3.  **System 1 (Reflex)**: 绕过复杂计算，直接基于当前帧产生高优先级的安全反射（如防碰撞）。
4.  **System 2 (Executive)**:
      * 处理用户指令（Task）或主动环境维护（Cognition）。
      * **RAG Loop**: 向 `Strategy Optimizer` 检索历史策略。
      * **Working Memory**: 读写 `Cognitive Stream`。
5.  **Behavior Arbiter**: 接收所有子系统的 `ActionProposal`，根据优先级（Priority）和置信度选择唯一的动作交给硬件执行。

-----

## 📂 Core Modules (核心模块说明)

### `include/titan/cognition/`

  * **`object_cognition.h`**: 核心认知引擎。处理物体追踪、卡尔曼滤波速度估计和语义属性注入。
  * **`scene_memory.h`**: 场景记忆引擎。负责环境识别、具身度量计算（电量/宽度）和拓扑地图构建。

### `include/titan/agent/`

  * **`multi_task_executive.h`**: 系统的大脑。
      * 管理任务队列 (`ActiveTask`)。
      * 集成 `injectStrategyOptimizer` 进行 RAG 增强规划。
      * 运行 `getCognitionProposal` 进行常驻环境维护。
  * **`behavior_arbiter.h`**: 系统的裁判。接收来自 Executive、Safety 和 Curiosity 的 `ActionProposal`，执行赢家通吃（Winner-Take-All）逻辑。

### `include/titan/core/`

  * **`types.h`**: 定义了通用的数据结构。
      * `WorldEntity`: 包含物理状态（位置/速度）和认知图谱的实体。
      * `EnvironmentMetrics`: 描述环境相对于机器人身体的属性（`clearance_ratio`, `estimated_runtime`）。
      * `ActionProposal`: 包含优先级、描述和执行闭包的原子行为单元。

-----

## 🛠️ Build & Integration (构建与集成)

Titan-AGI 是一个纯 C++17 Header-Only 核心库（依赖 OpenCV, Eigen, nlohmann\_json）。

### Dependencies

  * **C++17** Compiler
  * **Eigen3** (Matrix & Vector math)
  * **OpenCV 4.x** (Vision processing)
  * **nlohmann\_json** (Serialization)

### Example Usage (Pseudo-code)

```cpp
// 1. Initialize Modules
TitanAgentImpl agent;
agent.multi_executive_.injectStrategyOptimizer(&optimizer);
agent.multi_executive_.injectSceneMemory(&scene_memory);

// 2. Main Loop (e.g., 100Hz)
while (running) {
    // Perception -> Cognition -> Planning -> Action
    agent.tick(); 
    
    // The agent will automatically:
    // - Track objects (ObjectCognition)
    // - Measure environment width/battery (SceneMemory)
    // - Execute user tasks OR explore (Executive)
}
```

-----

## 🔮 Roadmap (未来规划)

1.  **VLA (Vision-Language-Action) Model Integration:** 将目前的 YOLO+LLM 管道替换为端到端的 VLA 模型（如 RT-2 或 OpenVLA）接口。
2.  **Episodic Memory Replay:** 在夜间充电模式下，对白天的 `CognitiveStream` 进行回放和强化学习训练。
3.  **Sim2Real Pipeline:** 提供基于 Isaac Sim 的标准仿真环境接口。

-----

## 📄 License

MIT License. See `LICENSE` for more information.