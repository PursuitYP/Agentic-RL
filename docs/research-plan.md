# 研究计划

> 状态：前期规划
> 核心问题：结构化 KG 环境中的 SFT cold-start 与 Curriculum RL，能否同时提升 KGQA、跨 KG 泛化和可迁移的 Agentic 推理能力？

## 1. 目标

将知识图谱从静态知识源转化为可交互、可执行、可验证的 **Structured Agentic Environment**。模型通过多轮工具交互完成关系选择、实体探索、证据整合、回溯纠错和答案生成，并采用：

1. **SFT cold-start**：用少量、多样、经环境验证的轨迹建立合法交互与基础探索能力；
2. **Curriculum RL**：逐步训练检索、规划、约束处理、纠错和长程决策；
3. **分层评测**：分别验证 KGQA、跨 KG、通用推理与跨环境 Agentic transfer。

## 2. 研究动机

知识图谱适合作为 Agentic RL 环境，因为它同时具备：

- 明确的实体、关系、状态转移和可执行查询；
- 可控制的跳数、分支、约束、噪声与信息缺失；
- 可自动验证的动作、路径和最终答案；
- 可替换的 KG、schema 和工具接口；
- 可规模化生成的组合任务。

直接从基础模型进行在线 KG-RL 往往面临格式错误、无效调用、长程稀疏奖励和探索失败。SFT cold-start 用于建立可探索的初始策略，RL 用于突破轨迹模仿的上限；curriculum 则用于控制训练难度和奖励稀疏度。

KGQA 提升并不自动代表通用推理提升。模型可能只记住实体、关系语义或工具格式，因此能力迁移必须通过独立任务与接口控制实验验证。

## 3. 研究问题

- **RQ1 — KGQA**：SFT cold-start + Curriculum RL 是否优于纯 SFT 和无课程 RL？
- **RQ2 — Agent 行为**：增益是否来自更好的探索、证据利用、回溯和停止策略？
- **RQ3 — 跨 KG**：策略能否迁移到未见实体、关系组合、拓扑和 schema？
- **RQ4 — 通用推理**：结构化交互训练能否迁移到非 KG 的逻辑、数学或多跳推理？
- **RQ5 — 跨环境**：改变观察形式、工具名称或动作接口后，Agentic 能力能否保留？
- **RQ6 — Curriculum**：easy-to-hard、reverse 和基于策略成功率的课程分别适用于什么条件？

## 4. 方法方案

### 4.1 Environment

首版环境保持最小：

- `get_relations(entity)`：获取候选关系；
- `get_triples(entity, relations)`：获取选定关系对应的三元组；
- `answer(answer_set)`：提交最终答案。

状态包含问题、历史动作、已检索证据和剩余预算；观察结果经过裁剪与去重；输出答案或达到调用/token 上限时终止。首版不引入复杂规划器、多 Agent 或 MCTS。

### 4.2 Data and SFT cold-start

训练以 WebQSP、CWQ 和 Freebase 为起点。轨迹来源包括 gold path 转换、强模型 rollout 和成功轨迹拒绝采样，并满足：

- 工具调用可执行；
- 最终答案正确且有环境证据支持；
- 覆盖不同 hop、约束、分支、空结果、回溯和多答案问题；
- 环境 observation 不参与目标 token loss。

SFT 的目标是学习协议与基础策略，不是记忆唯一 gold path。

### 4.3 Curriculum RL

按复杂度逐步实现：

1. **无课程基线**：均匀采样；
2. **静态课程**：按 hop、问题类型或约束难度分阶段；
3. **反向课程**：优先复杂样本，检验是否能减少简单任务 shortcut；
4. **经验课程**：依据当前策略多次 rollout 的成功率，优先采样“偶尔成功但尚未掌握”的样本；
5. **能力课程**：在前述方案稳定后，按探索、约束、回溯、停止等技能动态调整分布。

阶段切换时保留已学样本回放，监控遗忘和 reward 分布突变。

### 4.4 Reward

首版使用可解释的最小奖励：

\[
R = R_{answer} + \lambda_f R_{format} - \lambda_c R_{cost}
\]

- `R_answer`：答案集合 F1，并独立报告 EM/Hit；
- `R_format`：工具调用和最终输出是否合法；
- `R_cost`：无效、重复调用与过长轨迹的轻量惩罚。

path match、graph distance 和 retrieval progress 作为后续对照。任何依赖 gold path 或答案实体的过程奖励都必须检查泄漏、reward hacking 和跨 KG 可用性。

## 5. 评测与证据边界

| 层次 | 目标 | 候选评测 |
| --- | --- | --- |
| KGQA in-domain | 基本任务效果 | WebQSP、CWQ |
| 结构与跨 KG | 组合与 schema 泛化 | GrailQA、未见拓扑、Wikidata/TKG |
| 通用推理 | 非 KG reasoning transfer | 逻辑、数学、多跳文本任务 |
| 跨 Agent 环境 | observation/action interface 迁移 | 一个成本可控的异构工具环境 |

关键对照包括 Base、SFT、SFT + uniform RL、静态 curriculum、reverse curriculum 和经验 curriculum。进一步使用关系匿名化、工具改名和输出格式替换，区分推理迁移与语义/格式记忆。

所有结论按证据强度表述：域内提升、跨 KG 提升和跨环境迁移是三种不同主张，不能相互替代。

## 6. 最小可行实验

第一阶段只回答：**在现有 Freebase 在线环境中，Curriculum RL 能否比 SFT + uniform RL 更稳定地提升复杂 KGQA？**

- 模型：默认 `/mnt/shared-storage-user/wenxiaoyu/models/Qwen2.5-7B-Instruct`；必要时使用小模型做快速调试；
- 数据：WebQSP + CWQ；
- 环境：Virtuoso + 最小工具集；
- 训练：SFT cold-start 后采用现有 veRL/GRPO 路线；
- 对照：Base、SFT、uniform RL、静态 curriculum、经验 curriculum；
- 指标：论文数据集官方答案指标，以及调用数、无效调用、token、训练稳定性；
- 重复：在资源允许范围内使用多 seed，并如实报告限制。

该阶段成立后，再开展跨 KG、通用推理和跨环境迁移实验。

## 7. 当前边界

- cold-start、GRPO、curriculum 和过程奖励均已有相关工作，现阶段不将单一组件视为创新点；
- 不预设 curriculum 一定优于均匀训练，也不预设 easy-to-hard 一定优于 reverse curriculum；
- 不把 KGQA 或跨 KG 提升直接解释为通用 Agentic reasoning；
- 不在首版同时引入 PRM、MCTS、自博弈、多 Agent 和多环境联合训练；
- 先跑通相关项目的核心流程和代表性实验，从中提炼经验，再依据证据收敛最终方案。

相关工作的逐篇复现方法见 [复现计划](reproduction-plan.md)，当前执行状态见 [任务清单](tasks.md)。
