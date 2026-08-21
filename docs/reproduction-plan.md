# 相关项目复现计划

> 目标：跑通 `related-projects/` 中 12 个项目的核心流程，每个项目完成 1–2 个重要实验，提炼可复用的环境、轨迹、SFT、RL、reward 与 curriculum 经验。

## 1. 复现原则

本计划基于仓库内 12 篇论文、对应官方代码及训练脚本综合整理。需要区分三类事实：

- **论文设定**：论文报告的方法、资源和结果；
- **代码现状**：当前固定 commit 中实际提供的入口与限制；
- **本项目复现**：在离线 GPU 节点上跑通其环境、训练、推理和评测闭环，并验证最有价值的核心机制。

本轮复现不以完整还原论文所有数值为目标。每个项目采用分级验收：

| 级别 | 验收内容 |
| --- | --- |
| L0 资料核验 | 固定论文、commit、依赖、数据来源、配置和入口脚本 |
| L1 服务与推理 | 官方样例或小数据端到端运行，输出可被 evaluator 解析 |
| L2 训练闭环 | 小数据、小步数完成 rollout、更新、checkpoint、推理与评测 |
| L3 关键实验 | 完成预先选定的 1–2 个代表性实验并总结经验 |

默认完成到 L3 即可。允许根据资源缩小数据、steps、模型或评测范围；结果只需与论文报告的趋势进行合理核对，不设置固定数值容差。完整主表、全部数据集、穷举消融和多 seed 均非必需；若核心流程未跑通或结果完全反常，再增加诊断实验。

## 2. 复现准备与边界

### 2.1 固定版本与离线资产

论文原始设定用于理解方法和选择关键实验，实际运行以可获得资源和跑通流程为优先：

- 论文指定且本地已有的 backbone 可以直接使用；否则训练与本地推理默认使用 `/mnt/shared-storage-user/wenxiaoyu/models/Qwen2.5-7B-Instruct`。
- 需要 LLM API（如轨迹生成、relation filtering、prompting baseline 或 LLM-as-judge）时使用 `gpt-5`，并记录与论文原模型的差异。
- 原始设定与替代设定的配置、结果和结论分开记录，不混合比较。

| 项目 | 当前 commit | 主要框架 |
| --- | --- | --- |
| Search-R1 | `598e61b` | veRL + vLLM |
| Logic-RL | `9d2c457` | veRL + vLLM |
| ToG | `7ccbb92` | LLM API/local LLM + SPARQL |
| PoG | `6d1627e` | LLM API + Freebase |
| RoG | `ccf8ec8` | Transformers/DeepSpeed |
| KnowCoder-A1 | `9f8881d` | LLaMA-Factory + veRL |
| GraphWalker | `1d8860c` | LLaMA-Factory + slime |
| SIE | `3d716c7` | veRL + vLLM |
| ISP-KGR | `dddb150` | slime + SGLang/Megatron |
| KG-R1 | `dc5e3d9` | veRL + vLLM |
| EoG | `5e2355d` | veRL/DAPO + vLLM |
| Temp-R1 | `44ab58f` | LLaMA-Factory + veRL |

GPU 节点不联网。联网节点需提前准备：模型与 tokenizer 快照、数据集、Freebase/Wikidata/TKG 文件、检索索引、容器镜像、conda/pip 包及 CUDA wheel。`flash-attn` 等二进制包必须按 Python、PyTorch、CUDA、架构和 C++ ABI 从官方 release 选择。所有资产记录 URL、版本、大小和 SHA256。

新增数据、项目环境和复现资产统一存放在 `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/yupeng/agentic-rl/`：

```text
agentic-rl/
├── data/<project>/{raw,processed,indexes}
├── envs/<project>
├── deps/{wheels,source,images}
├── services/<project>
└── runs/<project>/<experiment>
```

该规范只约束存放位置，不统一不同论文的数据格式、软件环境或评测方式。现有共享模型继续使用其原路径，不重复复制。具体路径通过 `.env` 读取。

### 2.2 按项目准备环境与服务

各项目独立建立环境、数据目录、服务和配置，不强制共享接口或转换轨迹格式。需要准备的主要服务包括：

1. **Freebase/Virtuoso**：按 ToG、PoG、KnowCoder-A1、GraphWalker、ISP-KGR 各自要求加载对应 dump、schema、alias 和索引。
2. **子图/KG API**：按 KG-R1 的数据预处理和 `get_*` 操作部署其专用后端。
3. **文本/TKG retriever**：为 Search-R1 和 Temp-R1 建立 E5/FAISS 索引，服务与训练进程分离并固定 top-k。

即使多个项目都使用 Freebase，也不能默认共用同一预处理结果：必须核对论文采用的是全局 KG、Virtuoso、预抽取子图还是项目自定义 API。每个服务分别完成官方样例、固定查询、超时和空结果测试。禁止训练时访问公共 SPARQL 或在线搜索 API。

### 2.3 按论文执行评测

- 每个项目从论文主结果和核心贡献中选择 1 个主实验，必要时再选择 1 个最关键的机制对照；不要求覆盖其他数据集和小型消融。
- 选定实验尽量使用官方数据处理、答案规范化和 evaluator；EM、Hits@1、RHits@1、F1、LLM-as-judge 等指标不可互相替代。
- 官方 evaluator 用于确认流程和结果方向。为总结经验可额外记录 reward、KL、响应长度、工具调用和延迟。
- 如发现论文、README 和 evaluator 定义不一致，三者分别记录，以论文主表采用的口径为准，并说明判断依据。
- 默认允许单次运行；只有结果波动明显或无法判断核心机制时，才追加 seed 或重复实验。

## 3. 分阶段复现顺序

| 阶段 | 项目 | 目的 |
| --- | --- | --- |
| A：推理基线 | ToG → PoG → RoG | 建立图搜索、规划、自纠错和可解释路径基线 |
| B：RL 基座 | Logic-RL → Search-R1 → SIE | 验证 rule-based RL、工具交互 RL 与结构化环境迁移 |
| C：KG Agent RL | KG-R1 → KnowCoder-A1 → ISP-KGR | 验证多轮 KG 环境、SFT cold-start、turn/process reward |
| D：课程与奖励 | GraphWalker → EoG → Temp-R1 | 验证轨迹课程、path reward 和正/逆 curriculum |

阶段顺序表达依赖关系，不代表只复现后期工作。A/B 阶段提供故障定位基线，C/D 阶段提供本项目核心训练组件。

### 3.1 关键实验选择

| 项目 | 必做实验 | 可选第二实验 |
| --- | --- | --- |
| ToG | ToG vs 无 KG CoT | ToG-R 效率对照 |
| PoG | PoG vs ToG | 成功/失败回退案例分析 |
| RoG | 官方 checkpoint 主流程 | 小规模训练闭环 |
| Logic-RL | RL 前后 K&K 表现 | mixed vs curriculum |
| Search-R1 | 训练前/RAG vs Search-R1 | retrieved-token masking |
| SIE | RL 前后结构化推理 | 一个通用推理迁移集 |
| KG-R1 | 训练前 vs KG-R1 | 一个 cross-KG 目标集 |
| KnowCoder-A1 | SFT vs curriculum RL | — |
| ISP-KGR | curriculum GRPO 主流程 | dense vs outcome-only reward |
| GraphWalker | 两阶段 SFT vs RL | — |
| EoG | outcome-only vs path-refined RL | — |
| Temp-R1 | reverse vs no curriculum | 一个 OOD 推理集 |

## 4. 各项目复现方案

### 4.1 ToG：训练外的动态 KG 探索基线

**方法与代码。** ToG 让 LLM 在 KG 上执行 beam search：定位 topic entity，按“关系搜索/裁剪—实体搜索/裁剪—充分性判断”循环，达到最大深度后生成答案。论文使用 Freebase 和 Wikidata，覆盖 CWQ、WebQSP、GrailQA、QALD10-en、SimpleQuestions、WebQuestions、T-REx、Zero-Shot RE 和 Creak；默认 beam width/depth 均为 3。仓库提供 Freebase/Wikidata 部署、主推理和 EM 评测代码，不包含训练。

**复现步骤。**

1. 优先部署 Freebase Virtuoso，运行仓库固定查询并核对关系方向；Wikidata 仅在需要第二类 KG 经验时再部署。
2. 在 CWQ 或 WebQSP 的代表性样本上跑通 topic entity、relation prune、entity prune、early stop 和官方评测。
3. 保持论文 prompt、搜索参数和调用流程，LLM API 使用 `gpt-5`；由于模型不同，该结果用于复现 ToG 机制，不作为论文原始数值的严格复现。
4. 关键实验选择 **ToG vs 无 KG CoT**；如成本允许，再运行 ToG-R 作为效率对照。无需覆盖其余数据集与搜索参数消融。

**验收。** 一个主要 KGQA 数据集端到端可运行，轨迹与官方指标可解析，并能总结 graph search 相对普通 CoT 的行为和成本差异。

**对本项目的价值。** 提供无需训练的 Structured Agentic Environment 原型和强搜索基线；其显式 search/prune/reason 状态机可用于定义 action space、生成 SFT 轨迹和评估 RL 是否真正超过 prompt-time beam search。局限是 LLM 调用多、依赖强模型裁剪，不能直接证明能力被参数化到小模型中。

### 4.2 PoG：自纠错与自适应规划基线

**方法与代码。** PoG 在 ToG 上加入问题条件分解、memory 更新、充分性评估、reflection、backtracking 和自适应探索，使用 CWQ、WebQSP、GrailQA 与 Freebase。当前仓库是推理型实现，默认示例为 GPT-3.5、深度 4，指标为 EM。

**复现步骤。**

1. 复用 ToG 的 Freebase 和数据快照，跑通 objective decomposition、memory、reflection/backtracking 的状态更新。
2. 保持官方 prompt 和深度设置，在 CWQ 或 WebQSP 上使用 `gpt-5` 完成推理与官方评测。
3. 关键实验选择 **PoG vs ToG**；额外抽取少量成功回退与失败回退案例，理解 self-correction 的实际作用。无需逐项关闭所有模块。

**验收。** PoG 推理链完整可执行，并能从指标与案例解释自适应规划、memory 和 reflection 带来的收益或额外成本。

**对本项目的价值。** PoG 的 objective、memory 和 reflection 可直接转化为 SFT trajectory schema 与 curriculum 技能标签，并为 RL 中的 progress/recovery reward 提供行为定义。它也提供“模型何时回退”的可解释基线，但提示词编排与闭源模型耦合较强。

### 4.3 RoG：规划—检索—推理的 SFT 基线

**方法与代码。** RoG 用 relation path 作为 KG-grounded plan，先生成关系路径，再从 KG 检索有效 reasoning paths 并生成答案；在 WebQSP/CWQ 上训练和评测。仓库提供预训练权重、数据构造、planning、reasoning 与 plug-and-play 脚本；推理需约 12 GB GPU，论文训练脚本标注 2×A100-80GB。

**复现步骤。**

1. 离线下载官方 RoG 权重和 WebQSP 或 CWQ 处理后数据，校验 relation-path 与子图字段。
2. 跑通 planning → path retrieval → reasoning → official evaluation，并确认答案可追溯到检索路径。
3. 重建少量 question–relation path 和 joint-training 数据，完成短程训练及 checkpoint 推理闭环。
4. 关键实验选择 **官方 RoG checkpoint 的主结果**；训练侧只需证明数据构造和优化流程可运行，不要求重训论文规模模型或完成全部消融。

**验收。** 一个数据集的官方模型推理与小规模训练均可运行，并能总结静态 relation-path SFT 的优点和限制。

**对本项目的价值。** RoG 是 SFT cold-start 的关键对照：验证静态 relation-path imitation 能提供多少结构先验，以及它与在线 agent trajectory SFT 的差距。relation path 可作为课程难度、path reward 和解释性评测的低成本监督，但 RoG 不学习环境中的动态纠错。

### 4.4 Logic-RL：可验证推理与课程对照

**方法与代码。** Logic-RL 使用可程序生成和验证的 Knights and Knaves 谜题，格式奖励与答案奖励驱动 RL。论文主设置为 Qwen2.5-7B-Instruct-1M、少于 5,000 个样本、REINFORCE++、3–7 人混合难度、3,600 steps、学习率 `4e-7`、rollout 8、最大响应 4,096；仓库也提供 GRPO/PPO 和 3→7 人 curriculum 脚本，README 主入口为 4×A100-80GB。

**复现步骤。**

1. 重新生成训练所需的 K&K 数据，固定规则、seed 和答案解析器；用格式伪装、重复答案和部分答案做少量 reward 对抗测试。
2. 使用仓库主算法完成一次 RL 训练闭环，比较训练前后 K&K 准确率、reward 和响应长度。
3. 第二个关键实验选择 **mixed difficulty vs 3→7 curriculum**；可用缩小数据与 steps 观察趋势，不再遍历 PPO/GRPO/REINFORCE++ 或全部迁移集。

**验收。** rule-based reward 与训练闭环可用，并能判断课程顺序在当前配置下是否带来明显影响；结论允许为负。

**对本项目的价值。** 提供最干净的可验证 RL 和通用推理迁移对照，可用于调试算法、reward parser 与 curriculum 调度，而不受 KG 服务噪声干扰；同时帮助判断 KGQA 增益究竟来自通用推理还是知识检索。

### 4.5 Search-R1：多轮工具交互 RL 基座

**方法与代码。** Search-R1 在 `<think>/<search>/<information>/<answer>` 循环中训练 LLM 自主查询文本检索器，采用最终答案 outcome reward，并 mask 环境返回 token 的 loss。论文以 NQ+HotpotQA 训练，在 NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle 上评测；使用 Qwen2.5 3B/7B Base/Instruct、E5、Wikipedia 2018、top-3。论文主配置为 8×H100、500 steps、batch 512；代码支持 PPO/GRPO/REINFORCE。

**复现步骤。**

1. 离线准备论文训练所需的 NQ/HotpotQA、Wikipedia 2018、E5 和一种本地索引，验证检索服务。
2. 使用默认模型和仓库可用的 GRPO 或 PPO 配置完成训练—推理—EM 评测闭环。
3. 关键实验选择 **Search-R1 vs 训练前模型/RAG 基线**，记录有效查询、搜索轮数、EM 和服务吞吐。
4. 如需第二个机制实验，只比较 retrieved-token masking on/off；不再遍历算法、group size、轮数、top-k 和所有 backbone。

**验收。** 检索服务、多轮 rollout、RL 更新和官方 evaluator 全部跑通，并观察到可解释的搜索行为变化；训练异常和 collapse 如实记录。

**对本项目的价值。** 它是将普通检索环境替换为 KG 环境的直接工程母版，提供 rollout、state masking、服务解耦和 outcome-only RL 基线；也用于比较文本搜索与结构化 KG 操作对通用 Agentic 能力的影响。

### 4.6 SIE：结构化上下文环境与通用迁移

**方法与代码。** SIE 从 KG 自动构造 structured in-context environments：由问题/答案侧检索 supporting subgraph，加入 distractors，并生成信息完整度逐步下降的 context mode。使用 answer+format rule reward 和 GRPO。代码以 Qwen2.5-7B-Instruct、8 GPU、context mode 0–6、最多 600 steps 为示例，同时评测 WebQSP、CWQ、GrailQA、GSM8K、MATH-500 和 K&K。

**复现步骤。**

1. 下载并抽查官方 SIE 数据，理解 support、distractor 和 context mode 的构造，检查明显答案泄漏。
2. 选择一个代表性 context mode，使用默认 7B 模型完成 GRPO 训练与 KGQA 评测闭环。
3. 关键实验选择 **训练前后结构化推理表现**；第二个实验只选一个通用推理集，观察是否存在迁移。
4. 如需理解 partial environment，再追加一个完整/部分 context 对照；无需遍历 mode 0–6、全部 reward 和 RL 算法。

**验收。** SIE 数据、GRPO 和两个选定评测可运行，并能区分结构化任务增益与通用推理迁移；没有迁移也属于有效经验。

**对本项目的价值。** SIE 与项目目标最直接对应：给出可规模化环境构造、可控信息量和通用迁移评测框架。其不足是环境主要作为单轮上下文，而非完整的多轮工具 MDP；我们可将 context difficulty 转化为交互式 curriculum。

### 4.7 KG-R1：多轮 KG-RL 与跨 KG 基线

**方法与代码。** KG-R1 以单 agent 交替短推理与四类 KG 操作，结合 turn-level 格式/查询奖励和 global F1/检索覆盖奖励，用 multi-turn GRPO 学习策略。论文以 Qwen2.5-3B/7B-Instruct 在 WebQSP/CWQ 训练，并通过替换 KG 后端零样本迁移到 SimpleQA、GrailQA、T-REx、QALD-10en 和 MultiTQ。代码主脚本为 4 GPU、3B、7 turns、400 steps、每题 16 rollouts。

**复现步骤。**

1. 运行 `initialize.py` 的等价离线流程，固定 CWQ/WebQSP 子图、初始实体和 KG server。
2. 使用默认模型完成一个数据集（优先 CWQ）的多轮 GRPO 训练、checkpoint 推理和 F1/Hit@1 评测。
3. 关键实验选择 **训练前 vs KG-R1 训练后**，同时记录 KG action、轮数和错误类型。
4. 第二个实验只选一个目标数据集进行 backend 替换和 zero-shot cross-KG；无需覆盖另一个训练集、五个迁移集及全部 reward/format 消融。

**验收。** 一个 KGQA 训练—评测闭环与一个跨 KG 推理闭环可运行，并能总结多轮 turn reward、KG server 和 schema 迁移的工程经验。

**对本项目的价值。** 提供最直接的多轮 Structured Agentic KG Environment、turn credit assignment 和跨 KG 评测基线，是本项目后续环境设计的主要参考。其弱点是从 Instruct 模型直接 RL，缺少系统性的 SFT cold-start 与任务 curriculum。

### 4.8 KnowCoder-A1：SFT cold-start 与 reward curriculum

**方法与代码。** KnowCoder-A1 先在 WebQSP、CWQ、GrailQA 的高质量多轮轨迹上 SFT，再用 GRPO 和 outcome supervision 探索。工具包括 `SearchGraphPatterns`、`SearchTypes`、`ExecuteSPARQL`。reward 由 format 与答案集合 Fβ 构成，课程从偏 precision 的 `β=0.5` 过渡到平衡的 `β=1.0`，抑制输出超大候选集合的 reward hacking。仓库训练示例为 8 GPU、rollout repeat 8、最多 10 turns。

**复现步骤。**

1. 部署 Virtuoso/API DB server，逐个验证三类工具和 SPARQL 执行安全边界。
2. 抽查 SFT 轨迹并用默认模型完成 cold-start，确认多轮格式、工具调用和 checkpoint 推理正常。
3. 在一个主要数据集上依次运行 `F0.5` 与 `F1` 阶段，验证 resume、reward 切换和完整 RL 闭环。
4. 关键实验选择 **SFT checkpoint vs curriculum RL 最终 checkpoint**；不再训练 fixed F0.5、fixed F1、direct RL 等全部对照。

**验收。** SFT→F0.5→F1 三阶段可连续训练和恢复，并在一个数据集上观察课程前后的答案与轨迹变化。

**对本项目的价值。** 这是本项目“SFT cold-start + Curriculum RL”最直接的参考实现，尤其适合复用轨迹格式、工具协议和 reward strictness curriculum。需要警惕：课程主要改变答案评分严格度，并不等价于环境或推理深度课程。

### 4.9 ISP-KGR：交互式语义解析与稠密过程奖励

**方法与代码。** ISP-KGR 将逻辑形式拆成逐步可执行 action，用 K-beam tree rollout 探索；每节点 reward 为 `0.1×format + 0.3×progress + 0.6×outcome`，progress 由当前实体到答案实体的图距离衡量。论文使用 Qwen2.5-3B-Instruct、WebQSP/CWQ、16 samples，并按 easy→medium→hard 训练。当前主分支是 slime 插件版，示例使用 4 GPU、K=6、最多 6 次交互、DAPO 式 asymmetric clipping，且明确尚未支持 evaluation rollout。

**复现步骤。**

1. 部署 Virtuoso、embedding servers 和 KG query/distance server，先验证 `/kg_query`、`/entity_distance`、health 与并发。
2. 用人工小图验证 distance reward 的单调性、不可达节点、别名实体和最大距离截断。
3. 运行 tree rollout，检查 candidate dedup、group resize、zero-variance filter、max aggregation 和 advantage clamp。
4. 先实现或恢复可用的 evaluation rollout，再在 WebQSP 或 CWQ 上完成一次 curriculum GRPO 训练—评测闭环。
5. 第二个关键实验仅比较 **dense distance reward vs outcome-only reward**；不再遍历 linear/tree、所有课程和跨项目对照。

**验收。** reward 测试、tree trace、训练与独立 eval 均可运行，并能说明稠密 progress reward 对训练信号的影响。当前插件版与论文版差异需记录。

**对本项目的价值。** 提供比最终答案更细的拓扑 progress signal、树状探索和 executable semantic parsing，可用于解决长程 KGQA 的 reward sparsity。风险是图距离可能奖励“靠近答案但语义错误”的路径，并依赖答案实体，迁移到无标注环境时不可直接使用。

### 4.10 GraphWalker：合成轨迹与阶段式 SFT 课程

**方法与代码。** GraphWalker 直接访问全局 Freebase，工具为 `get_relations` 和 `get_triples`。GraphSynth-15k 通过受约束随机游走生成 2–5 hop composition/conjunction 结构；GraphRoll-6k 提供 reflection/backtracking 轨迹。训练按 GraphSynth SFT → GraphRoll SFT → sparse EM GRPO 进行，并将 `<information>` 环境 token 从 SFT loss 中 mask。论文仅在 CWQ 训练，评测 CWQ、WebQSP 和未见结构的 GraphWalkerBench。

**复现步骤。**

1. 构建全局 Freebase 服务，验证 relation branching、方向、alias 与 observation truncation。
2. 优先使用发布数据；如需验证生成管线，仅生成小批 GraphSynth/GraphRoll，API 使用 `gpt-5` 并记录模型差异。
3. 使用默认模型跑通 GraphSynth SFT → GraphRoll SFT → GRPO，验证 `<information>` masking 和 checkpoint 继承。
4. 关键实验选择 **两阶段 SFT checkpoint vs RL 最终 checkpoint**，在 CWQ 或 WebQSP 上评测；无需重建完整 21k 轨迹或完成所有 stage/order/data-size 消融。

**验收。** 小规模数据构造或发布数据可用，三阶段训练—评测闭环完整，并能总结结构预训练、反思轨迹和稀疏 RL 的衔接经验。

**对本项目的价值。** GraphWalker 给出最完整的“结构课程 → 反思课程 → RL”方案，直接支撑 SFT cold-start 和结构泛化目标。其核心可复用产物是拓扑难度生成器、grounding 过滤器和 error-recovery 轨迹；代价是数据生成依赖 Gemini/GPT API，且论文效果需验证是否对生成模型和全局 KG 配置敏感。

### 4.11 EoG：两阶段 RL 与 path-refined reward

**方法与代码。** EoG 先用长 CoT 做 SFT，再分两阶段 RL：先用最终答案 outcome reward 扩展探索，随后加入与 gold reasoning path 匹配比例相关的 path reward，联合优化正确性与语义路径质量。论文在 CWQ、WebQSP、GrailQA、QALD10-en、2WikiMultiHop 上评测 Qwen2.5-7B-Instruct 和 Llama-3.1-8B-Instruct；SFT 轨迹由 Gemini-2.5-Flash 生成。主 RL 使用 8×H100、长达 15k prompt/10k response，成本较高。

**复现步骤。**

1. 审计 SFT 数据、gold paths 和 `reward_func.py` 的答案解析、triplet 抽取、F1/Hit@1 与异常输入；若需重新生成 SFT 轨迹，使用 `gpt-5` 并标注与论文 Gemini-2.5-Flash 数据的差异。
2. 用人工轨迹验证 path reward：正确顺序、逆向关系、别名、冗余正确 triplet、部分路径和虚构 triplet。
3. 在一个数据集（优先 CWQ）上跑通 SFT → outcome RL → path-refined RL，允许缩短上下文、数据和 steps。
4. 关键实验选择 **outcome-only checkpoint vs path-refined checkpoint**；无需覆盖其余数据集、双 backbone 和其他 reward 权重消融。

**验收。** 三阶段流程与 reward 计算可运行，并能从少量指标和轨迹判断 path reward 是否改变 grounded exploration。仓库占位路径需在记录中说明。

**对本项目的价值。** 提供 trajectory-level 之外的结构过程监督，可与 ISP-KGR 的 distance reward 形成“语义路径 vs 拓扑距离”对照。它适合研究 reward curriculum，但依赖 gold paths，且规则 triplet 抽取可能被格式或同义表达影响。

### 4.12 Temp-R1：时序 KG 与逆课程 RL

**方法与代码。** Temp-R1 在 temporal KG 上扩展 action space，将内部 temporal reasoning actions 与外部检索解耦；先用约 1,000 条高质量轨迹 SFT，再以 GRPO 进行 **hard-first reverse curriculum**，防止模型先在简单问题上形成 shortcut。论文以 Llama-3.1-8B-Instruct 为主，使用 MultiTQ、TimelineKGQA-CRON 训练，并在 ICEWS-ACTOR 上 OOD；SFT 2 epochs、学习率 `2e-5`、batch 16，RL 使用 3×A800、group 5、actor lr `5e-7`，仅使用约 9% MultiTQ 训练集。

**复现步骤。**

1. 构建 TKG 四元组数据和 E5/FAISS 索引，校验时间标准化、interval、before/after、排序和多答案。
2. 审计 qtype 分类、过采样和 SFT 轨迹生成；确保 `<information>` token 在 SFT 中 mask。
3. 复现 SFT cold-start，再依次运行复杂样本 stage 1 与全量/较简单样本 stage 2，验证 checkpoint 继承。
4. 关键实验选择 **reverse curriculum vs 无 curriculum**，优先在 MultiTQ 上使用相同缩小预算比较；如资源允许，再在 ICEWS-ACTOR 做一次 OOD 推理。
5. 无需运行 easy-first、mixed、无 SFT、无 internal actions、全部 backbone 和所有数据集分类结果。

**验收。** SFT 与 reverse curriculum 两阶段训练可恢复，MultiTQ 官方评测可运行，并能判断 hard-first 是否影响复杂问题学习。

**对本项目的价值。** Temp-R1 是检验 curriculum 方向的关键反例：难度不一定应从易到难。时序 KG 也提供强跨环境测试，可判断普通 KG 上学到的检索与规划策略能否迁移到包含时间约束的新 action space。

## 5. 复现后的综合分析

跨项目比较不属于跑通任务的验收条件。完成各项目的核心流程和关键实验后，先基于论文结论、本地结果和实现差异做定性归纳，重点比较：

- SFT cold-start 使用 relation path、完整 agent trajectory 或 synthetic trajectory 的差异；
- curriculum 调度的是样本难度、reward 严格度、技能阶段还是训练顺序；
- reward 依赖最终答案、gold path、答案实体距离还是环境交互合法性；
- 方法验证的是域内 KGQA、跨 KG、未见结构、通用推理还是时序 KG 迁移。

只有后续确实需要验证本项目假设时，才另行设计受控对比；不在本轮复现中提前扩张实验范围。

## 6. 里程碑与产物

| 里程碑 | 完成条件 | 主要产物 |
| --- | --- | --- |
| M0 复现准备 | 共享目录就绪，当前项目的离线资产和服务通过 smoke test | asset manifest、环境与服务记录 |
| M1 推理基线 | ToG/PoG/RoG 各达到 L3 | 三套可运行流程与经验记录 |
| M2 RL 基座 | Logic-RL/Search-R1/SIE 各达到 L3 | 可运行训练模板与关键实验结果 |
| M3 KG-RL | KG-R1/KnowCoder/ISP-KGR 各达到 L3 | 多轮 KG 环境与 reward 经验 |
| M4 课程机制 | GraphWalker/EoG/Temp-R1 各达到 L3 | SFT、课程和过程奖励经验 |
| M5 综合分析 | 完成跨论文差异、可复用组件和局限梳理 | 项目方法选择与研究方案 |

每个项目单独建立 `docs/reproductions/<project>.md`，持续记录环境、数据、命令、checkpoint、结果、偏差和下一步；总览文档只维护状态与结论。训练日志、数据、模型、wheel 和缓存放在 Git 之外。

## 7. 风险与决策点

1. **API 模型变化**：ToG、PoG、GraphWalker、EoG 的原闭源模型可能不可用。使用 `gpt-5` 跑通流程，并记录与论文设定的差异，不追求原始数值一致。
2. **全局 KG 与预抽取子图不可比**：明确标注 backend scope，按各论文原环境复现，不直接横向比较数值。
3. **指标实现不一致**：实体别名、集合 F1、substring Hit 和 LLM-as-judge 会显著影响结果；分别保留并使用各项目官方 evaluator，不强行合并口径。
4. **过程奖励泄漏**：gold path、答案实体距离只能用于训练研究，不得在无标注部署场景中隐式使用。
5. **资源差异**：Search-R1/EoG 原配置使用 8×H100，Temp-R1 使用 3×A800；默认缩小规模验证核心流程，不自动投入论文级训练。
6. **代码成熟度不同**：ISP-KGR 缺 evaluation rollout，EoG 脚本含占位路径，KG-R1 有未就绪权重说明；修复前先记录最小差异补丁。
7. **离线节点**：任何运行时下载、在线 API、公共 SPARQL 或 Docker pull 都应在提交训练前消除。

最终选择本项目方案时，不以单篇论文最高 KGQA 分数为依据，而以四项证据共同决策：环境是否可控、训练是否稳定、能力是否跨 KG 泛化、推理策略是否能迁移到非 KG Agentic 环境。
