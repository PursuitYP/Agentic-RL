# Agentic-RL Agent Guide

## Mission

构建面向知识图谱问答（KGQA）的 **Structured Agentic Environment**，通过 **SFT cold-start** 与 **Curriculum RL** 训练可验证的结构化检索、工具交互和多步推理能力；提升 KGQA 与跨知识图谱泛化表现，并评估其向通用推理和跨环境 Agentic 推理的迁移能力。

研究主线：

1. 定义状态、动作、观测、终止条件和奖励明确的 KG 交互环境。
2. 用高质量轨迹进行 SFT cold-start，建立稳定的交互协议。
3. 通过 Curriculum RL 训练检索、规划、纠错和长程推理，并比较 easy-to-hard、reverse 与自适应课程。
4. 分别评测 KGQA、跨 KG、通用推理和 Agentic transfer。

## Priorities

1. 按 `docs/tasks.md` 和 `docs/reproduction-plan.md` 跑通 12 个相关项目的核心流程。
2. 每个项目参考论文与官方代码自己的环境、数据、模型、流程和 evaluator，选择 1–2 个关键实验；不要求复现完整主表或全部消融。
3. 完成复现后归纳可复用的环境、轨迹、SFT、RL、reward 和 curriculum 设计。
4. 依据复现证据建立本项目的 SFT、Curriculum RL 与迁移实验。

优先实现可运行、可验证、可比较的系统，不为表面创新偏离主线。

## Repository

- `README.md`：项目概览与当前状态。
- `docs/README.md`：文档导航；`docs/tasks.md`：当前任务状态。
- `docs/`：研究方案、复现计划、设计决策和实验报告。
- `resources/`：研究、复现和实现中发现的可复用参考资料，以及前期笔记与原始材料；其中待验证内容不作为当前结论。
- `related-papers/`：论文 PDF。
- `related-projects/`：以 Git submodule 固定官方仓库 commit 的浅克隆，仅作为参考实现；未经明确要求不要修改。
- `.env`：本地密钥、服务地址和路径，已被 Git 忽略。

项目代码应放在仓库顶层的独立目录中，不要污染第三方仓库。

## Workflow

1. 先用一句话定义成功标准。
2. 阅读 `README.md`、`docs/tasks.md`、相关计划、目标代码、调用方、依赖和周边上下文，理解现有行为后提出方案；不得跳过探索步骤。
3. 对需求、事实、路径、配置、权限或预期结果存在不确定，且无法从现有文档和代码可靠确认时，先向用户提问并等待确认；不得凭猜测修改、执行或记录结论。
4. 检查并保留工作区现有修改。复现前核对论文版本、官方仓库、commit、依赖和数据来源。
5. 采用满足目标的最小实现；避免无关重构、顺带清理、推测性抽象和未要求的配置。
6. 环境接口需显式定义 observation、action、transition、termination 和 reward；数据、提示词、环境、训练与评测逻辑应可独立替换。
7. 随机实验支持固定 seed，并记录模型、数据、配置和代码版本；严禁测试答案、评测反馈或 oracle 信息泄漏。
8. 执行与风险相称的验证。训练先做 smoke test，再启动高成本任务；如实记录失败、退化和异常。

未运行的内容只能标为“计划”或“预期”，不得写成已完成或已验证。

## Code Annotation

新增或修改模块、函数时，在定义上方添加简短的英文注释：

```python
### agentic-rl: <feature description> ###
```

再次修改同一对象时更新原注释，不叠加过期标注。纯文档、格式和配置值修改无需标注。

## Secrets and Offline GPU Nodes

- 运行时按需加载 `.env`：`set -a; source .env; set +a`。密钥不得写入或回显到代码、配置、日志、文档和 Git 历史；缺少变量时向用户索取。
- **GPU 节点不连接互联网**。提交任务前准备并传输代码、模型、数据、tokenizer、配置和完整 Python 依赖，不得依赖在线 API、公共 KG 或运行时下载。
- 服务器、存储和任务运行方式优先参考 `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/yupeng/server-infra/` 下的相关文档；执行前核对当前节点和路径是否与文档一致。
- 本项目新增的数据、环境、离线依赖、服务资产和实验产物统一放在 `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/yupeng/agentic-rl/`，不得散落在仓库、Home 或共享目录顶层。各论文仍使用独立环境和数据处理，不因共用存储根目录而强行合并配置。
- 共享存储采用固定目录：`data/<project>/{raw,processed,indexes}`、`envs/<project>`、`deps/{wheels,source,images}`、`services/<project>`、`runs/<project>/<experiment>`。项目名使用小写短名称，实验目录包含方法、模型和日期等可识别信息。
- 路径优先读取 `.env` 中的 `AGENTIC_RL_STORAGE_ROOT`、`DATA_ROOT`、`ENV_ROOT`、`DEPS_ROOT`、`SERVICE_ROOT` 和 `RUN_ROOT`，脚本中不得重复硬编码共享存储绝对路径。
- 下载二进制依赖前核对操作系统、架构、Python、PyTorch、CUDA 和 C++ ABI。`flash-attn` 优先从 [官方 Releases](https://github.com/Dao-AILab/flash-attention/releases) 获取匹配 wheel；无匹配版本时再准备源码与完整编译工具链。
- 使用本地 wheel 目录离线安装：`pip install --no-index --find-links <wheel-dir> <package>`。W&B 使用 `WANDB_MODE=offline`，回到联网环境后同步。
- 记录依赖来源、版本、校验值和传输方式。wheel、模型、数据、缓存、日志和权重等大型产物不得提交到 Git。
- 持续检查磁盘配额和剩余空间。实验确认可恢复后，及时清理 rollout 缓存、重复导出和无保留价值的中间 checkpoint；默认保留最佳、最终和最近可恢复 checkpoint，并在实验文档记录保留与清理情况。
- 清理前必须列举并核对具体路径、文件类型、数量和占用空间，只处理当前实验的明确产物；不得使用未验证变量、宽泛 glob 或递归删除仓库、共享模型、原始数据及其他用户文件。

## Experiments and Evaluation

每项复现或实验在 `docs/` 中记录：目标、论文与仓库版本、commit、环境与硬件、关键命令、数据划分、模型与 SFT/RL 配置、reward、curriculum、指标、基线、结果、差异、问题和下一步。单项目复现记录放在 `docs/reproductions/<project>.md`；已验证且可跨项目复用的经验按主题放在 `docs/reproductions/common/`，并注明来源、适用前提和验证范围，不以共享总结替代项目执行记录。

相关工作复现以跑通环境、训练、推理和评测闭环并获得可解释经验为目标。关键实验沿用论文与官方代码的 evaluator 和基本设置；允许因资源或模型可用性缩小规模或替换模型，但必须记录差异。默认每个项目只完成 1–2 个代表性实验，不要求完整主表、全部数据集、穷举消融或多 seed；只有结果不稳定或无法判断核心机制时再追加实验。

本项目方法实验应区分：

1. **KGQA in-domain**：准确率、Hits/F1、成功率、交互成本和轨迹合法率。
2. **Cross-KG / OOD**：未见实体、关系、子图、数据集或知识图谱。
3. **General reasoning**：不依赖 KG 工具的逻辑、数学和多步推理。
4. **Agentic transfer**：其他环境中的规划、工具调用、纠错和长程任务完成能力。

本项目的关键消融应覆盖 SFT-only、RL-only、SFT + RL、无 curriculum、不同 reward，以及结构化 KG 环境与文本检索环境的比较。不得用单一 KGQA 指标代替迁移证据。

## Documentation and Completion

- 默认使用中文，保留标准英文术语。先写结论，再写证据与限制；明确区分事实、引用、推断、假设和计划。
- 研究、实现和实验推进时及时更新相关文档：`README.md` 保持稳定总览，`docs/tasks.md` 维护当前状态，其他 `docs/` 文档保留设计、命令、结果和问题，确保工作可总览、可回顾、可理解并可从中断处恢复。
- 复现或实现中发现值得反复查阅的论文、数据、工具、技术文档和实现线索时，整理到 `resources/` 并更新其索引；项目执行事实仍写入单项目记录，经验证的跨项目经验再沉淀到 `docs/reproductions/common/`。
- 研究方向或关键设计变化时，同步更新总览和对应细节文档，避免代码、实验状态与文档不一致。
- 只有在实现或分析完成、必要验证通过、结果与限制已记录，且无密钥泄漏或未说明的大型产物时，任务才算完成。
