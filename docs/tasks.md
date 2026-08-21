# 任务清单

> 更新时间：2026-08-21
> 当前阶段：P0 复现准备（进行中）
> 状态：`待开始`、`进行中`、`已完成`、`阻塞`

## 当前目标

跑通 12 个相关项目的核心流程，各完成 1–2 个代表性实验并提炼可复用经验，为本项目的 Structured Agentic Environment、SFT cold-start、Curriculum RL 和迁移评测提供证据。

## 路线与状态

| 阶段 | 状态 | 范围 | 退出条件 |
| --- | --- | --- | --- |
| P0 复现准备 | 进行中 | 论文、仓库、计划、服务器、存储与离线资产 | 资料与版本已固定；共享目录就绪；首批项目 asset manifest 完整 |
| P1 推理基线 | 待开始 | ToG → PoG → RoG | 各跑通核心流程并完成 1–2 个代表性实验 |
| P2 RL 基座 | 待开始 | Logic-RL → Search-R1 → SIE | 各跑通训练—推理—评测闭环并验证核心机制 |
| P3 KG Agent RL | 待开始 | KG-R1 → KnowCoder-A1 → ISP-KGR | 各跑通多轮 KG 环境、训练与独立评测 |
| P4 课程与奖励 | 待开始 | GraphWalker → EoG → Temp-R1 | 各验证一个关键 SFT、reward 或 curriculum 机制 |
| P5 综合分析 | 待开始 | 汇总 12 个项目 | 明确可复用组件、有效证据、局限和本项目取舍 |
| P6 本项目实验 | 待开始 | 环境、SFT、Curriculum RL 与四层评测 | 完成实现、关键对照和迁移评测；标准由 P5 证据细化 |

P0 中，12 篇论文、官方仓库、固定 commit、研究计划和复现计划已就绪；服务器核验、共享存储初始化及首批离线资产尚未完成。

## 当前行动

1. 依据 `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/yupeng/server-infra/` 核对可用节点、GPU、CUDA、PyTorch、调度方式、配额和共享路径。
2. 建立规范的共享存储目录，核对 `.env` 路径变量，并为 ToG、PoG、RoG 建立离线 asset manifest。
3. 创建 `docs/reproductions/tog.md`，固定论文设定、commit、关键实验和成功标准，完成 L0 核验。
4. 准备并 smoke test ToG 所需的 Freebase/Virtuoso、数据集和模型/API 配置，再完成官方样例与小规模关键对照。
5. 更新 ToG 记录和任务状态；将确认可复用的经验写入 `docs/reproductions/common/`，随后推进 PoG、RoG。

## 记录规则

- 项目事实写入 `docs/reproductions/<project>.md`；跨项目共享经验按主题写入 `docs/reproductions/common/`。
- 每次项目开始、完成、阻塞或优先级变化时更新本页；命令、日志和结果不在本页重复维护。
- 单项目按 L0–L3 分级验收，只做预先选择的 1–2 个关键实验；具体标准见[复现计划](reproduction-plan.md)。
- 未执行内容保持“计划”，未验证判断不得写成结论。
