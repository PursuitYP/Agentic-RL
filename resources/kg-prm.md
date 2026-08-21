# KG-CoT 与过程奖励：前期研究笔记

# Introduction

基于KG构建CoT思维数据，训练PRM通过RL的方式训练LLM，增强其知识推理能力，例如KGQA、Multi\-Hop QA。



## key words

knowledge graph

question answering

qa

multi\-hop

process reward

reward model

process supervision

process supervised

process verifier

process verify

https://paperswithcode\.com/task/multi\-hop\-question\-answering

https://paperswithcode\.com/task/graph\-question\-answering

https://github\.com/heathersherry/Knowledge\-Graph\-Tutorials\-and\-Papers/blob/master/topics/Knowledge%20Graph%20Question%20Answering%20\(KGQA\)\.md

https://github\.com/heathersherry/Knowledge\-Graph\-Tutorials\-and\-Papers/blob/master/topics/Knowledge%20Graph%20Embedding%2C%20Learning%2C%20Reasoning%2C%20Rule%20Mining%2C%20and%20Path%20Finding\.md

https://github\.com/heathersherry/Knowledge\-Graph\-Tutorials\-and\-Papers/blob/master/topics/Knowledge%20Graph%20and%20LLMs\.md



## 构建CoT数据

Generative PRM

基于Multi\-Hop QA数据构造Reward训练数据集，比如根据MetaQA中的1\-hop/2\-hop/3\-hop训练数据，构造带推理路径的CoT标注数据，结合自动化过程监督方法，标注Outcome\-Supervision和Process\-Supervision数据。

更长的推理路径

带反思/回溯的思维过程

O1 journey参考



**如何基于KGQA/Multi\-Hop QA数据\<自动化地构建有效的PRM数据集?**



## Thoughts

\+ PRM Verifier

\+ PRM guided RL FT

\+ PRM guided RL FT \+ PRM Verifier



提升answerer的generation能力

提升verifier的judgement能力

以self\-play的方式同时提升answerer和verifier的能力

通过verifier的evaluation提升answerer的reasoning能力

以RL的方式通过verifier的feedback提升answerer的能力



1. SFT warm\-up

Golden CoT的数据量，pacing

使用带推理路径的Golden CoT标注数据做SFT训练，（Q，R =\> A）



2. ORM标注/PRM标注

根据Golden CoT的partial数据进行多次rollout，得到新的CoT数据和结果，（Q，R' =\> R''，A'）

对于ORM标注，根据R''\+A'和A是否符合，对新推理路径给出1/0的outcome reward

对于PRM标注，根据多次rollout的结果，对partial CoT路径根据success/rollout给出process reward

KG可以从头推理，



**process reward的underestimate和overestimation，类似TD3/SAC，考虑clipped\-double\-prm？**



## ORM训练/PRM训练

1. ORM训练

当作binary classification使用cross\-entropy loss训练，

$L_{ORM} = - (y_s \log r_s + (1-y_s) \log (1-r_s))$

2. PRM训练

当作binary classification训练，

$L_{PRM} = - (\sum_{i=1}^{K} y_{s_i} \log r_{s_i} + (1-y_{s_i}) \log (1-r_{s_i}))$

其中$y_{s_i}$可以设置为 $\{y_{s_i} = success/rollout\}$ 或 $\{y_{s_i} = 1 | success>0\}$

> 其他的process supervision设置，比如语言判断，generative feedback
>
>

prefix\-PRM、mse\-PRM、cls\-PRM



## 基于ORM/PRM的RL训练

基于ORM和PRM给出的reward，执行step\-by\-step的RL训练



# Topics

KGQA

Multi\-Hop QA



# Todo

调研是否有利用QA数据蒸馏CoT做SFT的方法

根据KGQA和Multi\-Hop QA数据集构建CoT数据，训练PRM做verifier

阅读最新的KGQA相关paper

重要：先探索可行性

1. KG限制的动作空间

    1. o1 r1的think空间比较泛、相对的泛化性更强；

    2. kgqa上学习空间受限、学习难度应该会下降

2. 不同scale的kg，sub\-kg，low\-resource场景

3. 融合KGQA和MHQA，triple2paragraph，paragraph2triple，process reward construction



大语言模型的科研思维能力提升

为了提升大语言模型在科研思维能力上的表现，可以通过分析科研论文的引用网络、研究人员的关键文献脉络以及研究主题的发展流向，来全面梳理抽象科研思维的继承与发展关系。首先，通过追踪一篇科研论文的引用链，可以了解该论文如何影响后续研究，以及不同领域之间如何通过引用建立联系。其次，结合文献脉络分析，可以揭示某个科研问题从最初提出到不断发展的过程，帮助模型理解如何从一个简单的假设逐步引导出复杂的科学发现。此外，通过研究主题的发展流向，能够识别出科研领域中的热点问题和未解决的难题，为模型提供关于科研创新的方向。最终，整合这些信息，模型可以逐步掌握科研领域内的思维方式，模拟科研人员在文献分析、问题推理和理论发展的过程中所表现出的系统性和创新性思维，从而在提升科研思维能力的同时，增强其为科研工作提供支持的能力。





# Papers

## R1 related

[Understanding R1\-Zero\-Like Training: A Critical Perspective](https://arxiv.org/pdf/2503.20783)

各种R1改进的对比：SimpleRL\-Zero、Prime\-Zero、OpenReasoner\-Zero、Oat\-Zero（本文）

## PRM\-RL related

[Training Verifiers to Solve Math Word Problems](https://arxiv.org/pdf/2110.14168)

Verifier（PRM）细节，Appendix E，4\.2\-4\.3

[Solving math word problems with process\- and outcome\-based feedback](https://arxiv.org/abs/2211.14275)

[Let's Verify Step by Step](https://openreview.net/forum?id=v8L0pN6EOi)

ORM和PRM细节，Appendix E和Appendix F

选择convincing wrong\-answer solutions进行标注

[ReFT: Reasoning with Reinforced Fine\-Tuning](https://aclanthology.org/2024.acl-long.410.pdf)

[On Designing Effective RL Reward at Training Time for LLM Reasoning](https://arxiv.org/abs/2410.15115)

Process\-Reward（PR）可能损害RL微调，加入Clip和Delta技巧可以有效缓解PR失效的问题

OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models

PRM（math\-psa）训练细节是什么？

https://github\.com/openreasoner/openr

相关论文：https://github\.com/openreasoner/openr?tab=readme\-ov\-file\#reference

o1\-Coder: an o1 Replication for Coding

https://mp\.weixin\.qq\.com/s/8hksqHOvJL2YQ8eb5BHPjA

[OVM, Outcome\-supervised Value Models for Planning in Mathematical Reasoning](https://arxiv.org/pdf/2311.09724)

PRM的训练标注有问题

[Scaling LLM Test\-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/pdf/2408.03314)

\[GenRM\][Generative Verifiers: Reward Modeling as Next\-Token Prediction](https://arxiv.org/pdf/2408.15240)，ICLR 2025

[Generative Reward Models](https://arxiv.org/pdf/2410.12832)，Arxiv，ICLR 2025 Reject

[AlphaMath Almost Zero: Process Supervision without Process](https://openreview.net/pdf?id=VaXnxQ3UKo)，NeurIPS 2024

[Qwen2\.5\-math technical report: Toward mathematical expert model via self\-improvement](https://arxiv.org/pdf/2409.12122)

[Deepseekmath: Pushing the limits of mathematical reasoning in open language models](https://arxiv.org/pdf/2402.03300)

[Improving large language models via fine\-grained reinforcement learning with minimum editing constraint](https://aclanthology.org/2024.findings-acl.338.pdf)

[Teaching large language models to reason with reinforcement learning](https://arxiv.org/pdf/2403.04642)

[Enhancing LLM Reasoning with Reward\-guided Tree Search](https://arxiv.org/pdf/2411.11694)

[Imitate, Explore, and Self\-Improve: A Reproduction Report on Slow\-thinking Reasoning Systems](https://arxiv.org/pdf/2412.09413)

[WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol\-Instruct](https://openreview.net/pdf?id=mMPMHWOdOy)

ICLR 2025 Review，8888



[Natural Language Reinforcement Learning](https://arxiv.org/pdf/2402.07157)

[Natural Language Fine\-Tuning](https://arxiv.org/pdf/2412.20382)



https://github\.com/PRIME\-RL/PRIME

Enhancing Multi\-Step Reasoning via Process\-Supervised Reinforcement Learning from Human Feedback

STEP\-RLHF: Step\-wise Reinforcement Learning from Human Feedback

Step\-level Value Preference Optimization for Mathematical Reasoning

Relative Preference Optimization: Enhancing LLM Alignment through Contrasting Responses across Identical and Diverse Prompts

Self\-Generated Critiques Boost Reward Modeling for Language Models

[Critique\-out\-Loud Reward Models](https://openreview.net/pdf?id=e3odKmatZr)，ICLR 2025 Review

Improving Reward Models with Synthetic Critiques

Reinforcing Thinking through Reasoning\-Enhanced Reward Models

\[rStar\][Mutual Reasoning Makes Smaller LLMs Stronger Problem\-Solvers](https://openreview.net/pdf?id=6aHUmotXaw)，ICLR 2025 Review

[rStar\-Math: Small LLMs Can Master Math Reasoning with Self\-Evolved Deep Thinking](https://arxiv.org/pdf/2501.04519)，Microsoft，**Good**

[The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/pdf/2501.07301)，Qwen

[ProcessBench: Identifying Process Errors in Mathematical Reasoning](https://arxiv.org/pdf/2412.06559)，Qwen

[Entropy\-Regularized Process Reward Model](https://arxiv.org/pdf/2412.11006)

[PRMBench: A Fine\-grained and Challenging Benchmark for Process\-Level Reward Models](https://arxiv.org/pdf/2501.03124)

[Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain\-of\-Thought](https://arxiv.org/pdf/2501.04682)



## Advanced papers

[AutoPSV: Automated Process\-Supervised Verifier](https://openreview.net/pdf?id=eOAPWWOGs9)，NeurIPS 2024

[Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning](https://openreview.net/pdf?id=A6Y7AqlzLW)，ICLR 2025 Review

[Process Reward Model with Q\-value Rankings](https://openreview.net/pdf?id=wQEdh2cgEk)，ICLR 2025 Review

\[ImplicitPRM\][Free Process Rewards without Process Labels](https://arxiv.org/pdf/2412.01981)，Arxiv 2024\.12，**Good**，More Recent Works

Enhancing reinforcement learning with dense rewards from language model critic，EMNLP 2024

Dense reward for free in reinforcement learning from human feedback，ICML 2024

Entropy\-regularized process reward model，Arxiv 2024\.12

> 内部图片链接已移除。



## related work in *Math\-Shepherd*

[Making language models better reasoners with step\-aware verifier](https://aclanthology.org/2023.acl-long.291.pdf)，ACL 2023，early work，**Good**

[Grace: Discriminator\-guided chain\-of\-thought reasoning](https://aclanthology.org/2023.findings-emnlp.1022.pdf)

[Let’s reward step by step: Step\-level reward model as the navigators for reasoning](https://arxiv.org/pdf/2310.10080)

[Let’s reinforce step by step](https://arxiv.org/pdf/2311.05821)

[Fine\-grained human feedback gives better rewards for language model training](https://openreview.net/pdf?id=CSbGXyCswu)，NeurIPS 2023



## related work in *MiPS*

[Refiner: Reasoning feedback on intermediate representations](https://aclanthology.org/2024.eacl-long.67.pdf)

[Alphazero\-like tree\-search can guide large language model decoding and training](https://openreview.net/pdf?id=C4OpREezgj)，ICML 2024



## related work in *OmegaPRM*

[Mindstar: Enhancing math reasoning in pre\-trained llms at inference time](https://arxiv.org/pdf/2405.16265)

[ReST\-MCTS\*: LLM Self\-Training via Process Reward Guided Tree Search](https://openreview.net/pdf?id=8rcFOqEud5)，NeurIPS 2024

Tree of thoughts: Deliberate problem solving with large language models

Reasoning with language model is planning with world model



## Automated Process Supervision

[Math\-Shepherd: Verify and Reinforce LLMs Step\-by\-step without Human Annotations](https://aclanthology.org/2024.acl-long.510.pdf)

论文使用hard estimation标注PRM数据和训练PRM，以保持与标准语言建模的一致性和最小改动

> *For the sake of convenience, we train the PRM using the hard estimation version because it allows us to utilize a standard language modeling pipeline by selecting two special tokens to represent ‘has potential’ and ‘no potential’ labels, thereby eliminating the need for any specific model adjustments\.*
>
>

[Multi\-step Problem Solving Through a Verifier: An Empirical Analysis on Model\-induced Process Supervision](https://aclanthology.org/2024.findings-emnlp.429.pdf)

着重分析了自动标注PRM训练数据中的噪声对最终PRM的影响，提供了训练PRM时reward标注的建议

讨论了hard estimation和soft estimation的区别，倾向soft estimation

> *For the noisy MiPS data, we suggest aggregation functions that focus on high predicted scores\.*
>
> *The max aggregation is better for the soft objective and the min aggregation is better for the hard objective\.*
>
>

[Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)

> *To address this challenge, we introduce OmegaPRM, a novel divide\-and\-conquer Monte Carlo Tree Search algorithm inspired by AlphaGo Zero for automated process supervision data collection\.*
>
> Previous works \(Lightman et al\., 2023; Wang et al\., 2024a,b\) use rule\-based strategies to split a solution into steps, e\.g\., using newline as delimiters\. In contrast, we propose a more flexible method for step division, treating any sequence of consecutive tokens in a solution as a valid step\. Therefore, we hypothesize that semantically explicit cutting is not necessary for training a PRM\.
>
>



## Blogs

https://huggingface\.co/spaces/HuggingFaceH4/blogpost\-scaling\-test\-time\-compute

https://mp\.weixin\.qq\.com/s/E1FaaOurAb\-QlCX3BASi9Q

OpenRLHF源码解读：理解PRM\(过程奖励模型\)训练过程

https://zhuanlan\.zhihu\.com/p/16027048017



## KGQA related

\[MindMap\][MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models](https://aclanthology.org/2024.acl-long.558.pdf)

\[CoK\][Boosting Language Models Reasoning with Chain\-of\-Knowledge Prompting](https://aclanthology.org/2024.acl-long.271.pdf)

prompt中加入evidence triple和evidence hint，OpenBookQA

\[ChatKBQA\][ChatKBQA: A Generate\-then\-Retrieve Framework for Knowledge Base Question Answering with Fine\-tuned Large Language Models](https://aclanthology.org/2024.findings-acl.122.pdf)，webqsp

\[KnowGPT\][KnowGPT: Knowledge Graph based Prompting for Large Language Models](https://openreview.net/pdf?id=PacBluO5m7)，NeurIPS 2024，**Good**

OpenBookQA

\[KGR\][Mitigating Large Language Model Hallucinations via Autonomous Knowledge Graph\-Based Retrofitting](https://arxiv.org/pdf/2311.13314)

HotpotQA

\[PoG\][Plan\-on\-Graph: Self\-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs](https://openreview.net/pdf?id=CwCUEr6wO5)

Finetuned KG\-Augmented LLM Methods \[Appendix D Baseline Descriptions\]

https://scholar\.google\.com/scholar?cites=2667826903821217588\&as\_sdt=5,33\&sciodt=0,33\&hl=zh\-CN

with code，webqsp，with all full prompts，**Good**

\[ToG\][Think\-on\-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph](https://openreview.net/pdf?id=nnVO1PvbTv)

webqsp

\[ToG\-2\.0\][Think\-on\-Graph 2\.0: Deep and Faithful Large Language Model Reasoning with Knowledge\-guided Retrieval Augmented Generation](https://arxiv.org/pdf/2407.10805)，webqsp

[Improving LLM\-based KGQA for multi\-hop Question Answering with implicit reasoning in few\-shot example](https://aclanthology.org/2024.kallm-1.13.pdf)

metaqa

[Retrieval and Reasoning on KGs: Integrate Knowledge Graphs into Large Language Models for Complex Question Answering](https://aclanthology.org/2024.findings-emnlp.446.pdf)，with kg\-cot training，with train detials，webqsp

[Interleaving Retrieval with Chain\-of\-Thought Reasoning for Knowledge\-Intensive Multi\-Step Questions](https://aclanthology.org/2023.acl-long.557.pdf)，musique

[Making Long\-Context Language Models Better Multi\-Hop Reasoners](https://aclanthology.org/2024.acl-long.135.pdf)，musique

[HOLMES: Hyper\-Relational Knowledge Graphs for Multi\-hop Question Answering using LLMs](https://aclanthology.org/2024.acl-long.717.pdf)，misuque，~~code~~

[EfficientRAG: Efficient Retriever for Multi\-Hop Question Answering](https://aclanthology.org/2024.emnlp-main.199.pdf)，完整的prompt，with code，musique

\[RoG\][Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning](https://openreview.net/pdf?id=ZGNWW7xZ6Q)

ICLR，WebQSP，CWQ，with training

[Enhancing Complex Question Answering over Knowledge Graphs through Evidence Pattern Retrieval](https://dl.acm.org/doi/pdf/10.1145/3589334.3645563)，webqsp

[Generate\-then\-Ground in Retrieval\-Augmented Generation for Multi\-hop Question Answering](https://aclanthology.org/2024.acl-long.397.pdf)，musique

[Reasoning with Trees: Faithful Question Answering over Knowledge Graph](https://aclanthology.org/2025.coling-main.211.pdf)，webqsp

[RGR\-KBQA: Generating Logical Forms for Question Answering Using Knowledge\-Graph\-Enhanced Large Language Model](https://aclanthology.org/2025.coling-main.205.pdf)，webqsp

[ReARTeR: Retrieval\-Augmented Reasoning with Trustworthy Process Rewarding](https://arxiv.org/pdf/2501.07861)

**Most Recent**，Arxiv 2025\.01\.08，Multi\-step QA，PRM，musique

KGQA

Reasoning of Large Language Models over Knowledge Graphs with Super\-Relations

Training Large Language Models for Retrieval\-Augmented Question Answering through Backtracking Correction

Chain\-of\-Action: Faithful and Multimodal Question Answering through Large Language Models

[KnowPath: Knowledge\-enhanced Reasoning via LLM\-generated Inference Paths over Knowledge Graphs](https://arxiv.org/pdf/2502.12029?)

Harnessing Large Language Models for Knowledge Graph Question Answering via Adaptive Multi\-Aspect Retrieval\-Augmentation

Multi\-Hop QA

[AMOR: A Recipe for Building Adaptable Modular Knowledge Agents Through Process Feedback](https://arxiv.org/pdf/2402.01469)

[Chain\-of\-Thought Matters: Improving Long\-Context Language Models with Reasoning Path Supervision](https://arxiv.org/pdf/2502.20790)

[Chain of Preference Optimization: Improving Chain\-of\-Thought Reasoning in LLMs](https://openreview.net/pdf?id=2cczgOfMP4)

[Mitigating Lost\-in\-Retrieval Problems in Retrieval Augmented Multi\-Hop Question Answering](https://arxiv.org/pdf/2502.14245)

[Chain\-of\-Retrieval Augmented Generation](https://arxiv.org/pdf/2501.14342)

[SymAgent: A Neural\-Symbolic Self\-Learning Agent Framework for Complex Reasoning over Knowledge Graphs](https://arxiv.org/pdf/2502.03283)

[Generate\-on\-Graph: Treat LLM as both Agent and KG for Incomplete Knowledge Graph Question Answering](https://aclanthology.org/2024.emnlp-main.1023.pdf)

Implement KG



[SeRTS: Self\-Rewarding Tree Search for Biomedical Retrieval\-Augmented Generation](https://aclanthology.org/2024.findings-emnlp.71.pdf)

[RAG\-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment](https://arxiv.org/pdf/2412.13746)

[Trustworthy Alignment of Retrieval\-Augmented Large Language Models via Reinforcement Learning](https://openreview.net/pdf?id=XwnABAdH5y)

[MultiHop\-RAG: Benchmarking Retrieval\-Augmented Generation for Multi\-Hop Queries](https://openreview.net/pdf?id=t4eB3zYWBK)

[RAG\-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment](https://arxiv.org/pdf/2412.13746)



# Datasets

## KGQA

WebQSP

CWQ

GrailQA

KQA Pro

MetaQA



## Multi\-Hop QA \(MHQA\)

MuSiQue

HotpotQA

2WikiMultiHopQA

Bamboogle



COM^2（[Complex Reasoning over Logical Queries on Commonsense Knowledge Graphs](https://aclanthology.org/2024.acl-long.613.pdf)）

math领域、logic知识、meta知识

CommonsenseQA

OpenBookQA

MedQA
