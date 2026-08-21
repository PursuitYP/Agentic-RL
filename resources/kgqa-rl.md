# KGQA Agentic RL：工程与实验记录

# Introduction

Unify KGQA \(Knowledge Graph QA\) and MHQA \(Multi\-Hop QA\) with CoT reasoning

给定question，检索KG三元组或者检索Paragraph进行answering

【这个是之前写的，针对的是QA场景，范围比较局限，现在需要再改改】



两种setting：

open setting，只提供question，在freebase数据库或者wikipedia百科资料中进行开放式检索

closed setting，提供question和context，基于额外的上下文进行封闭式问答

context中包含有用的triple或者paragraph，同时有相关但没用的distractor信息



关键步骤：

question decomposition

step\-by\-step answering

sub\-question status update

sub\-question dynamic revise



Todo：

automated CoT process annotation \(with LLM assistance\)

triple2paragraph

paragraph2triple



# Method

## KGQA\-R1

## **KGQA\-R1**

**研究背景与动机：**

当前LLMs在自然语言问答任务中表现出色，但在处理多跳推理和复杂知识整合时仍存在不足。引入结构化的KG知识可以为模型提供更精准的知识支持和推理路径，从而提升答案的准确性和解释性。

通过将KG检索操作嵌入到LLMs的推理流程中，并利用强化学习（RL）训练模型自主决策何时调用KG检索，使模型能够在检索与推理的交替过程中高效利用结构化知识。

**两阶段RL训练算法：**

检索阶段：学习何时以及如何发起KG检索请求。

奖励：子问题分解、检索发起、响应格式

推理阶段：基于检索到的信息进行推理和答案生成。

基本奖励：最终答案的F1值、响应格式

路径一致奖励：检索到推理路径上关键实体

**Rule\-Based RL训练：**

首选GRPO和REINFORCE\+\+

将KG检索融入到policy model的rollout中

训练时将检索到的内容mask掉来计算loss

**课程学习策略：**

构建一个课程学习体系，从简单问题（低跳数）开始，逐步引入多跳、复杂推理任务。

随着RL训练的深入，逐步增加知识图谱中检索跳数，提升模型的多跳推理能力。

**数据集和评测：**

训练数据集：

WebQSP和CWQ，基于FreeBase的KGQA

泛化测试：

MetaQA，基于movie的KGQA

KQA Pro，基于wikidata的更多跳KGQA

MuSiQue、HotpotQA，基于wikidata的Multi\-Hop QA

如TriviaQA、PopQA，其他更general的QA推理



> https://arxiv\.org/abs/2503\.09516
>
> https://github\.com/PeterGriffinJin/Search\-R1
>
>



> deepseek\-r1、logic\-rl \(curriculum learning\)、ReMa \(meta\-think\)
>
> ReARTeR、r1\-searcher、search\-r1、ReSearch
>
> kbga\-o1、mcts\-kbqa
>
> Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains
>
> Revisiting Reinforcement Learning for LLM Reasoning from A Cross\-Domain Perspective
>
>

```Plain Text
reasoning_prompt = """You are a helpful assistant specialized in multi-hop question answering using structured knowledge graphs.
When given a question, start by reasoning step-by-step inside <think> and </think> tags.
As you reflect, if you identify a need for specific structured knowledge to solve a sub-problem, invoke a knowledge graph retrieval by issuing a <kg_search> query </kg_search> and then incorporate the returned information provided between <kg_information> and </kg_imformation> tags.
Continue this iterative process of decomposing the question, verifying, and refining your reasoning—alternating between internal thought and KG retrieval as needed—until you are ready to finalize your response.
Once confident in your solution, provide the final answer enclosed within <answer> and </answer> tags, ensuring that the exact final answer is formatted with LaTeX as \boxed{final answer}."""


kg_reasoning_prompt = f"""Answer the given Question.
You must conduct reasoning inside <think> and </think> first every time you get new information.
Your goal is to answer the question through multi-hop reasoning. Specifically, you should:
1. Decompose the question into sub-questions if needed;
2. Iteratively perform the following three steps:
   (a) Conduct a knowledge graph retrieval using exactly ONE entity with <search> entity </search>;
   (b) Update the current sub-question state based on the top retrieved knowledge triples returned between <information> and </information>;
   (c) Evaluate whether the current knowledge is sufficient to continue reasoning or answer the question.

To perform external retrieval, use the <search> entity </search> tag. For example, <search> London </search>.
Each search query must involve exactly ONE entity, and the search engine will return top relevant knowledge graph triples (entity | relation | entity) between <information> and </information>.
You can search as many times as your want. You can begin your search by using the entities in Topic Entities.
You may ONLY search using entities that come from either the initial Topic Entities or entities mentioned in the retrieved knowledge graph triples.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
Question: {question}
Topic Entities: {topic_entities}
"""
```

```JSON
'extra_info': {'id2name': '{"m.0hn47qp": "The Baltimore Fight Song", "m.06x5s": "Super Bowl"}', 'index': 1, 'inference_chain': '["sports.sports_team.fight_song", "sports.sports_team.championships", "sports.sports_championship_event.championship"]', 'name2id': '{"The Baltimore Fight Song": "m.0hn47qp", "Super Bowl": "m.06x5s"}', 'split': 'train', 'topic_entity': '{"m.0hn47qp": "The Baltimore Fight Song", "m.06x5s": "Super Bowl"}'}
```

## 0518进度更新

- **数据集准备**

    - **完成**webqsp和cwq数据集的预处理，包含question，answer，topic\_entity，inference\_chain等关键字段

        - question为原始问题，answer为答案\[列表\]

        - topic\_entity为初始实体\[列表\]，inference\_chain为中间关系\[列表\]

        - **0529更新：将topic\_entity和inference\_chain放在parquet数据文件的extra\_info字段中**

    - **完成**reward\_function设计，根据answer字段和inference\_chain字段构造reward

        - if：response中\<answer\>能exact match匹配answer，得1\.0分【答案正确】

            - **0528更新：使用f1值作为得分**

            - https://github\.com/Alibaba\-NLP/ZeroSearch/blob/main/verl/utils/reward\_score/qa\_em\.py\#L70

        - elif：response中包含inference\_chain的元素，得0\.1分【答案错误但探索到关键relation】

        - else： 得0\.0分【答案错误且无效探索】

    - **完成**课程学习数据准备，根据inference\_chain的元素个数构造跳数逐步增加的数据集

        - 推理难度依次增加，从1\-hop\[简单\]到5\-hop\[困难\]

    - ***失败的尝试***

        - 基于之前论文的预处理方法准备数据 =》存在训练数据未标注、实体id缺失、原始sparql丢失等问题

- **KG检索服务准备**

    - **完成**KG部署和测试，将freebase三元组导入virtuoso数据库，支持sparql查询

        - 一般三元组为（entity\_id，relation，entity\_id）的形式，entity\_id 比如 m\.01qdhx / m\.01428y

        - 特殊三元组，（entity\_id，ns:type\.object\.name，entity\_name），即每个符号的 entity\_id 通过 ns:type\.object\.name 连接到自然语言的 entity\_name

    - **完成**KG检索工具调用建模，外层以entity\_name进行调用和推理，内层以entity\_id进行检索和处理

        - 大模型外层调用search entity\_name接口，传入entity\_name列表 \[限制至多三个\]

        - 后台内层search接口维持一个batch内的name2id和id2name动态映射，先用name2id将entity\_name转换为entity\_id进行KG检索，筛选处理后再利用id2name将结果转换成自然语言的三元组返回给大模型

    - **待优化**的KG检索工具调用 \[可选\]

        - 直接搜索给的entity\_name的三元组可能数量过多，考虑分两步调用KG检索，先根据entity检索relation，再根据entity\+relation检索三元组，其中大模型需要推理给出第一步扩展的entity和第二步扩展的relation

        - 添加一个triple ranking功能，根据截止当前的消息对三元组进行排序，保留前若干个triple，比如50个

    - ***失败的尝试***

        - 考虑到entity\_id和entity\_name之间的语义gap，预处理大规模KG，将entity\_id统一替换为entity\_name，直接利用entity\_name检索KG =》存在entity linking \(EL\) 问题，即多个entity\_id对应同一个entity\_name

        - 是否可以省去batch内的name2id和id2name动态映射，直接将id\<\-\>name的转换嵌入到查询sparql语句，类似SELECT ?rel ?e1 WHERE \{ ?e0 ns:type\.object\.name "Justin" \. ?e0 ?rel ?e1 \.\} =》同样存在EL问题

- **代码编写**

    - **完成**各模块代码的编写与测试，正在将KG数据和检索相关部分融入search\_r1框架

        - 修改 ray\_trainer\.py L729 中的 run\_llm\_loop 函数，对应 \./search\_r1/llm\_agent/generation\.py，关键代码在 generation\.py L253/L296/L353 中的 execute\_predictions 函数，正在修改传参列表和返回格式

        - 对于 batch 内的检索

            - 参考 ZeroSearch L580，使用多线程 ThreadPoolExecutor 实现 batch\_search

            - 参考 schr1 retrieval\_server\.py 中的 DenseRetriever，基于 e5 计算 embedding 实现 top\-k 检索

                - 潜在缺点：每次都需要对三元组列表建立索引，可能会比较慢

                - 存在某个实体的三元组特别多的情况

                    - 优化：对distinct relation建立索引，而不是对三元组建立索引，可以减少建立索引时间

```JSON
{
    "question_id": "WebQTest-0",
    "question": "what does jamaican people speak?",
    "answer": {
        "m.01428y": "Jamaican English",
        "m.04ygk0": "Jamaican Creole English Language"
    },
    "topic_entity": {
        "m.03_r3": "Jamaica"
    },
    "inference_chain": [    # 1-hop，2-path
        [
            "location.country.languages_spoken"
        ],
        [
            "location.country.official_language"
        ]
    ]
},
{
    "question_id": "WebQTest-1",
    "question": "what did james k polk do before he was president?",
    "answer": {
        "m.02_bcst": "United States Representative",
        "m.04x_n9q": "Governor of Tennessee",
        "m.0cgqx": "Speaker of the United States House of Representatives"
    },
    "topic_entity": {
        "m.042f1": "James K. Polk"
    },
    "inference_chain": [    # 4-hop，1-path
        "government.politician.government_positions_held",
        "government.government_position_held.office_position_or_title",
        "government.government_position_held.basic_title",
        "government.government_position_held.from"
    ]
}
```

```JSON
{
    "question_id": "WebQTest-832_c334509bb5e02cacae1ba2e80c176499",
    "webqsp_id": "WebQTest-832",
    "question": "Lou Seal is the mascot for the team that last won the World Series when?",
    "answer": {
        "m.0117q3yz": "2014 World Series"
    },
    "aliases": [
        "2014 World"
    ],
    "topic_entity": {
        "m.03_dwn": "Lou Seal"
    },
    "inference_chain": [    # 3-hop，1-path
        "sports.sports_team.team_mascot",
        "sports.sports_team.championships",
        "time.event.start_date"
    ]
},
{
    "question_id": "WebQTrn-1259_1997cb4922db71983be26e6a509950f4",
    "webqsp_id": "WebQTrn-1259",
    "question": "Where did the \"Country Nation World Tour\" concert artist go to college?",
    "answer": {
        "m.01qdhx": "Belmont University"
    },
    "aliases": [
        "Belmont",
        "Belmont University, main campus"
    ],
    "topic_entity": {
        "m.010qhfmm": "Country Nation World Tour",
        "m.019v9k": "Bachelor's degree"
    },
    "inference_chain": [    # 4-hop，1-path
        "music.artist.concert_tours",
        "people.person.education",
        "education.education.institution",
        "education.education.degree"
    ]
}
```



## 0526进度更新

- **KG检索服务准备**

    - **完成**检索后三元组的过滤

        - 即使每次对一两个实体进行KG检索，也存在几百个三元组结果的情况，需要对三元组进行二次过滤

        - 使用小模型部署基于embedding的re\-ranking服务，先选择top\-k1的relation，再选择top\-k2的triple

    - **完成**batch三元组检索接口

        - 区别于search\-r1的batch query，使用ThreadPoolExecutor多线程实现三元组的batch检索

        - 修改search\_r1的rerank\_server\.py实现检索结果重排序，CrossEncoder可以指定batch\_size

- **正在进行**

    - 将KG检索启动为后端API服务，方便大模型进行即插即用的检索调用

    - 继续集成各项功能代码，预计三天内完成模块整合，本周内取得初步实验结果



## 0602进度更新

- **KG检索服务**

    - **完成**基于entity\_name列表的三元组search和batch\_search，使用多线程实现batch\_search加速

    - **完成**检索结果的二次筛选，针对检索返回大量三元组的情况，使用CrossEncoder进行reranking

    - **完成**KG检索服务完整pipeline的测试，将服务启动在指定API端口，以便大模型自动调用KG检索

- **数据集校验**

    - **完成**最终数据集的转换和测试，将预处理好的webqsp\-cwq联合数据转换成verl训练的parquet格式

        - 更新数据集字段，将topic\_entity和inference\_chain放在parquet文件的extra\_info字段

    - **完成**reward\_function的修改，同时计算EM和F1得分，使用F1作为reward，同时支持EM评测

- **本周计划**

    - 在现有完整代码框架上继续进行RL训练，并开展后续**优化**

        - 优化维度1：检索三元组的两阶段过滤，先选择top\-k1的relation，再选择top\-k2的triple

        - 优化维度2：完善reward\_function设计，加入严格的format\_reward

    - 开始编写在其他数据集上进行**泛化**测试的代码



## 0616进度更新

- **数据集准备**

    - 修改数据集的prompt并调整格式，question和topic\_entity融合进prompt，inference\_chain放在extra\_info

    - 修改reward\_function，加入格式奖励；根据PoG/ToG代码改写EM评分机制，仍使用F1作为score

- **本周计划**

    - 实现batch中filtered\_name2id和filtered\_id2name的每轮更新

        - 将两个字典从gen\_batch中单独拿出来，不用更新gen\_batch

    - 完成各模块代码融合，取得初步实验结果；根据inference\_chain构造课程学习数据集



## 0620进度更新

- **整体框架**

    - 按照最开始的设想和每周实际进度，完成各模块测试，代码正在运行中

    - 推理框架

        - llm agent在推理过程中，如需额外知识，选择一个实体发起search请求，agent生成暂停；

        - 后端server接收这个请求，并进行三元组检索和三元组重排序，将最相关的100个三元组返回给agent；

        - agent根据新的information继续推理；

        - 上述整个过程持续若干轮。

- **当前的效果**

    - 在 wiki search 检索环境中验证 reward 和 dataset 有效性，qwen2\.5 3B 7B

    - 在 kg search 环境中验证完整 pipeline 的有效性，qwen2\.5 3B

    - query处理应该放在工具层面，而不是限制llm输出entity

- **存在的问题**

    - 运行速度很慢，一个 step 需要6\.5分钟（基于 wiki search 约2分钟）

        - 当前三元组检索和重排序花费时间较长【非常长！】

- **后续优化**

    - 检索加速

        - retriever：优化 virtuoso 数据库的配置，提高最大连接数和缓存配置

        - reranker：提高 reranker 的 batch\_size

        - retrieval\-rerank server: 提高 batch\_search 的 max\_workers

    - rerank 优化

        - 先选择 top\-k1 的 relation，再选择 top\-k2 的 triple

        - 手动滤除过长的 entity，大部分是意义不大的 quotation

    - agent 完善

        - 精简 prompt

        - 修改 execute\_predictions 函数的状态更新部分

            - 针对 information 为空的检索，提示检索的实体要来自 topic\_entity 或者之前的 information

            - 针对错误 action 的 next\_obs，提示要检索 entity 而不是 query

    - reward 改进

        - 加入一跳reward和多跳reward（需要对数据集进行再次处理）

    - SFT 冷启动

        - 修改整体框架，使用 Deepseek\-R1 蒸馏 trajectory，作为 base LLM 的冷启动数据

    - **运行中 dubug**

        - 处理 non\-empty retrieval size 为零的情况，此时不进行rerank

        - Eval 调试，参数里 test\_freq 改成50，方便查看效果改进

        - 处理 rerank 后实体中包含符号“\|”的情况，会导致 \_string\_to\_triple 参数数量不一致

            - 临时处理，保留前三个；后续可以根据 retrieved\_relations\_ls 进行处理

            - 优化处理，在 \_triple\_to\_string 处理的时候，提前去掉三元组中的“\|”符号

            - 返回原始的triple



- **当前prompt**

    ```XML
    kg_reasoning_prompt = f"""Answer the given Question.
    You must conduct reasoning inside <think> and </think> first every time you get new information.
    Your goal is to answer the question through multi-hop reasoning. Specifically, you should:
    1. Decompose the question into sub-questions if needed;
    2. Iteratively perform the following three steps:
       (a) Conduct a knowledge graph retrieval using exactly ONE entity with <search> entity </search>;
       (b) Update the current sub-question state based on the top retrieved knowledge triples returned between <information> and </information>;
       (c) Evaluate whether the current knowledge is sufficient to continue reasoning or answer the question.

    To perform external retrieval, use the <search> entity </search> tag. For example, <search> London </search>.
    Each search query must involve exactly ONE entity, and the search engine will return top relevant knowledge graph triples (entity | relation | entity) between <information> and </information>.
    You can search as many times as your want. You can begin your search by using the entities in Topic Entities.
    You may ONLY search using entities that come from either the initial Topic Entities or entities mentioned in the retrieved knowledge graph triples.
    If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
    Question: {question}
    Topic Entities: {topic_entities}
    """
    ```



- **KG检索服务实现细节**

    - LLM仅可见entity\_name三元组，search\_query和三元组排序也是根据entity\_name

        - run\_llm\_loop中调用**execute\_predictions**，传入参数：response\_str、gen\_batch\['question'\]、gen\_batch\['name2id'\]、gen\_batch\['id2name'\]、active\_mask

        - execute\_prediction中调用**batch\_search**，传入参数：search\_entities、question、name2id、id2name

        - batch\_search中调用**\_batch\_search**，传入参数 payload 中加上 topk\_rerank 和 return\_scores

    - 输入：原始问题question，当前的回答response（包含search\_query），name2id字典，id2name字典

    - 输出：首先检索head\_triple和tail\_triple，处理后保留最终的triple list，以及name2id和id2name字典

    - 处理步骤

        - 根据name2id字典，将search\_query中的entity\_name列表转换成entity\_id列表

        - 根据entity\_id列表搜索head\_triple和tail\_triple，然后是三元组的后处理

            - 使用abandon\_rels函数筛除参考意义不大的三元组，得到all\_triple

            - 使用id2entity\_name\_v2函数将all\_triple转换成entity\_name三元组

                - 如果entity\_id已经在id2name字典中，直接转换，无需查询

                - 同时更新name2id字典和id2name字典【需要再次更新，缩减字典大小】

            - 根据all\_triple提取不重复的relation得到all\_relation

        - 如果all\_triple数量小于100

            - 将response和triple拼接后作为CrossEncoder的输入，得到相关性分数

            - 根据相关性分数从大到小排序，将排序后的三元组作为输出

        - 否则如果all\_triple数量小于1000

            - 将response和triple拼接后作为CrossEncoder的输入，得到相关性分数

            - 根据相关性分数从大到小排序，取前100个结果，将排序后的三元组作为输出

        - 否则（all\_triple数量大于1000）

            - 将response和triple拼接后作为CrossEncoder的输入，得到相关性分数

            - 根据相关性分数从大到小排序，取前100个结果，将排序后的三元组作为输出

            - \[可选\] 先根据all\_relation进行第一轮筛选，再根据all\_triple进行第二轮筛选

        - **根据筛选后保留的三元组实体，过滤name2id和id2name得到再次更新后的字典**



**初步想法**

将KG检索作为LLMs可选的操作，使用RL训练让LLMs在推理过程中自主调用KG检索

Reward design

Meta\-QA

high\-level，执行子问题分解

low\-level，查询KG解决子问题

回答准确性的F1作为reward

另外包括query到关键实体的reward

还有format reward

检索到推理路径上的实体给reward

构建由简单到难的课程学习（跳数逐步增加）

分两步RL训练：检索RL、推理RL

在WebQSP和CWQ上训练，泛化到MetaQA，进一步泛化到Multi\-Hop QA和其他推理数据集

考虑base模型和instruct模型的prompt细微差别，参考ReSearch

> deepseek\-r1、logic\-rl \(curriculum learning\)、ReMa \(meta\-think\)
>
> ReARTeR、r1\-searcher、search\-r1、ReSearch
>
> kbga\-o1、mcts\-kbqa
>
> Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains
>
>

基于verl，search\-r1改动比较清晰

model\.generate指定stopping\_criteria参数

基于verl，ReSearch改动有点大

基于openrlhf，r1\-searcher改动有点大



**实现细节**

1. 数据

https://drive\.google\.com/drive/folders/18H2JXDFPWfe4WeSXVpf1lUxdpraB2OvW

webqsp和cwq数据预处理

提取relation path

构造reward function

2. 服务

部署kg server，测试检索

根据发起检索的实体，构造kg retrieval语句

细化reasoning promp

3. 训练

基于verl框架，参考search\-r1进行修改

llm\_agent、info\_mask、state\_masking

修改generactor\_rollout\_wg\.generate\_sequence



**【Record】**

数据集处理

字段：question\_id，question，answer，topic\_entity，inference\_chain

webqsp和cwq都有answer的entityId和entityName

清除数据集中没有answer的11个question

webqsp有topicEntity和inferenceChain，可以处理得到relationChain

WebQTest\-1133特殊处理，ns:common\.topic\.alias，类似的WebQTrn\-2789和WebQTrn\-1466

cwq需要首先根据原始sparql处理得到topicEntity，然后检索子图构造relationChain

也可以基于数据集原始sparql提取topicEntity和inferenceChain，再构造relationChain

~~根据q\_entity和a\_entity分别检索两跳子图，构造relationChain~~

确定每个问题的大致跳数，构造课程学习数据集



基于kg的entity\_name批量检索三元组接口

entity\_name \-\-\> entity\_id，很多情况是1\-n

entity\_id \-\-\> head/tail relation triple

entity\_id \-\-\> entity\_name，大部分情况是1\-1

维持 name2id 和 id2name 的两个映射？

\-\- 或者直接用interactiveKBQA中的es搜索方式 SearchGraphPatterns\(sparql, semantic\)

SELECT ?e WHERE \{ ?e ns:type\.object\.name "Malcolm X"@en \. \}

SELECT ?e WHERE \{ ?e0 ns:type\.object\.name "Justin Bieber"@en \. ?e0 ns:people\.person\.sibling\_s ?s \. ?s ns:people\.sibling\_relationship\.sibling ?e \. \}

\-\- Search\_r1 \- L372：self\.batch\_search\(search\_entity\_ls\)

search\_entity\_ls是batch内的多个search\_entity\_dict字典组成的list

search\_entity\_dict：\{entitd\_id: entity\_name, \.\.\., entity\_id: entity\_name\}



kg检索

v1：SELECT ?e0 ?rel ?e1 WHERE \{ ?e0 ns:type\.object\.name "Justin Bieber"@en \. ?e0 ?rel ?e1 \.\}

v2：根据topicEntity维持一个batch内的name2id和id2name映射

修改schr1\-L253的self\.execute\_predictions，增加name2id和id2name参数

LLM仅可见entityName三元组，search\_query也是根据entityName

execute\_predictions根据name2id执行查询，根据id2name执行转换



## KGQA\-GenRM

Generative RM

reasoning sampling：提供question，使用R1生成reasoning和answer

generative judging：提供question和groundtruth answer，使用R1判断第一步reasoning的正确性

genRM training：基于第二步的数据训练较小的模型作为专门的generative reward model

> Generative Verifiers: Reward Modeling as Next\-Token Prediction
>
> Generative Reward Models
>
> Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains
>
> GenPRM: Scaling Test\-Time Compute of Process Reward Models via Generative Reasoning
>
> Inference\-Time Scaling for Generalist Reward Modeling
>
>



构建简单的reasoning step进行SFT训练效果一般

根据KGQA和Multi\-Hop QA数据集构建CoT数据



重要：先探索可行性

1. KG限制的动作空间

    1. o1 r1的think空间比较泛、相对的泛化性更强；

    2. kgqa上学习空间受限、学习难度应该会下降

2. 不同scale的kg，sub\-kg，low\-resource场景

3. 融合KGQA和MHQA

    1. triple2paragraph，paragraph2triple，process reward construction



0317讨论：【转移研究重点】

**使用QA数据中的结构化知识和结构化推理过程来增强LLMs，在非QA场景的跨领域通用能力上进行评测**

## reasoning prompts

```Python
### 1. 利用R1、o3-mini等推理模型，基于groundtruth构造长CoT数据集 ###
# 提示LLMs基于问题、问题实体和三元组知识，生成中间思维过程的长CoT，来正确回答目标问题
### 2.1 使用长CoT数据直接SFT微调LLMs(<10B)，*通过提高结构化推理能力增强其通用推理能力 *###
### 2.2 基于上一步的数据集，进行MC-rollout，进一步构造PRM训练数据集 ###
```

```python
# 优化后的reasoning prompt
reasoning_prompt = f"""
You are a structured knowledge reasoning specialist. Follow this protocol to solve multi-hop KGQA tasks:

[Input Components]
1. QUESTION: May contain multiple entities requiring multi-step reasoning
2. QUESTION ENTITIES: Explicit entity list from the question
3. KG TRIPLES: Relevant (subject, predicate, object) triples from Freebase

[Reasoning Protocol]

# STEP 1: Question Decomposition
- Break the question into logically ordered sub-questions using linguistic cues and entity dependencies
- Maintain semantic continuity between sub-questions

# STEP 2: Knowledge Graph Exploration
- Anchor exploration using question entities as starting points
- For each sub-question:
  a) Select candidate triples with matching subjects/objects
  b) Prioritize triples with predicates matching the sub-question's intent
  c) Maintain reasoning chain continuity through shared entities
- If path breaks:
  ↳ Attempt alternative triple combinations
  ↳ Document dead ends before proceeding

# STEP 3: Knowledge Integration
When KG evidence is insufficient:
- Use your own knowledge and combine it with the explored KG triples to answer the question
- List the specific pieces of knowledge you used from your own knowledge to ensure transparency and clarity in your reasoning process.

# STEP 4: Path Validation
- Verify answer consistency through:
  1) Forward chaining: Confirm path leads to target answer
  2) Triple coverage: All critical triples must be utilized

[Output Requirements]
Strictly follow this sequence:
<REASONING>
- Explicitly mark each step with # STEP N prefixes
- For each reasoning action:
  * Document used triples as <TRIPLE_USED>(subject,predicate,object)
  * Record failed attempts as <ATTEMPT_FAIL>(reason)
  * Note knowledge integration as <KNOWLEDGE_FUSION>(content)
- Maintain step continuity through entity co-reference
</REASONING>

<ANSWER>
  FINAL_ANSWER
</ANSWER>

Required elements:
- All reasoning steps must precede the final answer
- Each protocol step must have corresponding documentation
- Triples usage must be explicitly linked to sub-questions

[Input Instance]
QUESTION:
{{question}}

QUESTION ENTITIES:
{{question_entities}}

KG TRIPLES:
{{triples}}
"""
```

## Long CoT examples

```Plain Text
{
    "question": "what country did jacques cartier came from",
    "question_entities": [
        "Jacques Cartier"
    ],
    "answer": [
        "France"
    ]
}


content:
<REASONING>
# STEP 1: Question Decomposition
- Determine the citizenship/nationality or country of origin of Jacques Cartier

# STEP 2: Knowledge Graph Exploration
- Anchor: 'Jacques Cartier'

TRIPLE_USED(Jacques Cartier, people.person.place_of_birth, Saint-Malo)
TRIPLE_USED(Saint-Malo, location.location.containedby, Brittany)
TRIPLE_USED(Brittany, location.administrative_division.country, France)

TRIPLE_USED(Jacques Cartier, people.person.nationality, France)

# STEP 3: Knowledge Integration
None required - sufficient KG evidence found

# STEP 4: Path Validation
1) Birthplace chain: Saint-Malo → Brittany → France confirms geographic origin
2) Nationality triple directly answers the question
</REASONING>

<ANSWER>
France
</ANSWER>


reasoning_content:
Okay, let's try to figure this out. The question is asking which country Jacques Cartier came from. First, I need to find out his nationality or place of birth from the KG triples. Let me look through the triples related to Jacques Cartier.

Hmm, here's a triple: (Jacques Cartier, people.person.place_of_birth, Saint-Malo). So he was born in Saint-Malo. Now, where is Saint-Malo located? Looking through the triples, I see (Saint-Malo, location.location.containedby, Brittany). Wait, and another triple says (Brittany, location.administrative_division.country, France). So Brittany is part of France. That means Saint-Malo is in France, so Jacques Cartier was born in France.

Also, there's a triple (Jacques Cartier, people.person.nationality, France). So that directly answers the question. His nationality is France, so he came from France. I think that's the answer. Let me make sure there aren't any conflicting triples. No, it seems consistent. All triples point to France as his country.
```

```Plain Text
{
    "question": "What is the sport played by the team with a mascot known as Carlton the Bear?",
    "question_entities": [
        "Carlton the Bear"
    ],
    "answer": [
        "Ice Hockey"
    ]
}


content:
<REASONING>
# STEP 1: Question Decomposition
1. Identify the team associated with Carlton the Bear
2. Determine the sport played by the identified team

# STEP 2: Knowledge Graph Exploration
## Sub-question 1: Team identification
- Scan triples with Carlton the Bear as subject/object
<TRIPLE_USED>(Carlton the Bear, sports.mascot.team, Toronto Maple Leafs)
<TRIPLE_USED>(Toronto Maple Leafs, sports.sports_team.team_mascot, Carlton the Bear)

## Sub-question 2: Sport determination
- Direct sport triple (highest priority):
<TRIPLE_USED>(Toronto Maple Leafs, sports.sports_team.sport, Ice Hockey)
- Confirmatory triples:
<TRIPLE_USED>(Eastern Conference, sports.sports_league.sport, Ice Hockey)
<TRIPLE_USED>(m.0774ylr, base.events.event_feed.type_of_event, Ice Hockey)

# STEP 3: Knowledge Integration
<KNOWLEDGE_FUSION>(Common sports knowledge confirms Toronto Maple Leafs are historic NHL team)
<KNOWLEDGE_FUSION>(NHL teams exclusively play ice hockey)

# STEP 4: Path Validation
1. Forward chaining confirmed path: Carlton → Maple Leafs → Ice Hockey
2. All key triples used:
   - Mascot-team relationship
   - Team-sport relationship
   - League-sport confirmation
</REASONING>

<ANSWER>
Ice Hockey
</ANSWER>


reasoning_content:
Okay, let's tackle this question step by step. The question is: "What is the sport played by the team with a mascot known as Carlton the Bear?" So, the main entity here is Carlton the Bear.

First, I need to figure out which team Carlton the Bear is associated with. Looking at the KG triples provided, I should check for triples where Carlton the Bear is the subject or object related to a team.

Searching through the triples, I see "(Carlton the Bear, sports.mascot.team, Toronto Maple Leafs)". That's exactly what we need! So, the team is the Toronto Maple Leafs. But wait, there's another triple: "(Toronto Maple Leafs, sports.sports_team.team_mascot, Carlton the Bear)", which confirms the relationship both ways. Good, that's solid.

Now, the next step is to find out what sport the Toronto Maple Leafs play. Let's look for triples that mention Toronto Maple Leafs and their sport. Scanning the list, there's "(Toronto Maple Leafs, sports.sports_team.sport, Ice Hockey)". That's the key. Also supporting this, the conference they're in (Eastern Conference) has "(Eastern Conference, sports.sports_league.sport, Ice Hockey)", which aligns.

Just to ensure there aren't any other sports mentioned, I'll double-check. The other triples about their location, colors, championships, etc., but nothing else about their sport. Their previous names, like Toronto Arenas and Toronto St. Patricks, both have "(Toronto Arenas, sports.sports_team.sport, Ice Hockey)" and "(Toronto St. Patricks, sports.sports_team.sport, Ice Hockey)", so consistently ice hockey.

No conflicting info here. So the sport must be Ice Hockey. And since Ice Hockey is a sport, that answers the question.
```

```Plain Text
{
    "question": "In the year 2013 which man will be the coach of the club that claimed victory in the 2004 NFC Championship Game?",
    "question_entities": [
        "2004 NFC Championship Game"
    ],
    "answer": [
        "Ron Rivera"
    ]
}


content:
<REASONING>
# STEP 1: Question Decomposition
- Identify the club that won the 2004 NFC Championship Game
- Determine who was the head coach of that club in 2013

# STEP 2: Knowledge Graph Exploration
- Core entity: 2004 NFC Championship Game
- Check triples for champion of 2004 NFC Championship Game:
  <TRIPLE_USED>(2004 NFC Championship Game, sports.sports_championship_event.champion, Carolina Panthers)
- Find coaches linked to Carolina Panthers:
  <TRIPLE_USED>(Carolina Panthers, american_football.football_team.current_head_coach, Ron Rivera)
- Verify temporal consistency (2013):
  <KNOWLEDGE_FUSION>(Ron Rivera was head coach of Carolina Panthers from 2011-2019)

# STEP 3: Knowledge Integration
- Used personal knowledge to validate coaching tenure period (2011-2019) covers 2013

# STEP 4: Path Validation
- Forward chaining confirms Carolina Panthers won 2004 NFC Championship
- All critical triples utilized with supporting real-world knowledge
</REASONING>

<ANSWER>
Ron Rivera
</ANSWER>


reasoning_content:
Okay, let's try to tackle this question step by step. The question is: "In the year 2013 which man will be the coach of the club that claimed victory in the 2004 NFC Championship Game?"

First, I need to identify the main entities involved. The question mentions the 2004 NFC Championship Game. The goal is to find the coach of the team that won that game in 2013. Since the event is from 2004, and we're looking for their 2013 coach, this seems like a two-step problem: first find out which team won the 2004 NFC Championship Game, and then find out who their coach was in 2013.

Starting with the 2004 NFC Championship Game. Let's look through the provided KG triples for any information about this event. There are several triples related to different NFC Championship Games. Specifically, I need to look for triples that mention the 2004 NFC Championship Game's champion.

Looking through the list, I find a triple that says (2004 NFC Championship Game, sports.sports_championship_event.champion, Carolina Panthers). Additionally, another triple states (Philadelphia Eagles, sports.sports_team.championships, 2005 NFC Championship Game). Wait, that's about the 2005 game. But there's another triple: (2004 NFC Championship Game, sports.sports_championship_event.champion, Carolina Panthers). So, the 2004 champion was the Carolina Panthers?

Wait, wait. Let me double-check. The triples listed include (The NFC Championship Game, sports.sports_championship.events, 2004 NFC Championship Game) which links the 2004 game as part of the NFC Championship. Then, another triple (2004 NFC Championship Game, sports.sports_championship_event.champion, Carolina Panthers). That seems definitive. However, conflicting information might exist? Let me check again. Oh, also, there's a triple stating (2004 NFC Championship Game, sports.sports_championship_event.runner_up, Philadelphia Eagles). So the Eagles were the runner-up, and the champion is the Panthers.

Now, next step: find the coach of the Carolina Panthers in 2013. Looking through the triples, there's (Carolina Panthers, american_football.football_team.current_head_coach, Ron Rivera). Is there any information about the timeframe when Ron Rivera was the coach? The triples mention "current_head_coach" but without dates. However, another triple (Ron Rivera, american_football.football_coach.current_team_head_coached, Carolina Panthers). Depending on when the KG was last updated, "current" in the triples could refer to the time the data was recorded. Since the question is about 2013, I need to verify if Ron Rivera was the coach then.

In real-life knowledge, Ron Rivera became the head coach of the Panthers in 2011 and was there until 2019. So in 2013, he was indeed the coach. The KG here lists him as the current_head_coach, but if the data is up to date post-2013, that would still confirm it. Assuming the KG's information aligns with real-world timelines, the answer should be Ron Rivera.

Wait, but let me make sure there aren't conflicting triples. Another triple: (Philadelphia Eagles, american_football.football_team.current_head_coach, Chip Kelly). But the Eagles weren't the 2004 champions; the Panthers were. So the answer remains Rivera.
```



```Plain Text
# deepsek-r1 api上下文长度32k，存在context length超过长度的情况，绝大部分token是三元组

openai.BadRequestError: Error code: 400 - {'object': 'error', 'message': "This model's maximum context length is 32768 tokens. However, you requested 35620 tokens in the messages, Please reduce the length of the messages.", 'type': 'BadRequestError', 'param': None, 'code': 400}
```



## Todo

改进推理模式，一次性放子图中的所有三元组太多了

根据子问题进度分批提供三元组，LLM根据推理进度判断是否需要新的三元组，并提供用来检索三元组的实体列表



- reasoning\_prompt优化

    - 处理多个answer的情况

        - 在graph上继续探索，找到其他可能的answer

    - 处理子图三元组超过context length的情况

        - sub\-graph搜索与sub\-question回答交替进行

        - 更新sub\-question状态，存档所有使用到的sub\-graph，记录之前所有的探索和推理路径



## Structure\-RL

构造结构化KG推理数据集，大模型基于提供的KG子图进行探索和推理，完成QA任务

数据集构造思路

根据问题跳数的增加，构造不同跳数路径的子图，同时子图包含适当的无关/干扰三元组

根据子图复杂度的增加，构造包含不同三元组数量的子图，确保包含正确推理路径三元组

数据集来源选择

webqsp/cwq/grailqa，基于freebase的1\-4跳问题

metaqa\-1/2/3，基于movie的1\-3跳问题

KQA Pro，基于wikidata的1\-10跳问题

泛化测试数据集

TriviaQA、PopQA

GSM8K、MATH

数据集处理细节

abandon\_relation加一条：kg\.object\_profile\.prominent\_type

或者，filtered\_lines中包含两个relation的丢弃

训练模型选择

Qwen2\.5\-7B、Qwen2\.5\-Math\-7B、Qwen2\.5\-7B\-Instruct

在这里添加reward\_function注册，verl/utils/reward\_score/\_\_init\_\_\.py



Origin，

webqsp，3098，1639

cwq，27639，3519，3531

RoG，webqsp的train和test分别有26个和11个空的answer，去除，其他保留

webqsp，2826\+246，1628

cwq，27639，3519，3531

PoG，

webqsp，3098，1639

cwq，27734，nan，3531



> 内部图片链接已移除。



**快速构造：对RoG预处理后的数据集进行二次处理**

使用networkx库从数据集graph字段中提取triple\_path并保存

如果triple\_path为空，对三元组进行排序，取前1000个**【更新：取前1200个】**

如果triple\_path不为空，有n个，对三元组进行排序，取前1000\-n个并加上n个path三元组

再对处理保留的三元组**进行shuffle【Todo】**，得到最终的三元组上下文

**注：webqsp有一个问题的graph为空，已经丢弃**

**Todo：处理优化，rerank的时候同时传入question和answer【暂不】**

**Todo：额外处理 triple\_path 为空的数据，额外处理两跳以上数据**

Todo：修改max\_response\_length为2k；修改graph\_length为**7\.2k**，即可保持max\_prompt\_length为8k

**webqsp数据集构造流程**

从原始 \[sparql\] 中提取 topic\_entity 和 inference\_chain，从数据集 \[Answers\] 中提取 answers

对 topic\_entity 和 inference\_chain 进行二次过滤，根据 inference\_chain 确定问题的跳数

从原始 \[sparql\] 中提取 answer\_id，使用 networks 库提取从 topic\_entity 到 answer\_id 的 triple\_chain

根据 triple\_chain 确定问题的跳数，后续根据问题跳数决定 answers 子图检索跳数

查询 topic\_entity 的两跳子图【更新：使用数据集自带的 topic\_entity】

使用 networkx 库，提取从 answers 到 topic\_entity 的 triple\_chain，作为 key\_graph

两路同时读取 ds\_split\.jsonl 和 ds\_inv\_split\.jsonl，**考虑逐行读取jsonl**

预先从原始数据中读取 topic\_entity\_graph 和 answer\_entity\_graph

查询数据库转换成 topic\_entity\_name 和 answer\_entity\_name（暂不）

预先从 sparql 中处理 max\_hop，作为 get\_truth\_paths 的新增参数 cutoff

先缩减每一跳的子图规模，筛选 relation，控制在20个，同时每跳加入 inference\_chain 中的 relation

根据 answers 构造 correct reward，根据 inference\_chain 构造 path reward

Done：训练集中 topic\_entity \(TopicEntityMid\) 为 null 的数据直接丢掉

**cwq数据集构造流程**

从原始 \[sparql\] 中提取 topic\_entity 和 inference\_chain，从数据集 \[answers\] 中提取 answers

对 topic\_entity 和 inference\_chain 进行二次过滤，使用新的 abandon\_relation

从原始 \[sparql\] 中提取 asnwer\_id，使用 networks 库提取从 topic\_entity 到 answer\_id 的 triple\_chain

根据 triple\_chain 确定问题的跳数，后续根据问题跳数决定 answers 子图检索跳数

查询 topic\_entity 的两跳子图

根据问题的跳数，查询 answers 的一跳或者两跳子图

如何处理 answer 很多的情况，比如 10 个及以上，第二跳太多了？

方案：根据 triple\_chain，对 answers 的一跳子图进行 relation prune；没裁剪直接读取

使用 networkx 库，提取从 answers 到 topic\_entity 的 triple\_chain，作为 key\_graph

两路同时读取 ds\_split\.jsonl 和 da\_inv\_split\.jsonl，**考虑逐行读取jsonl**

预先从原始数据中读取 topic\_entity\_graph  和 answer\_entity\_graph

查询数据库转换成 topic\_entity\_name 和 answer\_entity\_name（暂不）

预先从 sparql 中处理 max\_hop，作为 get\_truth\_paths 的新增参数 cutoff

先缩减每一跳的子图规模，筛选 relation，控制在20个，同时每跳加入 inference\_chain 中的 relation

根据 answers 构造 correct reward，根据 inference\_chain 构造 path reward

Done：训练集中没有 topic\_entity 的数据直接丢掉

> 内部图片链接已移除。

```Markdown
Golden answers: ['Saudi Arabia', 'Angola', 'Libya', 'Singapore', 'Tunisia', 'United Arab Emirates', 'Jordan']
Extracted answer is not None: Angola
Solution string: To answer the question "Which countries does Portugal trade with, where the population is less than 5,362,099?", we need to explore the information provided in the knowledge graph.

From the graph, we know that Portugal is a statistical region and that its statistical region includes countries where the population is less than 5,362,099. However, the knowledge graph does not directly tell us which specific countries fit this description. To find the answer, we need to use our general knowledge.

In general knowledge, the only country that fits this description is Angola. Angola is a country in southern Africa with a population that is less than 5,362,099.

Therefore, the country that Portugal trades with, where the population is less than 5,362,099, is Angola.

<thought> We started by looking for Portugal and found that it is a statistical region, but then we found that the only country that fits the description is Angola. </thought>
<answer> Angola </answer>
--------------------------------
Golden answers: ['Seth MacFarlane']
Extracted answer is not None: Seth MacFarlane
Solution string: To answer the question "who plays stewie griffin on family guy?", we need to explore the information provided in the knowledge graph.

From the graph, we know that Stewie Griffin is a character from the family guy television series and that his role is part of the family guy cast.

However, the knowledge graph does not directly tell us the actor who portrays Stewie Griffin. To find this information, we need to use our general knowledge.

In the world of entertainment, the most obvious connection is to the character of Stewie Griffin and the actor who has portrayed him. The most likely answer is that Stewie Griffin is voiced by the actor Seth MacFarlane.

<thought> We started by looking for Stewie Griffin and found that he is a character from the family guy television series. However, we didn't find the actor who portrays him. But in the world of entertainment, the most likely answer is that Stewie Griffin is voiced by the actor Seth MacFarlane. </thought>
<answer> Seth MacFarlane </answer>
--------------------------------
Golden answers: ['Greenlandic language', 'Danish Language', 'Faroese', 'German Language']
Extracted answer is not None: Danish Language
Solution string: To answer the question "People from the country that contains Greenland speak what languages?", we need to explore the information provided in the knowledge graph.

From the graph, we know that Greenland is a statistical region and that the language spoken in Greenland is Danish. This information directly tells us that the people of Greenland primarily speak the Danish language.

<thought> We started by looking for Greenland and found that it is a statistical region and that the language spoken is Danish. </thought>
<answer> Danish Language </answer>
--------------------------------
Golden answers: ['Mike Tomlin']
Extracted answer is not None: Mike Tomlin
Solution string: To answer the question "Who is the man in the position of head coach of the squad that earned victory in the 2011 AFC Championship Game?", we need to explore the information provided in the knowledge graph.

From the graph, we know that the AFC Championship Game is the championship game of the American Football Conference in the NFL. The most relevant information is that the Pittsburgh Steelers won the 2011 AFC Championship Game. In the graph, the most direct connection is that the Pittsburgh Steelers are an American football team, and the most relevant information is that the head coach of the Pittsburgh Steelers is the person we need to find.

In the knowledge graph, the most relevant information is that the Pittsburgh Steelers are part of the American Football Conference and that the head coach of the Pittsburgh Steelers is the man we need to find. However, the most direct connection is that the Pittsburgh Steelers are part of the American Football Conference and that the head coach of the Pittsburgh Steelers is the man we need to find. The most direct connection is that the Pittsburgh Steelers are part of the American Football Conference and that the head coach of the Pittsburgh Steelers is the man we need to find.

Therefore, the man in the position of head coach of the squad that earned victory in the 2011 AFC Championship Game is Mike Tomlin.

<thought> We started by looking for the AFC Championship Game and found that it is the championship game of the American Football Conference. Then, we found that the Pittsburgh Steelers won the game and that they are part of the American Football Conference. Finally, we found that the head coach of the Pittsburgh Steelers is Mike Tomlin. </thought>
<answer> Mike Tomlin </answer>
```



# Papers

KGQA

\[ToG\][Think\-on\-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph](https://openreview.net/pdf?id=nnVO1PvbTv)

\[RoG\][Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning](https://openreview.net/pdf?id=ZGNWW7xZ6Q)

\[PoG\][Plan\-on\-Graph: Self\-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs](https://openreview.net/pdf?id=CwCUEr6wO5)

[Retrieval and Reasoning on KGs: Integrate Knowledge Graphs into Large Language Models for Complex Question Answering](https://aclanthology.org/2024.findings-emnlp.446.pdf)

[KnowPath: Knowledge\-enhanced Reasoning via LLM\-generated Inference Paths over Knowledge Graphs](https://arxiv.org/pdf/2502.12029?)

MHQA

[Making Long\-Context Language Models Better Multi\-Hop Reasoners](https://aclanthology.org/2024.acl-long.135.pdf)

[ReARTeR: Retrieval\-Augmented Reasoning with Trustworthy Process Rewarding](https://arxiv.org/pdf/2501.07861)

[AMOR: A Recipe for Building Adaptable Modular Knowledge Agents Through Process Feedback](https://arxiv.org/pdf/2402.01469)

[Chain\-of\-Thought Matters: Improving Long\-Context Language Models with Reasoning Path Supervision](https://arxiv.org/pdf/2502.20790)



与上周讨论内容比较相关的两篇论文

[SymAgent: A Neural\-Symbolic Self\-Learning Agent Framework for Complex Reasoning over Knowledge Graphs](https://arxiv.org/pdf/2502.03283)

在KG上限制LLMs的动作空间进行优化，评测QA任务

[Generate\-on\-Graph: Treat LLM as both Agent and KG for Incomplete Knowledge Graph Question Answering](https://aclanthology.org/2024.emnlp-main.1023.pdf)

针对不完整KG场景，提升LLM在QA任务上的鲁棒性



# Datasets

## Knowledge Graph QA \(KGQA\)

WebQSP

CWQ

MetaQA

GrailQA

KQA Pro

```json
{
    "id": "WebQTest-832_c334509bb5e02cacae1ba2e80c176499",
    "question": "Lou Seal is the mascot for the team that last won the World Series when?",
    "q_entity": [
        "Lou Seal"
    ],
    "a_entity": [
        "2014 World Series"
    ],
    "graph": [
        [
            "Mascot",
            "type.type.expected_by",
            "sports_team_mascot"
        ],
        [
            "San Francisco Giants",
            "baseball.baseball_team.team_stats",
            "m.05n69q3"
        ],
        [
            "Mascot",
            "type.type.properties",
            "Team"
        ],
        [
            "San Francisco Giants",
            "base.schemastaging.sports_team_extra.training_ground",
            "m.0k079pf"
        ],
        [
            "m.0k079rd",
            "base.schemastaging.team_training_ground_relationship.team",
            "San Francisco Giants"
        ]
    ]
}
```



## Multi\-Hop QA \(MHQA\)

MuSiQue

HotpotQA

2WikiMultiHopQA

Bamboogle

```Plain Text
{
    "question_id": "2hop__482757_12019",
    "question_text": "When was the institute that owned The Collegian founded?",
    "contexts": [
        {
            "idx": 3,
            "paragraph_text": "Bankhaus Lampe is a private bank in Germany, founded in 1852 and headquartered in Bielefeld. It is wholly owned by the Oetker Group. The bank owns 50% of Universal Investment.",
            "title": "Bankhaus Lampe",
            "is_supporting": false
        },
        {
            "idx": 4,
            "paragraph_text": "Publix Super Markets, Inc., commonly known as Publix, is an employee - owned, American supermarket chain headquartered in Lakeland, Florida. Founded in 1930 by George W. Jenkins, Publix is a private corporation that is wholly owned by present and past employees. It is considered the largest employee - owned company in the world. Publix operates throughout the Southeastern United States, with locations in Florida (785), Georgia (186), Alabama (68), South Carolina (58), Tennessee (42), North Carolina (35), and Virginia (8).",
            "title": "Publix",
            "is_supporting": false
        },
        {
            "idx": 5,
            "paragraph_text": "The Collegian is the bi-weekly official student publication of Houston Baptist University in Houston, Texas. It was founded in 1963 as a newsletter, and adopted the newspaper format in 1990.",
            "title": "The Collegian (Houston Baptist University)",
            "is_supporting": true
        },
        {
            "idx": 8,
            "paragraph_text": "Renaissance Broadcasting, founded in 1982 by Michael Finkelstein, was a company that owned several UHF television stations, it was sold to Tribune Broadcasting in 1997. The company was headquartered in Greenwich, Connecticut.",
            "title": "Renaissance Broadcasting",
            "is_supporting": false
        },
        {
            "idx": 9,
            "paragraph_text": "Several private institutions of higher learning\u2014ranging from liberal arts colleges, such as The University of St. Thomas, Houston's only Catholic university, to Rice University, the nationally recognized research university\u2014are located within the city. Rice, with a total enrollment of slightly more than 6,000 students, has a number of distinguished graduate programs and research institutes, such as the James A. Baker Institute for Public Policy. Houston Baptist University, affiliated with the Baptist General Convention of Texas, offers bachelor's and graduate degrees. It was founded in 1960 and is located in the Sharpstown area in Southwest Houston.",
            "title": "Houston",
            "is_supporting": true
        }
],
    "answers_objects": [
        {
            "number": "",
            "date": {
                "day": "",
                "month": "",
                "year": ""
            },
            "spans": [
                "1960"
            ]
        }
    ],
    "reasoning_steps": [
        "The Collegian >> owned by >>>> Houston Baptist University",
        "When was Houston Baptist University founded? >>>> 1960"
    ]
}
```



## KnowLogic

[KnowLogic: A Benchmark for Commonsense Reasoning via Knowledge\-Driven Data Synthesis](https://arxiv.org/pdf/2503.06218)



# Data Correction

"WebQTest\-31"：question中where应该是when
