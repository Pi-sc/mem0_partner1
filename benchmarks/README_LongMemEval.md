# LongMemEval 独立跑分模式

官方来源：[GitHub](https://github.com/xiaowu0162/LongMemEval) · [清理版数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)。这里只适配 LongMemEval，不包含 LoCoMo。

本次完整下载校验、同题真实试跑与对照结果见 [验证记录](VERIFICATION_LongMemEval.md)。

## 文件位置与版本

- 官方仓库：`benchmarks/LongMemEval/`，保留原代码，Git 提交 `9e0b455f4ef0e2ab8f2e582289761153549043fc`。
- 数据：该仓库的 `data/` 内，遵循官方 README 的位置与三个下载文件名。
- 数据固定为清理版提交 `98d7416c24c778c2fee6e6f3006e7a073259d48f`，不是论文最初发布的未清理版。
- `download_manifest.json` 记录地址、字节数和 SHA-256，下载程序核对官方 LFS 校验值；大文件支持分块续传。
- Oracle 约 15.4 MB；S 约 277 MB；M 约 2.74 GB。每个版本各 500 道题，S/M 是相同题目在不同长度历史下的测试，不能当作 1,500 个独立问题。
- 适配代码在 `benchmarking/`，入口在项目根目录 `longmemeval_benchmark.py`；结果在 `benchmarks/runs/`。

官方原项目的 GPU 检索器和 Linux shell 实验环境没有全部安装。本模式使用伴侣已有的 DeepSeek、Ollama 和 Mem0 SDK，在 Windows 的现有 `.venv` 下执行。

## 评测的三个模式

| 模式 | 输入与流程 | 可以说明什么 |
| --- | --- | --- |
| `mem0` | 全部所选历史 → SDK `add(infer=True)` → SDK `search()` → DeepSeek 回答 | 当前伴侣记忆链路的端到端表现 |
| `oracle` | 官方 Oracle 文件中的全部证据会话直接交给 DeepSeek | 减少检索问题后的阅读与推理诊断；必须配 `--dataset oracle` |
| `no-memory` | 不读历史，不调用 Mem0，仅给问题与提问日期 | 无记忆对照，帮助观察依靠常识猜答案的情况 |

`--dataset oracle --mode mem0` 则是对证据会话建 Mem0 记忆的调试测试，和直接阅读的 `oracle` 模式不同。Oracle 调试成功不能代表 S/M 长历史测试成功。

本模式使用正式 QA 提示词，去掉伴侣的昵称、表情和简短聊天风格；回答时只提供检索记忆与提问日期，不额外塞最近 16 条原聊天。这有助于测出长期记忆的贡献，因此分数也不代表整个人工智能伴侣的所有交互能力。

## 关键适配规则

1. 每道题创建独立 Qdrant 目录、SQLite 历史库和用户空间；SDK 主目录也位于本次跑分目录。正常聊天的 `data/`、`sessions/` 不会被读写。不要在聊天进程中导入并启动跑分；使用独立终端运行。
2. 按 `haystack_dates` 稳定排序，逐轮将原始用户和助手消息共同交给 SDK。保留完整历史与孤立消息；`--limit` 只减少测试题数，不截断单题历史。不会生成替代助手回复。
3. 入库只传 `role/content` 及普通来源日期元数据；`answer`、`has_answer`、`answer_session_ids` 等标准答案和证据标签只参与后续评分，不进入提取、检索或回答提示。
4. 当前本地 SDK 开源版拒绝 `add(timestamp=...)`。评测进程在单次写入期间适配 SDK 原有提示词生成函数的 `timestamp/current_date`，将“观察日期”和“当前日期”设为原会话日期，同时保存日期元数据。不修改 SDK 源文件、提取规则或 `infer=True` 行为。这是回放历史的日期适配，不是论文中的时间范围过滤检索。
5. 当前 SDK 是 V3 ADD-only 默认提取链路，和 2025 年 Mem0 论文中的自动 ADD/UPDATE/DELETE 流程不同。检索使用该 SDK 实际的默认算法与 `top_k=8, threshold=0.1`，不是重新写一个向量检索器。维持伴侣当前 `deepseek-chat + nomic-embed-text(768)` 配置；模型名的服务端版本可能变化，不能视作论文原模型复现。
6. 技术错误、提取 JSON 不合法、模型因 token 上限被截断时，该题记录失败，不当作成功答案。进度保存在 `completed/` 的原子检查点。续跑跳过完成题；未完成题用全新目录重新建立完整记忆，避免半成品污染。相同目录禁止并发运行。
7. 官方 S 数据中有少量重复 session ID，以及部分晚于提问时间的会话时间戳。本适配保留官方原文件和全部会话，检查报告会显示数量，不自行去重或按提问日期删会话。研究时应记录数据版本并进一步审计这些情况。
8. 测试记忆默认只保留最近使用的 10 次 `mem0` 实验。这里一次实验指一次 `run` 的输出目录，不是 10 道题或 10 条事实。开始回放和退出实验时自动清理更早实验的 Qdrant、SQLite（含 WAL/SHM）和 SDK 主目录，保留回答、评分、断点与诊断日志。正在运行的实验不会被删除；若文件被占用，终端会提示延后清理。Oracle/无记忆实验不占记忆保留名额。
9. `--output` 必须位于本项目的 `benchmarks/runs/` 内，支持其子目录；拒绝符号链接和 Windows junction，避免清理触及其他目录。旧实验即使记忆库已清理，仍能 `score`；续跑跳过完成题，未完成题重新建库。本次纯存储管理更新允许已记录旧版本指纹的实验兼容续跑，其他模型/数据/SDK/推理代码变化仍拒绝混用。

## 开始使用

在项目目录打开 PowerShell，沿用聊天程序使用的 `DEEPSEEK_API_KEY` 环境变量，并先启动 Ollama、确认 `nomic-embed-text` 已安装。不要把密钥写入代码或结果文件。

```powershell
cd "D:\My project\AgentProject\mem0_partner1"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe longmemeval_benchmark.py --help
.\.venv\Scripts\python.exe longmemeval_benchmark.py inspect --dataset s
```

先计算 1 题的完整历史工作量，不调用模型：

```powershell
.\.venv\Scripts\python.exe longmemeval_benchmark.py run --dataset s --mode mem0 --limit 1 --dry-run
```

PyCharm 的「终端」也要输入完整命令；如果 `.venv` 已激活，可把解释器路径缩写为 `python`。仅在运行配置的「脚本参数」栏里，才从 `run --dataset ...` 开始填写。例如预览 10 题：

```powershell
python longmemeval_benchmark.py run --dataset s --limit 10 --seed 42 --dry-run --max-write-calls 1000
```

`--dry-run` 不运行模型、不创建实验记忆库；它会显示完整工作量，即使超过 1000。去掉它正式运行时，预算不足会拒绝执行，需按预览数量提高 `--max-write-calls`。

当前固定 S 数据、`--limit 10 --seed 42` 的预览是 487 个会话、2454 次历史写入。正式运行可使用：

```powershell
python longmemeval_benchmark.py run --dataset s --limit 10 --seed 42 --max-write-calls 3000 --output benchmarks/runs/s_10_01
```

`s_10_01` 应是新的输出目录；同目录恢复中断则追加 `--resume`。

再执行相同样本，结果保存在明确的新目录：

```powershell
.\.venv\Scripts\python.exe longmemeval_benchmark.py run --dataset s --mode mem0 --limit 1 --max-write-calls 300 --output benchmarks/runs/my_s_01
```

恢复中断时，保持原参数和输出目录，并加 `--resume`：

```powershell
.\.venv\Scripts\python.exe longmemeval_benchmark.py run --dataset s --mode mem0 --limit 1 --max-write-calls 300 --output benchmarks/runs/my_s_01 --resume
```

默认 `--limit 1`，随机种子为 42；先按题型和拒答分组，再依次选取，可复现。相同数据、seed 和 limit 会选出相同问题。增加 `--limit 7` 可开始覆盖六种题型与拒答；正式成绩需要 `--limit 0` 测完 500 题。`--question-id ID` 支持指定调试题，可多次传入；这类定向样本不能包装成代表性的整体分数。

跨模式、跨数据版本作小样本对照时，请用 `run_manifest.json` 中相同的题目 ID，并通过 `--question-id` 明确指定。不同数据文件的顺序可能不同，相同 seed/limit 不保证跨文件选出同一题。实际验证的 S 样本是 `b6019101`，对应的 Oracle 和无记忆对照均使用同一个 ID。

完整 S 逐轮提取约需 124,358 次逻辑写入，此外还要回答与评分，因此请先检查 `--dry-run`。`--max-write-calls` 默认 1,000，超额会在付费调用前拒绝执行；可明确提高预算，`0` 表示主动取消该上限。它限制逻辑写入数，不是实际请求重试次数、费用或 token 上限。

快速证据阅读诊断、无记忆对照与 M 工作量检查：

```powershell
.\.venv\Scripts\python.exe longmemeval_benchmark.py run --dataset oracle --mode oracle --limit 1 --output benchmarks/runs/my_oracle_01
.\.venv\Scripts\python.exe longmemeval_benchmark.py run --dataset s --mode no-memory --limit 1 --output benchmarks/runs/my_nomemory_01
.\.venv\Scripts\python.exe longmemeval_benchmark.py run --dataset m --mode mem0 --limit 1 --dry-run
```

更换机器后可恢复下载，三个文件沿用官方地址：

```powershell
git clone --depth 1 https://github.com/xiaowu0162/LongMemEval.git benchmarks/LongMemEval
.\.venv\Scripts\python.exe longmemeval_benchmark.py download
```

如果仓库已存在，不要再执行 clone；只执行 download 即可。下载大型 M 数据需要时间与磁盘空间。

## QA 评分与可比性

标准裁判是官方脚本中的 `gpt-4o-2024-08-06`。设置 `OPENAI_API_KEY` 后运行：

```powershell
.\.venv\Scripts\python.exe longmemeval_benchmark.py score --run-dir benchmarks/runs/my_s_01 --judge gpt-4o
```

该命令直接加载官方 `evaluate_qa.py` 的 `get_anscheck_prompt()`，保留不同题型的评分规则、日期差一天的容忍规则、拒答规则，以及 `temperature=0, max_tokens=10` 和官方 yes/no 判定方式。原始官方代码保持不变。评分逐题落盘，重新执行会复用已有判断。

如果只使用现有 DeepSeek 密钥，可以明确选择替代裁判：

```powershell
.\.venv\Scripts\python.exe longmemeval_benchmark.py score --run-dir benchmarks/runs/my_s_01 --judge deepseek
```

仍使用官方提示词，但裁判换成 `deepseek-chat`，报告会标记 `official_paper_judge=false`。这适合自测，不能称为论文标准 GPT-4o 分数。`gpt-4o-mini` 同样属于另一裁判选择。

预测文件严格使用官方的两列 JSONL 格式：`question_id` 与 `hypothesis`。也可直接运行未修改的官方评分脚本：

```powershell
# Windows 上启用 UTF-8，避免官方脚本使用默认编码读取 JSON。
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe benchmarks/LongMemEval/src/evaluation/evaluate_qa.py gpt-4o benchmarks/runs/my_s_01/predictions.jsonl benchmarks/LongMemEval/data/longmemeval_oracle.json
# 原脚本会重新评分所有预测；不要仅为重看分数重复调用。
.\.venv\Scripts\python.exe benchmarks/LongMemEval/src/evaluation/print_qa_metrics.py benchmarks/runs/my_s_01/predictions.jsonl.eval-results-gpt-4o benchmarks/LongMemEval/data/longmemeval_oracle.json
```

官方 README 的 `.log` 文件名与当前代码不一致；此处采用当前代码的实际 `.eval-results-gpt-4o` 文件名。`print_qa_metrics.py` 实际只需要两个位置参数，并且断言裁判模型是 GPT-4o。引用 Oracle 文件作标准答案是官方推荐方式；本项目包装的 score 命令则使用本次跑分的数据文件。

## 结果怎么看

自动保留策略无需增加运行参数。也可以随时预览或手动执行同一清理策略，不调用模型：

```powershell
python longmemeval_benchmark.py cleanup --dry-run
python longmemeval_benchmark.py cleanup
```

清理不会删除实验的结果目录或正式聊天记忆。结果和日志仍会保留，因此这里只限制测试数据库的累积，不限制全部结果文件占用的空间。

| 文件 | 内容 |
| --- | --- |
| `run_manifest.json` | 数据 SHA-256、选题、模型与检索参数、SDK/适配代码指纹 |
| `memory_retention.json` | 最近使用时间、保留 10 次实验的规则及记忆库是否已清理 |
| `predictions.jsonl` | 官方兼容的逐题回答 |
| `diagnostics.jsonl` | 写入数量、抽取记忆数量、检索数、阶段耗时与 API 返回的 token usage |
| `cases/.../writes.jsonl` | 每轮 Mem0 的记忆写入结果与来源 |
| `cases/.../reader_input.json` | 实际检索结果及回答提示词，便于排查抽取、检索还是推理失误 |
| `errors.jsonl` | 失败题与原因；重试后旧错误仍保留，最终缺失题以 status 为准 |
| `status.json` | 完成数量、缺失题和部分调用统计 |
| `predictions.jsonl.eval-results-裁判名` | 兼容官方结构的判断日志 |
| `metrics_裁判名.json` | 总体准确率、题型准确率、拒答准确率和覆盖率 |

`overall_accuracy_on_scored` 是实际完成裁判的题目上的正确率；`coverage` 表示选定题中已完成评分的比例；`correct_over_selected` 的分母是所有选定题，避免遗漏失败题导致分数虚高。`task_averaged_accuracy_on_observed_tasks` 是已覆盖题型准确率的平均；少样本缺少某些题型时不能视作论文完整的宏平均。

token usage 来自 API 返回值；不把字符数假装成 token 数，也不估造本地 embedding token 消耗或价格。status 的 token 统计只涵盖成功题，失败/中断产生的费用可能另有发生。

暂不报告论文中的 Recall@k/NDCG@k：当前 Mem0 检索的是提取后的事实，一条事实可能揉合多轮内容，其来源元数据不能直接当作精确证据标注。先用端到端 QA 准确率和实际检索日志诊断，后续建立可靠事实溯源后再补检索指标。

默认当前环境缺少 SDK 可选的 spaCy/fastembed，SDK 会按现有配置降级相应实体、词法或 BM25 功能，和伴侣当前安装环境一致；本次没有为了提分擅自安装或切换这些功能。评测时应记录这点，后续作为单独消融实验比较。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest test_memory_lifecycle test_longmemeval_benchmark test_benchmark_retention -v
```

离线检查覆盖标签隔离、历史排序、日期适配、SDK 推断参数、QA 不回写、截断失败处理、来源路径、官方评分提示词、失败覆盖率、可复现选题与预算拒绝，以及最近 10 次实验的清理、结果保留、并发锁保护和路径边界；不调用付费 API。
