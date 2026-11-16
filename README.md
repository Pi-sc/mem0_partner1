# Mem0 Partner 1

新增独立 [LongMemEval 跑分模式](benchmarks/README_LongMemEval.md)：官方仓库和三个数据版本位于 `benchmarks/LongMemEval/`，通过 `longmemeval_benchmark.py` 回放历史、回答和评分，结果与聊天记忆隔离。先运行 `python longmemeval_benchmark.py run --dataset s --limit 1 --dry-run` 查看工作量；完整步骤见链接。

这是官方 Mem0 链路版本。应用层直接使用：

```python
from mem0 import Memory
```

不再自研 `MemoryStore` / `MemoryManager`，也不在应用层直接操作 `QdrantClient` 或 `sqlite3`。

## 架构

```text
Streamlit AI 伴侣
  |
  +-- DeepSeek: 聊天回复
  |
  `-- mem0.Memory
        |
        +-- DeepSeek: SDK 默认记忆提取
        +-- Ollama nomic-embed-text: 本地 embedding
        +-- Qdrant local: 向量检索
        `-- SQLite: Mem0 ADD/UPDATE/DELETE history
```

聊天结束后，`memory_lifecycle.process_turn()` 只将本轮原始用户消息和助手回复交给官方 `Memory.add(messages, user_id=..., metadata=..., infer=True)`。应用不再自行提取事实、判断遗忘请求、决定增删改、限制事实长度或数量，也不覆盖 SDK 的记忆提取提示词。近期消息上下文和相关旧记忆的收集由 SDK 内部负责。

`infer=True` 是 SDK 默认值，启用其自动记忆提取流程；`infer=False` 会跳过推断，将传入文本直接存为记忆，并不表示“用户设定优先”。侧栏手动添加也使用 `infer=True`，由 SDK 决定哪些内容值得保存。

本地源码的默认写入采用 V3 ADD-only 流程：从用户与助手消息提取新记忆、去重并累积信息，不在自动写入时执行 UPDATE 或 DELETE。因此，在聊天中说“忘记某事”不保证实际删除旧条目；实际修改或遗忘可通过侧栏 UPDATE、DELETE 操作完成，它们直接调用官方 SDK API。该行为与 2025 年论文的四类自动操作有所不同。原有记忆保持原样，新对话才采用新流程。

回答前仍通过 `Memory.search()` 按 `user_id` 检索最多 8 条相关记忆，连同最近聊天交给 DeepSeek 回答。

## 当前配置

- LLM provider: `deepseek`
- LLM model: `deepseek-chat`
- Embedder provider: `ollama`
- Embedder model: `nomic-embed-text`
- Embedding dims: `768`
- Vector store: `qdrant` local path
- History DB: `data/mem0_history.db`

## Ollama

模型目录：

```powershell
D:\javaweb-soft\ollama_models
```

启动 Ollama：

```powershell
D:\javaweb-soft\Ollama\ollama.exe serve
```

确认模型：

```powershell
D:\javaweb-soft\Ollama\ollama.exe list
```

需要的模型：

```powershell
D:\javaweb-soft\Ollama\ollama.exe pull nomic-embed-text
```

## 运行

官方 Mem0 v2.1.0 源码放在项目根目录的 `mem0` 文件夹，并通过 `requirements.txt` 以可编辑模式安装到本项目的 `.venv`。如果移动或删除该源码目录，需要在新位置重新安装。

```powershell
cd "D:\My project\AgentProject\mem0_partner1"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

PyCharm 的项目解释器应选 `D:\My project\AgentProject\mem0_partner1\.venv\Scripts\python.exe`。`app.py` 只导入 `mem0`，解释器路径由 PyCharm 的项目设置或启动命令决定。

如果看到 Qdrant Local 的 `already accessed by another instance`，说明还有旧的 Streamlit 进程没关，在旧终端按 `Ctrl + C` 停掉即可。

修改记忆配置后，先停止现有 Streamlit 进程再启动，以释放缓存的 Qdrant 客户端。

## 检查

```powershell
.\.venv\Scripts\python.exe -m unittest test_memory_lifecycle -v
```

这些测试执行本地 SDK 的实际 `add()` 推断分支，但以替身隔离 LLM、embedding 和存储，不调用付费 API、不访问现有记忆数据库。
