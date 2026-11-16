# LongMemEval 本次落地验证

验证日期：2026-10-10。完整官方数据已就位，独立命令行模式可运行。

## 数据检查

| 数据版本 | 题数 | 每题会话数范围 | 全集逐轮写入数量 | 文件大小 |
| --- | ---: | ---: | ---: | ---: |
| Oracle | 500 | 1–6 | 5,484 | 15.4 MB |
| S | 500 | 38–62 | 124,358 | 277.4 MB |
| M | 500 | 460–490 | 1,233,371 | 2737.1 MB |

三个文件均按官方 LFS SHA-256 校验通过，数据版本固定为 `98d7416c24c778c2fee6e6f3006e7a073259d48f`。检查只建立旁边的索引文件，没有改写原始数据。

## 同题真实试跑

问题 ID：`b6019101`，题型为 knowledge-update；问题是最近三个月看过多少部 MCU 电影。以下三个模式使用同一道题。

| 模式 | 数据 | 历史写入 | 模型回答 | DeepSeek 替代裁判 |
| --- | --- | ---: | --- | --- |
| Mem0 | S | 256 | Based on the evidence, you watched **5 MCU films** in the last 3 months (around March–May 2023). | 正确（1/1） |
| Oracle 证据直读 | Oracle | 0 | Based on the evidence, you said on 2023/05/25 that you had watched 12 films in the last 3 months, including 5 MCU films. Earlier, on 2023/05/20, you mentioned having watched 4 MCU films in the last 3 months. The most recent statement is 5 MCU films. | 正确（1/1） |
| 无记忆 | S | 0 | I don't know. There is no past conversation evidence available, so I can't determine how many MCU films you watched in the last 3 months. | 未答出（0/1） |

S 样本保留全部 49 次会话，完成 256 轮 SDK 推断写入，抽取 820 条记忆，检索 8 条。索引约 9.78 分钟，检索约 0.038 秒，回答约 1.34 秒。

该成功题的提取 API 返回输入 2,379,144 tokens、输出 55,781 tokens；回答输入 862 tokens。不根据这些数据估造账单价格。

这些只是一题的流程验证，不是 500 题整体准确率。裁判是 DeepSeek，复用了官方提示词；当前没有 OPENAI_API_KEY，因此没有运行论文标准 GPT-4o 裁判。M 版完成了下载、校验与工作量预览，没有启动其付费批量模型评测。

## 工程验证

- 17 项离线测试全部通过。
- 另完成 1 题 Oracle 历史上的 Mem0 写入、检索、回答和评分。
- 已完成样本的实际续跑复用了检查点，没有重新调用回答接口。
- 日常聊天 app.py 与 memory_lifecycle.py 保持原样；评测使用独立存储。

下一步可先覆盖六种题型与拒答的小样本，再考虑全量 S；M 逐轮写入远多于 S。跨模式比较时请明确指定相同 question_id。

操作命令、评分协议与结果文件说明见 [使用说明](README_LongMemEval.md)。
