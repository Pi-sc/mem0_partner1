测试评价输出字段	通俗含义
Judged …: True	这道题被裁判判为正确
selected: 2	本轮选了 2 题
scored: 2	已完成 2 题的评分
correct: 2	其中 2 题正确
coverage: 1.0	本轮选中的题全部完成评分；这是评分覆盖率
overall_accuracy_on_scored: 1.0	已评分题目的正确率：2 ÷ 2 = 100%
correct_over_selected: 1.0	正确数 ÷ 本轮全部选题数，也是 100%
task_averaged_accuracy_on_observed_tasks: 1.0	本轮覆盖题型的准确率取平均，也是 100%
missing_ids: []	没有缺少评分的题
by_question_type 显示你这次覆盖了两种题型：
- knowledge-update：知识更新，例如用户后来改变了某个信息，系统能否记住更新后的情况。
- single-session-assistant：回答所需信息来自某次会话中的助手发言。
每种各测 1 题，都被判为正确。abstention_count: 0 表示没有抽到拒答题，因此拒答准确率是 null，意思是没有数据可以计算。
official_paper_judge: false 表示你用了 DeepSeek 替代裁判，并非论文标准的 GPT-4o 裁判，不表示评分失败。最后的校验值和规则描述用于记录评分程序，方便复现。


运行主要参数如下：
参数	含义
--dataset s	使用 S
--limit 1	测一题；10 表示十题；0 表示全部 500 题
--seed 42	固定选题，方便重复比较
--top-k 8	回答时最多提供 8 条检索记忆
--output ...	本次实验的保存目录
--max-write-calls 1000	最多允许多少次历史写入；增加题数后可能需要提高
--dry-run	只查看选题和工作量，不调用模型

理想模型对照参数要写 --dataset oracle，不能缩写成 o；还要加 --mode oracle
用于跑证据直接给出的情况下模型的准确率