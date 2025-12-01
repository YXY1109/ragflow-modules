# Query改写系统

## 概述

实现了一个完整的Query改写系统，包含5种独立的Query改写器：

1. **上下文依赖型** - 补全上下文信息
2. **对比型** - 明确对比维度
3. **模糊指代型** - 消除指代歧义
4. **多意图型** - 拆分复合查询
5. **反问型** - 识别并转换情绪化表达

## 文件结构

```
query_rewrite/
├── __init__.py                 # 模块初始化
├── utils.py                    # 公共工具函数
├── auto_query_rewriter.py      # 统一改写调度器
├── context_dependent.py        # 上下文依赖型改写器
├── comparison.py               # 对比型改写器
├── reference.py               # 模糊指代型改写器
├── multiIntent.py             # 多意图型改写器
├── rhetorical.py              # 反问型改写器
├── test.py                    # 综合测试文件
└── README.md                  # 说明文档
```

## 使用方法

```python
from query_rewrite import AutoQueryRewriter

# 创建改写器实例
rewriter = AutoQueryRewriter()

# 改写查询
result = rewriter.rewrite(
    query="它不会要排队2小时吧？",
    conversation_history="用户: \"创极速光轮是什么项目？\" AI: \"是明日世界的过山车，全球最快\""
)

print(f"改写结果: {result['final_query']}")
print(f"识别类型: {result['query_types']}")
```

## 运行测试

```bash
python query_rewrite/test.py
```

## 改写优先级

1. 多意图型 - 必须先拆分
2. 模糊指代型 - 让语义清晰
3. 对比型 - 补充对比维度
4. 上下文依赖型 - 补全信息
5. 反问型 - 最后处理情绪