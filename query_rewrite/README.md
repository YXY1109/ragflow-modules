# Query改写系统 - 基于OpenRouter API

## 概述

这是一个完整的Query改写系统，基于OpenRouter API和真实的LLM模型实现。系统能够识别和改写5种不同类型的用户查询，提升RAG系统的检索准确性和用户体验。

## 核心功能

### 🔍 支持的Query类型

1. **多意图型 (Multi-Intent)**
   - 识别：一个查询包含多个独立问题
   - 标志：多个问号、顿号分隔、并列连词
   - 改写：拆分为独立的子查询，标注优先级
   - 示例：`"门票价格？开放时间？交通方式？"`

2. **模糊指代型 (Reference)**
   - 识别：包含需要上下文的指代词
   - 标志：`"它"`、`"这个"`、`"都"`等
   - 改写：用明确的名词替换指代词
   - 示例：`"它不会要排队2小时吧？"` → `"创极速光轮不会要排队2小时吧？"`

3. **对比型 (Comparison)**
   - 识别：包含对比词汇
   - 标志：`"哪个更好"`、`"比较"`、`"对比"`等
   - 改写：明确对比对象，补充对比维度
   - 示例：`"疯狂动物城和宝藏湾哪个更好？"`

4. **上下文依赖型 (Context-Dependent)**
   - 识别：需要对话历史才能理解
   - 标志：`"其他"`、`"还有"`、`"也"`等
   - 改写：补全上下文信息，生成独立查询
   - 示例：`"还有其他适合小孩的项目吗？"`

5. **反问型 (Rhetorical)**
   - 识别：使用反问句式表达情绪
   - 标志：`"不会...吧"`、`"怎么这么"`、`"难道"`等
   - 改写：转换为客观疑问句，识别用户情绪
   - 示例：`"门票怎么这么贵？"` → `"上海迪士尼乐园的门票价格是多少？"`

## 快速开始

### 1. 安装依赖

```bash
# 确保已安装所需依赖
uv sync

# 或者手动安装
pip install openai httpx
```

### 2. 基本使用

```python
from query_rewrite import AutoQueryRewriter

# 创建改写器（使用默认配置）
rewriter = AutoQueryRewriter()

# 改写查询
result = rewriter.rewrite(
    query="它不会要排队2小时吧？",
    conversation_history="用户: \"创极速光轮是什么项目？\" AI: \"是明日世界的过山车，全球最快\""
)

# 输出结果
print(f"改写结果: {result['final_query']}")
print(f"识别类型: {result['query_types']}")
print(f"情绪状态: {result['emotion']['emotion']}")
```

### 3. 自定义配置

```python
from query_rewrite import AutoQueryRewriter, QueryRewriterConfig

# 创建自定义配置
config = QueryRewriterConfig(
    model="z-ai/glm-4.5-air:free",
    timeout_read=120.0,    # 读取超时120秒
    timeout_connect=30.0,  # 连接超时30秒
    temperature=0.3,        # 较低温度确保稳定性
    max_connections=10,     # 最大连接数
    max_keepalive_connections=5
)

# 使用自定义配置
rewriter = AutoQueryRewriter(config=config)
```

## 高级功能

### 🔧 独立改写器使用

每个改写器都可以独立使用：

```python
from query_rewrite import (
    MultiIntentQueryRewriter,
    RhetoricalQueryRewriter,
    ReferenceQueryRewriter,
    ComparisonQueryRewriter,
    ContextDependentQueryRewriter,
    QueryRewriterConfig
)

config = QueryRewriterConfig()

# 多意图改写器
multi_rewriter = MultiIntentQueryRewriter(config=config)
result = multi_rewriter.rewrite(
    query="门票价格？开放时间？怎么去？",
    context="用户计划迪士尼之旅"
)
print(f"子查询数量: {result['intent_count']}")

# 反问改写器
rhetorical_rewriter = RhetoricalQueryRewriter(config=config)
result = rhetorical_rewriter.rewrite(
    query="这个项目不会要等很久吧？",
    context="用户在询问排队时间"
)
print(f"情绪识别: {result['emotion']}")
```

### 📊 批量处理

```python
queries = [
    {"query": "它好玩吗？", "context": "用户在询问创极速光轮"},
    {"query": "门票价格？开放时间？", "context": "用户计划迪士尼之旅"},
    {"query": "哪个更好玩？", "context": "用户在比较两个项目"}
]

results = []
for item in queries:
    result = rewriter.rewrite(**item)
    results.append(result)
    print(f"原: {item['query']} -> 改: {result['final_query']}")
```

### 🎭 情绪识别和处理

系统能够识别用户情绪并提供情绪标签：

```python
# 反问句情绪分析
result = rewriter.rewrite(
    query="门票怎么这么贵？",
    conversation_history=""
)

emotion = result['emotion']
print(f"情绪类型: {emotion['emotion']}")          # complaining, anxious, surprised
print(f"情绪强度: {emotion['emotion_intensity']}")  # low, medium, high
print(f"共情关键词: {emotion['empathy_keywords']}")  # 用于回复的情绪化表达
```

## 配置选项

### QueryRewriterConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | `"z-ai/glm-4.5-air:free"` | 使用的模型名称 |
| `timeout_connect` | float | `30.0` | 连接超时时间（秒） |
| `timeout_read` | float | `120.0` | 读取超时时间（秒） |
| `timeout_write` | float | `30.0` | 写入超时时间（秒） |
| `timeout_pool` | float | `30.0` | 连接池超时时间（秒） |
| `max_connections` | int | `10` | 最大连接数 |
| `max_keepalive_connections` | int | `5` | 最大保持连接数 |
| `api_key` | str | 预配置 | OpenRouter API密钥 |
| `base_url` | str | 预配置 | OpenRouter API基础URL |

### 支持的模型

推荐使用以下免费模型：

- `z-ai/glm-4.5-air:free` - 智谱AI GLM-4.5 Air（推荐）
- `meta-llama/llama-3.1-8b-instruct:free` - Meta Llama 3.1 8B
- `microsoft/wizardlm-2-8x22b:free` - Microsoft WizardLM-2

## API 使用示例

### 1. 基础Query改写

```python
from query_rewrite import AutoQueryRewriter, QueryRewriterConfig

# 配置
config = QueryRewriterConfig(
    model="z-ai/glm-4.5-air:free",
    timeout_read=60.0
)

# 创建改写器
rewriter = AutoQueryRewriter(config=config)

# 示例1: 反问 + 指代
result1 = rewriter.rewrite(
    query="它不会要排队2小时吧？",
    conversation_history="用户刚才询问了创极速光轮项目"
)
print(f"识别类型: {result1['query_types']}")  # ['rhetorical', 'reference']
print(f"改写结果: {result1['final_query']}")  # "创极速光轮的实际排队时间是多久？"

# 示例2: 多意图
result2 = rewriter.rewrite(
    query="门票价格？开放时间？怎么去？",
    conversation_history=""
)
print(f"意图数量: {len(result2['sub_queries'])}")  # 3
for sub in result2['sub_queries']:
    print(f"  {sub['priority']}: {sub['query']}")
```

### 2. 自定义改写器组合

```python
from query_rewrite import (
    AutoQueryRewriter, QueryRewriterConfig,
    RhetoricalQueryRewriter, ReferenceQueryRewriter
)

# 自定义配置
config = QueryRewriterConfig(temperature=0.1)  # 最低随机性

# 创建自定义改写器
custom_rhetorical = RhetoricalQueryRewriter(
    model="z-ai/glm-4.5-air:free",
    config=config
)

custom_reference = ReferenceQueryRewriter(
    model="z-ai/glm-4.5-air:free",
    config=config
)

# 使用自定义改写器
rewriter = AutoQueryRewriter(
    config=config,
    rhetorical_rewriter=custom_rhetorical,
    reference_rewriter=custom_reference
)
```

### 3. 错误处理和降级

```python
from query_rewrite import AutoQueryRewriter, QueryRewriterConfig

config = QueryRewriterConfig(
    timeout_read=30.0,  # 较短超时
    model="z-ai/glm-4.5-air:free"
)

rewriter = AutoQueryRewriter(config=config)

try:
    result = rewriter.rewrite(
        query="它好玩吗？",
        conversation_history="用户在询问某个项目"
    )

    # 检查结果有效性
    if result.get('final_query') and result.get('query_types'):
        print("改写成功")
        print(f"结果: {result['final_query']}")
    else:
        print("改写失败，使用原始查询")
        print(f"分析结果: {result.get('analysis', {})}")

except Exception as e:
    print(f"API调用失败: {e}")
    # 实现降级策略，比如使用原始查询或规则改写
```

## 性能优化

### 1. 连接池配置

```python
config = QueryRewriterConfig(
    max_connections=20,                # 增加最大连接数
    max_keepalive_connections=10,       # 增加保持连接数
    timeout_connect=15.0,              # 减少连接超时
    timeout_read=60.0,                 # 适中的读取超时
    timeout_pool=15.0                  # 减少连接池超时
)
```

### 2. 模型选择建议

- **高准确性需求**: `z-ai/glm-4.5-air:free`
- **快速响应需求**: `meta-llama/llama-3.1-8b-instruct:free`
- **复杂查询处理**: `microsoft/wizardlm-2-8x22b:free`

### 3. 温度参数调节

```python
# 生产环境 - 低随机性
config = QueryRewriterConfig(temperature=0.1)

# 测试环境 - 适度随机性
config = QueryRewriterConfig(temperature=0.3)

# 创意场景 - 较高随机性
config = QueryRewriterConfig(temperature=0.7)
```

## 测试和验证

### 运行完整测试

```bash
# 运行完整测试套件
python query_rewrite/complete_test.py
```

### 单独测试功能

```python
# 测试多意图识别
from query_rewrite import MultiIntentQueryRewriter
rewriter = MultiIntentQueryRewriter()
result = rewriter.rewrite("门票价格？开放时间？", "用户计划迪士尼之旅")

# 测试情绪识别
from query_rewrite import RhetoricalQueryRewriter
rewriter = RhetoricalQueryRewriter()
result = rewriter.rewrite("这个项目不会要等很久吧？", "询问排队时间")
```

## 常见问题

### Q: 如何处理API调用失败？
A: 系统包含完整的错误处理机制：
- 自动重试机制
- 降级策略（返回原始查询）
- 详细的错误日志

### Q: 支持哪些语言？
A: 系统基于中文设计，但支持多语言处理。主要优化用于中文查询改写。

### Q: 如何添加自定义改写器？
A: 继承BaseQueryRewriter类并实现rewrite方法：

```python
from query_rewrite import BaseQueryRewriter, QueryRewriterConfig

class CustomRewriter(BaseQueryRewriter):
    def __init__(self, model=None, config=None):
        self.model = model or "z-ai/glm-4.5-air:free"
        self.config = config or QueryRewriterConfig()

    def rewrite(self, query: str, context: str = ""):
        # 实现自定义改写逻辑
        return f"[改写] {query}"
```

### Q: 如何监控改写效果？
A: 系统提供详细的改写日志：

```python
from query_rewrite.auto_query_rewriter import print_rewrite_log

result = rewriter.rewrite(query, context)
print_rewrite_log(result)  # 打印详细改写日志
```

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 添加测试用例
4. 提交 Pull Request

## 许可证

MIT License

## 更新日志

### v2.0.0 - 基于OpenRouter API重构
- ✅ 集成OpenRouter API和真实LLM
- ✅ 重构utils.py，支持类型化的OpenAI客户端
- ✅ 更新所有改写器使用真实API
- ✅ 添加QueryRewriterConfig配置管理
- ✅ 优化AutoQueryRewriter，支持自定义改写器
- ✅ 完善错误处理和降级机制
- ✅ 添加完整测试和示例

### v1.0.0 - 基础版本
- ✅ 实现5种基础改写器
- ✅ 统一的改写调度器
- ✅ 模拟LLM响应

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。