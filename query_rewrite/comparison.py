# ==========
# 对比型Query改写器
# =========
from .utils import BaseQueryRewriter, get_completion, QueryRewriterConfig


class ComparisonQueryRewriter(BaseQueryRewriter):
    """对比型Query改写器

    核心功能:
    1. 识别对比词（"哪个"、"更"、"比较"）
    2. 明确对比对象（从Query或历史中提取）
    3. 细化对比维度（补充具体的评估标准）
    """

    def __init__(self, model: str = None, config: QueryRewriterConfig = None):
        self.model = model or (config.model if config else "z-ai/glm-4.5-air:free")
        self.config = config or QueryRewriterConfig()

    def rewrite(self, query: str, context_info: str = "") -> str:
        """改写对比型Query

        改写逻辑:
        1. 保留对比对象（如果明确）
        2. 补充对比对象（如果在上下文中）
        3. 细化对比维度（添加具体评估标准）
        4. 添加场景信息（地点、用途等）

        参数:
            query: 原始查询
            context_info: 上下文信息

        返回:
            改写后的对比性查询
        """
        instruction = """你是一个查询分析专家，专门处理对比型问题。
【任务】:分析用户的对比型问题，明确对比对象和对比维度。

【识别标志】:
- 对比词: "哪个"、"更"、"比较"、"还是"、"vs"
- 对比结构: "A和B哪个..."、"A比B..."、"选A还是B"

【改写步骤】:
步骤1 - 明确对比对象:
  • 如果Query中已有对比对象 → 保留
  • 如果对比对象在上下文中 → 提取并补充
  • 如果对比对象不明确 → 保持原样或标注需要补充

步骤2 - 细化对比维度:
  • "适合小孩" → 刺激程度、身高限制、互动性、安全性
  • "好玩" → 项目丰富度、体验时长、独特性、用户评价
  • "方便" → 交通便利、时间成本、经济成本
  • "值得" → 性价比、体验质量、独特性

步骤3 - 补充完整信息:
  • 添加地点: "上海迪士尼乐园的..."
  • 添加用途: "带小孩游玩"、"快速到达"等
  • 添加场景: 具体的使用情境

【示例1】:
原始: "疯狂动物城和宝藏湾哪个更适合小孩？"
上下文: "用户计划带5岁孩子游玩"
改写: "比较上海迪士尼乐园的疯狂动物城园区和宝藏湾园区，哪个更适合带5岁小孩游玩？请从以下方面对比分析：
1. 游乐项目的刺激程度（是否有过山车等刺激项目）
2. 身高和年龄限制（孩子是否能玩所有项目）
3. 互动性和趣味性（是否有适合儿童的互动体验）
4. 安全性和家长陪同（是否需要成人陪同）"

【示例2】:
原始: "地铁和打车哪个更方便？"
上下文: "用户从浦东机场去迪士尼"
改写: "从浦东机场到上海迪士尼乐园，乘坐地铁和打车哪个更方便？请从以下方面对比：
1. 交通便利性（是否需要换乘、是否有直达）
2. 时间成本（预计所需时间、是否会堵车）
3. 经济成本（票价/车费对比）
4. 舒适度（是否拥挤、是否方便携带行李）"""

        prompt = f"""
### 指令 ###
{instruction}

### 上下文信息 ###
{context_info}

### 原始查询 ###
{query}

### 改写后的查询 ###
"""
        response = get_completion(prompt, self.model)
        return response.strip()
