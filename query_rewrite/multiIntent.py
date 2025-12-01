# ==========
# 多意图型Query改写器
# ==========
from .utils import BaseQueryRewriter, get_completion
import json

class MultiIntentQueryRewriter(BaseQueryRewriter):
    """多意图型Query改写器

    核心能力:
    1. 识别一个Query中包含的多个意图
    2. 拆分为独立的子Query
    3. 保留原始Query的上下文信息
    4. 标注每个子Query的优先级

    应用场景:
    • 用户连续提问（"价格？时间？地点？"）
    • 复杂查询（"对比+列举+建议"）
    • 效率优化（一次问多个问题）
    """

    def rewrite(self, query: str, context: str = "") -> dict:
        """拆分多意图Query为多个独立子Query

        拆分策略:
        1. 显式标记识别: 问号、顿号、分号等
        2. 语义主题识别: 通过LLM理解不同意图
        3. 上下文保留: 每个子Query都包含必要的上下文
        4. 优先级标注: 标注哪些问题更重要

        参数:
            query: 原始查询（可能包含多个意图）
            context: 上下文信息
        """
        instruction = """
你是一个查询意图分析专家，专门识别和拆分多意图查询。

【任务】:判断用户Query是否包含多个意图，如果包含则拆分为独立的子查询。

【识别标志】:
1. 显式标记（最明显）: 多个问号、顿号分隔、分号分隔、并列连词
2. 语义层面（需要理解）:不同主题、不同层次、不同对象

【拆分原则】:
原则1 - 独立性: 每个子Query必须是一个完整、独立的问题，不依赖其他子Query
原则2 - 上下文保留: 每个子Query都要包含必要的上下文（地点、对象等）
原则3 - 优先级标注:
- high: 主要问题、核心需求
- medium: 次要问题、补充信息
- low: 可选问题、额外信息

【输出格式】:
请返回JSON格式的结果，包含以下字段：
- intent_count: 意图数量
- sub_queries: 子查询列表，每个包含query和priority
- original_query: 原始查询
- reasoning: 拆分理由
"""
        prompt = f"""
### 指令 ###
{instruction}

### 上下文信息 ###
{context}

### 用户查询 ###
{query}

### 分析结果（JSON格式）###
"""
        response = get_completion(prompt, self.model)

        # 解析JSON
        try:
            result = json.loads(response)
            return result
        except Exception as e:
            # 如果解析失败，返回原始Query
            return {
                "intent_count": 1,
                "sub_queries": [],
                "original_query": query,
                "reasoning": f"解析失败: {str(e)}"
            }