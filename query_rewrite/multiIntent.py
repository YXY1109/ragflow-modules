# ==========
# 多意图型Query改写器
# ==========
from typing import Dict, Any

from .utils import BaseQueryRewriter, get_json_completion, QueryRewriterConfig


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

    def __init__(self, model: str = None, config: QueryRewriterConfig = None):
        self.model = model or (config.model if config else "z-ai/glm-4.5-air:free")
        self.config = config or QueryRewriterConfig()

    def rewrite(self, query: str, context: str = "") -> Dict[str, Any]:
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
请返回严格的JSON格式，包含以下字段：
- intent_count: 意图数量（整数）
- sub_queries: 子查询列表，每个对象包含:
  - query: 拆分后的子查询字符串
  - priority: 优先级（"high", "medium", "low"）
  - topic: 主题类别（如"价格", "时间", "交通"等）
- original_query: 原始查询
- reasoning: 拆分理由和逻辑

【注意事项】:
1. 如果只有一个意图，intent_count设为1，sub_queries为空数组
2. 确保每个子查询都是完整的、可独立回答的问题
3. 优先级标注要合理，核心问题为high
4. topic字段要简明扼要，反映查询主题
"""

        prompt = f"""
### 指令 ###
{instruction}

### 上下文信息 ###
{context if context else "无特定上下文"}

### 用户查询 ###
{query}

### 分析结果（严格JSON格式）###
"""

        # 使用新的JSON completion方法
        try:
            result = get_json_completion(
                prompt,
                model=self.model,
                temperature=0.3,  # 使用较低温度确保一致性
                config=self.config
            )

            # 确保返回的数据结构正确
            if "error" in result:
                # 如果API调用失败，返回默认结果
                return self._get_default_result(query, "API调用失败")

            # 验证并补全必要字段
            return self._validate_and_normalize_result(result, query)

        except Exception as e:
            print(f"多意图改写失败: {e}")
            return self._get_default_result(query, str(e))

    def _get_default_result(self, query: str, error_msg: str) -> Dict[str, Any]:
        """获取默认结果（处理失败时使用）"""
        return {
            "intent_count": 1,
            "sub_queries": [],
            "original_query": query,
            "reasoning": f"解析失败或只有一个意图: {error_msg}"
        }

    def _validate_and_normalize_result(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """验证并标准化返回结果"""
        # 确保必要字段存在
        if "intent_count" not in result:
            result["intent_count"] = 1
        if "sub_queries" not in result:
            result["sub_queries"] = []
        if "original_query" not in result:
            result["original_query"] = original_query
        if "reasoning" not in result:
            result["reasoning"] = "无详细说明"

        # 标准化子查询格式
        if result["sub_queries"]:
            for i, sub_query in enumerate(result["sub_queries"]):
                if not isinstance(sub_query, dict):
                    continue

                # 确保必要字段
                if "query" not in sub_query:
                    sub_query["query"] = f"子查询 {i + 1}"
                if "priority" not in sub_query:
                    sub_query["priority"] = "medium"
                if "topic" not in sub_query:
                    sub_query["topic"] = "general"

                # 标准化优先级
                if sub_query["priority"] not in ["high", "medium", "low"]:
                    sub_query["priority"] = "medium"

        return result
