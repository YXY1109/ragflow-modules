# ========
# AutoQueryRewriter: 统一Query改写引擎
# ========
from typing import Dict, Any, List, Optional

from .comparison import ComparisonQueryRewriter
from .context_dependent import ContextDependentQueryRewriter
from .multiIntent import MultiIntentQueryRewriter
from .reference import ReferenceQueryRewriter
from .rhetorical import RhetoricalQueryRewriter
from .utils import get_json_completion, QueryRewriterConfig


class AutoQueryRewriter:
    """统一的Query改写调度器"""

    PRIORITY = [
        "multi_intent",  # 多意图型（必须先拆分）
        "reference",  # 模糊指代型（让语义清晰）
        "comparison",  # 对比型（补充对比维度）
        "context_dependent",  # 上下文依赖型（补全信息）
        "rhetorical"  # 反问型（最后处理情绪）
    ]

    def __init__(self,
                 config: Optional[QueryRewriterConfig] = None,
                 model: Optional[str] = None,
                 context_rewriter=None, comparison_rewriter=None,
                 reference_rewriter=None, multi_intent_rewriter=None,
                 rhetorical_rewriter=None):
        """
        初始化AutoQueryRewriter

        Args:
            config: Query改写器配置
            model: 使用的模型名称
            context_rewriter: 自定义上下文改写器
            comparison_rewriter: 自定义对比改写器
            reference_rewriter: 自定义指代改写器
            multi_intent_rewriter: 自定义多意图改写器
            rhetorical_rewriter: 自定义反问改写器
        """
        self.config = config or QueryRewriterConfig()
        self.model = model or self.config.model

        # 初始化各个改写器，传递配置
        self.context_rewriter = context_rewriter or ContextDependentQueryRewriter(
            model=self.model, config=self.config
        )
        self.comparison_rewriter = comparison_rewriter or ComparisonQueryRewriter(
            model=self.model, config=self.config
        )
        self.reference_rewriter = reference_rewriter or ReferenceQueryRewriter(
            model=self.model, config=self.config
        )
        self.multi_intent_rewriter = multi_intent_rewriter or MultiIntentQueryRewriter(
            model=self.model, config=self.config
        )
        self.rhetorical_rewriter = rhetorical_rewriter or RhetoricalQueryRewriter(
            model=self.model, config=self.config
        )

    # ---------------
    # Step1: 类型识别
    # -------------
    def analyze_query_type(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        """调用LLM识别Query类型，支持多标签"""
        instruction = """你是一个RAG系统的Query类型识别器，请判断当前Query是否属于以下类型:
        1. multi_intent: 多意图 - 一个查询包含多个独立的问题或意图
        2. reference: 模糊指代 - 包含"它"、"这个"、"都"等需要上下文才能理解的指代词
        3. comparison: 对比型 - 包含"哪个更好"、"比较"、"对比"等对比词汇
        4. context_dependent: 上下文依赖 - 需要"其他"、"还有"、"也"等上下文信息才能完整理解
        5. rhetorical: 反问型 - 使用反问句式表达情绪（"不会...吧"、"怎么这么..."等）

        请返回严格的JSON格式，包含以下字段：
        - query_type: 检测到的查询类型列表（数组）
        - confidence: 整体置信度（0-1之间的浮点数）
        - detected_keywords: 检测到的关键词列表
        - reasoning: 识别理由和逻辑说明

        注意：一个查询可能同时属于多个类型。"""
        prompt = f"""
        ### 指令 ###
        {instruction}

        ### 对话历史 ###
        {conversation_history if conversation_history else "无对话历史"}

        ### 当前Query ###
        {query}

        ### 输出JSON ###
        """

        try:
            result = get_json_completion(
                prompt,
                model=self.model,
                temperature=0.3,  # 使用较低温度确保一致性
                config=self.config
            )

            # 验证并标准化结果
            if "error" in result:
                return self._get_default_analysis(query, "API调用失败")

            return self._validate_analysis_result(result, query)

        except Exception as e:
            print(f"Query类型分析失败: {e}")
            return self._get_default_analysis(query, str(e))

    def _get_default_analysis(self, query: str, error_msg: str) -> Dict[str, Any]:
        """获取默认分析结果（处理失败时使用）"""
        return {
            "query_type": [],
            "confidence": 0.0,
            "detected_keywords": [],
            "reasoning": f"分析失败: {error_msg}"
        }

    def _validate_analysis_result(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """验证并标准化分析结果"""
        # 确保必要字段存在
        if "query_type" not in result:
            result["query_type"] = []
        if "confidence" not in result:
            result["confidence"] = 0.0
        if "detected_keywords" not in result:
            result["detected_keywords"] = []
        if "reasoning" not in result:
            result["reasoning"] = "无详细分析"

        # 标准化query_type为列表
        if isinstance(result["query_type"], str):
            result["query_type"] = [result["query_type"]]
        elif not isinstance(result["query_type"], list):
            result["query_type"] = []

        # 标准化detected_keywords为列表
        if isinstance(result["detected_keywords"], str):
            result["detected_keywords"] = [result["detected_keywords"]]
        elif not isinstance(result["detected_keywords"], list):
            result["detected_keywords"] = []

        # 标准化confidence为浮点数
        try:
            result["confidence"] = float(result["confidence"])
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))  # 限制在0-1之间
        except (ValueError, TypeError):
            result["confidence"] = 0.0

        # 验证query_type值
        valid_types = ["multi_intent", "reference", "comparison", "context_dependent", "rhetorical"]
        result["query_type"] = [qt for qt in result["query_type"] if qt in valid_types]

        return result

    # -------------
    # Step2: 调度改写器
    # -------------
    def rewrite(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        analysis = self.analyze_query_type(query, conversation_history)
        query_types = analysis.get("query_type", [])

        if not query_types:
            return {
                "final_query": query,
                "query_types": [],
                "sub_queries": [],
                "emotion": {
                    "emotion": "neutral",
                    "emotion_intensity": "none",
                    "empathy_keywords": []
                },
                "rewrite_steps": [],
                "analysis": analysis
            }

        # 先处理多意图
        if "multi_intent" in query_types:
            multi_result = self.multi_intent_rewriter.rewrite(query, conversation_history)
            if multi_result["intent_count"] > 1:
                final_sub_queries = []
                for item in multi_result["sub_queries"]:
                    processed = self._process_single_query(
                        query=item["query"],
                        conversation_history=conversation_history,
                        query_types=[qt for qt in query_types if qt != "multi_intent"]
                    )
                    processed.update({
                        "priority": item.get("priority", "medium"),
                        "topic": item.get("topic", "general")
                    })
                    final_sub_queries.append(processed)

                return {
                    "final_query": query,
                    "query_types": query_types,
                    "sub_queries": final_sub_queries,
                    "emotion": {
                        "emotion": "neutral",
                        "emotion_intensity": "none",
                        "empathy_keywords": []
                    },
                    "rewrite_steps": [{"type": "multi_intent", "result": multi_result}],
                    "analysis": analysis
                }

        # 单Query改写流程
        result = self._process_single_query(
            query=query,
            conversation_history=conversation_history,
            query_types=query_types
        )
        result.update({
            "query_types": query_types,
            "sub_queries": [],
            "analysis": analysis
        })
        return result

    # ----------
    # Step3: 针对单个Query按优先级改写
    # -------
    def _process_single_query(self, query: str, conversation_history: str, query_types: List[str]):
        current_query = query
        rewrite_steps = []
        emotion_info = {
            "emotion": "neutral",
            "emotion_intensity": "none",
            "empathy_keywords": []
        }

        for query_type in self.PRIORITY:
            # 多意图已在外层处理
            if query_type not in query_types:
                continue

            if query_type == "reference":
                rewritten = self.reference_rewriter.rewrite(current_query, conversation_history)
                rewrite_steps.append({"type": "reference", "rewrite": rewritten})
                current_query = rewritten
            elif query_type == "comparison":
                rewritten = self.comparison_rewriter.rewrite(current_query, conversation_history)
                rewrite_steps.append({"type": "comparison", "rewrite": rewritten})
                current_query = rewritten
            elif query_type == "context_dependent":
                rewritten = self.context_rewriter.rewrite(current_query, conversation_history)
                rewrite_steps.append({"type": "context", "rewrite": rewritten})
                current_query = rewritten
            elif query_type == "rhetorical":
                result = self.rhetorical_rewriter.rewrite(current_query, conversation_history)
                rewrite_steps.append({"type": "rhetorical", "rewrite": result})
                current_query = result["rewritten_query"]
                emotion_info = {
                    "emotion": result["emotion"],
                    "emotion_intensity": result["emotion_intensity"],
                    "empathy_keywords": result["empathy_keywords"]
                }

        return {
            "final_query": current_query,
            "emotion": emotion_info,
            "rewrite_steps": rewrite_steps
        }


def print_rewrite_log(result: Dict[str, Any]):
    print("=" * 80)
    print("🛠️ Query改写流水线")
    print("=" * 80)
    print(f"识别类型: {', '.join(result.get('query_types', []))}")
    print(f"情绪标签: {result['emotion']['emotion']} ({result['emotion']['emotion_intensity']})")

    for step in result.get("rewrite_steps", []):
        print("-" * 80)
        print(f"步骤: {step['type']}")
        print(f"结果: {step['rewrite']}")

    if result.get("sub_queries"):
        print("\n📋 子Query列表:")
        for idx, sub in enumerate(result['sub_queries'], 1):
            print(f"  [{idx}] ({sub['priority']}) {sub['final_query']}")
            print(f"      情绪: {sub['emotion']['emotion']}")

    print("=" * 80)
