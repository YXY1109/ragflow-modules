# ========
# AutoQueryRewriter: 统一Query改写引擎
# ========
import json
from typing import Dict, Any, List
classAutoQueryRewriter:
"""统一的Query改写调度器"""

    PRIORITY = [
"multi_intent",  # 多意图型（必须先拆分）
"reference",  # 模糊指代型（让语义清晰）
"comparison",  # 对比型（补充对比维度）
"context_dependent",  # 上下文依赖型（补全信息）
"rhetorical"# 反问型（最后处理情绪）
        ]

def__init__(self, llm_model: str = "qwen-turbo-latest"):
        self.llm_model = llm_model

        self.context_rewriter = context_rewriter or ContextDependentQueryRewriter(llm_model)
        self.comparison_rewriter = comparison_rewriter or ComparisonQueryRewriter(llm_model)
        self.reference_rewriter = reference_rewriter or ReferenceQueryRewriter(llm_model)
        self.multi_intent_rewriter = multi_intent_rewriter or MultiIntentQueryRewriter(llm_model)
        self.rhetorical_rewriter = rhetorical_rewriter or RhetoricalQueryRewriter(llm_model)

# ---------------
# Step1: 类型识别
# -------------
defanalyze_query_type(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
""" 调用LLM识别Query类型，支持多标签   """
        instruction = """你是一个RAG系统的Query类型识别器，请判断当前Query是否属于以下类型:1. multi_intent: 多意图2. reference: 模糊指代3. comparison: 对比型4. context_dependent: 上下文依赖5. rhetorical: 反问型请输出JSON，字段包括 query_type, confidence, detected_keywords, reasoning。"""
        prompt = f"""
        ### 指令 ###{instruction}
        ### 对话历史 ###{conversation_history}
        ### 当前Query ###{query}
        ### 输出JSON ###
        """
        response = get_completion(prompt, self.llm_model)
return json.loads(preprocess_json_response(response))
# -------------
# Step2: 调度改写器
# -------------
defrewrite(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        analysis = self.analyze_query_type(query, conversation_history)
        query_types = analysis.get("query_type", [])
ifnot query_types:
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
if"multi_intent"in query_types:
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
def_process_single_query(self, query: str, conversation_history: str, query_types: List[str]):
        current_query = query
        rewrite_steps = []
        emotion_info = {
"emotion": "neutral",
"emotion_intensity": "none",
"empathy_keywords": []
        }
for query_type in self.PRIORITY[1:]:
# 多意图已在外层处理
if query_type notin query_types:
continueif query_type == "reference":
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


defprint_rewrite_log(result: Dict[str, Any]):
    print("="*80)
    print("🛠️ Query改写流水线")
    print("="*80)
    print(f"识别类型: {', '.join(result.get('query_types', []))}")
    print(f"情绪标签: {result['emotion']['emotion']} ({result['emotion']['emotion_intensity']})")
for step in result.get("rewrite_steps", []):
        print("-"*80)
        print(f"步骤: {step['type']}")
        print(f"结果: {step['rewrite']}")
if result.get("sub_queries"):
        print("\n📋 子Query列表:")
for idx, sub in enumerate(result['sub_queries'], 1):
            print(f"  [{idx}] ({sub['priority']}) {sub['final_query']}")
            print(f"      情绪: {sub['emotion']['emotion']}")
    print("="*80)