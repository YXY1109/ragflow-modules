# ========
# 公共工具函数
# ========
import json
import re
from typing import Dict, Any

def get_completion(prompt: str, model: str = "qwen-turbo-latest") -> str:
    """
    模拟LLM调用的函数，用于测试和演示
    实际使用时替换为真实的LLM调用
    """
    # 简单的模拟响应逻辑
    if "多意图" in prompt or "multiple" in prompt.lower() or "价格？开放时间？怎么去？" in prompt:
        return """{
            "intent_count": 3,
            "sub_queries": [
                {
                    "query": "迪士尼乐园门票价格是多少？",
                    "priority": "high",
                    "topic": "ticket_price"
                },
                {
                    "query": "迪士尼乐园的开放时间是什么时候？",
                    "priority": "medium",
                    "topic": "opening_hours"
                },
                {
                    "query": "怎么去迪士尼乐园？",
                    "priority": "medium",
                    "topic": "transportation"
                }
            ],
            "reasoning": "用户询问三个独立的问题：价格、时间、交通方式"
        }"""
    elif "指代" in prompt or "reference" in prompt.lower() or "它不会要排队2小时吧？" in prompt:
        if "它" in prompt or "它不会要排队" in prompt:
            return "创极速光轮过山车项目的实际排队时间是多久？平日和周末分别需要等待多长时间？"
        elif "都" in prompt:
            return "疯狂动物城园区和宝藏湾园区的所有游乐项目都有身高限制吗？"
        else:
            return "上海迪士尼乐园的加勒比海盗：战争之潮项目适合小朋友吗？"
    elif "对比" in prompt or "comparison" in prompt.lower() or "哪个更好" in prompt:
        return "比较上海迪士尼乐园的疯狂动物城园区和宝藏湾园区，哪个更适合带小孩游玩？请从刺激程度、身高限制、互动性和安全性四个方面进行详细对比分析。"
    elif "上下文" in prompt or "context" in prompt.lower() or "其他适合" in prompt:
        if "其他设施" in prompt:
            return "除了警察局互动体验、朱迪警官训练营和尼克狐的冰淇淋店，疯狂动物城园区还有其他游乐设施吗？"
        else:
            return "除了疯狂动物城园区和宝藏湾园区，上海迪士尼乐园还有哪些其他适合小孩游玩的园区和项目？"
    elif "反问" in prompt or "rhetorical" in prompt.lower() or "不会要排队" in prompt or "怎么这么" in prompt:
        return """{
            "is_rhetorical": true,
            "emotion": "anxious",
            "emotion_intensity": "medium",
            "rewritten_query": "创极速光轮过山车的实际排队时间是多久？平日和周末分别需要等待多长时间？",
            "original_query": "它不会要排队2小时吧？",
            "empathy_keywords": ["理解您对排队时间的担心", "让我为您查询实际情况"],
            "reasoning": "用户使用反问句式表达对长时间排队的担心"
        }"""
    elif "RAG系统的Query类型识别器" in prompt:
        # 为Query类型识别器提供更智能的响应
        if "排队" in prompt and "吧" in prompt:
            return """{
                "query_type": ["rhetorical", "reference"],
                "confidence": 0.85,
                "detected_keywords": ["它", "不会...吧"],
                "reasoning": "用户使用反问句式询问排队时间，同时包含指代词'它'"
            }"""
        elif "其他适合" in prompt and "哪个更好" in prompt:
            return """{
                "query_type": ["context_dependent", "comparison"],
                "confidence": 0.9,
                "detected_keywords": ["其他", "哪个更好"],
                "reasoning": "用户询问其他适合的园区并进行对比，需要上下文信息"
            }"""
        elif "价格？开放时间？怎么去？" in prompt:
            return """{
                "query_type": ["multi_intent"],
                "confidence": 0.95,
                "detected_keywords": ["？", "？", "？"],
                "reasoning": "用户连续询问三个独立的问题，属于多意图查询"
            }"""
        else:
            return """{
                "query_type": [],
                "confidence": 0.3,
                "detected_keywords": [],
                "reasoning": "普通查询，无需特殊改写"
            }"""
    else:
        # 默认返回原query
        return "这是一个普通的查询"

def preprocess_json_response(response: str) -> str:
    """
    预处理JSON响应，提取JSON部分
    """
    # 尝试提取JSON部分
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return response

class BaseQueryRewriter:
    """Query改写器基类"""

    def __init__(self, model: str = "qwen-turbo-latest"):
        self.model = model

    def rewrite(self, query: str, context: str = "") -> Any:
        """改写查询的接口，子类需要实现"""
        raise NotImplementedError("子类必须实现rewrite方法")