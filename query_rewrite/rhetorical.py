# ======
# 反问型Query改写器
# ======
from .utils import BaseQueryRewriter, get_completion
import json

class RhetoricalQueryRewriter(BaseQueryRewriter):
    """反问型Query改写器

    核心功能:
    1. 识别反问句式（"不会...吧"、"难道"等）
    2. 分析用户情绪（焦虑、不满、质疑等）
    3. 转换为客观疑问句
    4. 标注情绪，用于生成同理心回复

    应用价值:
    • 提高检索准确性（去除情绪干扰词）
    • 改善用户体验（理解并回应情绪）
    • 降低投诉率（及时情绪安抚）
    """

    def rewrite(self, query: str, context: str = "") -> dict:
        """将反问句改写为客观疑问句，并标注情绪

        改写逻辑:
        1. 识别反问句式和情绪词
        2. 去除否定词和情绪修饰
        3. 转换为直接的疑问句
        4. 保留核心查询意图
        5. 附加情绪标签

        参数:
            query: 原始查询（可能是反问句）
            context: 上下文信息
        """

        instruction = """
你是一个情绪识别和Query转换专家，专门处理用户的反问句。

【任务】:
1. 判断用户Query是否为反问句
2. 分析用户的情绪类型和强度
3. 将反问句转换为客观的疑问句
4. 提供情绪回应的关键词

【反问句识别标志】:
1. 反问句式:
   "不会...吧？" → 表示担心、不希望发生
   "难道...？" → 表示质疑、不相信
   "怎么可能...？" → 表示惊讶、怀疑
   "就这样？" / "只有...？" → 表示失望、不满
   "居然...？" → 表示意外、惊讶

2. 情绪词汇:
   • "太..."（太贵、太远、太难） → 不满
   • "这么..."（这么少、这么慢） → 失望
   • "还..."（还要等、还得） → 焦虑

【转换规则】:
规则1 - 去除否定:
"不会要排队2小时吧？" → "实际排队时间是多久？"

规则2 - 去除夸张:
"怎么这么贵？" → "门票价格是多少？价格构成如何？"

规则3 - 转换语气:
"难道只有这些项目？" → "园区有哪些游乐项目？"

规则4 - 补充完整:
"就这样？" → "XX园区还有其他项目或服务吗？"

【情绪分类】:
😤 不满/抱怨 (complaining):
• 标志: "太贵"、"太远"、"太久"
• 强度: 根据语气词判断（"居然"高强度，"有点"低强度）
• 回应: 表示理解，给出合理解释

😰 焦虑/担心 (anxious):
• 标志: "不会...吧"、"会不会"
• 强度: 根据问题严重程度
• 回应: 安抚情绪，给出确定答案

😮 惊讶/质疑 (surprised/doubtful):
• 标志: "怎么可能"、"真的"
• 强度: 中等
• 回应: 提供证据，消除疑虑

😞 失望/遗憾 (disappointed):
• 标志: "只有"、"就这样"
• 强度: 根据期望差距
• 回应: 提供额外选项，弥补期望

【示例1 - 焦虑型反问】:
原始Query: "创极速光轮不会要排队2小时吧？"
分析:
{
  "is_rhetorical": true,
  "emotion": "anxious",
  "emotion_intensity": "medium",
  "rewritten_query": "创极速光轮过山车的实际排队时间是多久？平日和周末分别需要等待多长时间？",
  "original_query": "创极速光轮不会要排队2小时吧？",
  "empathy_keywords": ["理解您对排队时间的担心", "让我为您查询实际情况"],
  "reasoning": "用户使用'不会...吧'反问句式，表达对长时间排队的担心和焦虑。"
}

【示例2 - 不满型反问】:
原始Query: "门票怎么这么贵？"
分析:
{
  "is_rhetorical": true,
  "emotion": "complaining",
  "emotion_intensity": "medium",
  "rewritten_query": "上海迪士尼乐园的门票价格是多少？门票价格包含哪些内容和服务？",
  "original_query": "门票怎么这么贵？",
  "empathy_keywords": ["确实相对较高", "但是包含", "性价比"],
  "reasoning": "用户使用'怎么这么贵'表达对价格的不满。"
}

【示例3 - 非反问句】:
原始Query: "迪士尼门票多少钱？"
分析:
{
    "is_rhetorical": false,
    "emotion": "neutral",
    "emotion_intensity": "none",
    "rewritten_query": "迪士尼门票多少钱？",
    "original_query": "迪士尼门票多少钱？",
    "empathy_keywords": [],
    "reasoning": "正常的疑问句，无情绪色彩，无需转换。"
}

【输出格式】:
必须返回JSON格式，包含所有必需字段
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
                "is_rhetorical": False,
                "emotion": "neutral",
                "emotion_intensity": "none",
                "rewritten_query": query,
                "original_query": query,
                "empathy_keywords": [],
                "reasoning": f"解析失败: {str(e)}"
            }