# ===================
# 上下文依赖型Query改写器
# ===================
from .utils import BaseQueryRewriter, get_completion

class ContextDependentQueryRewriter(BaseQueryRewriter):
    """上下文依赖型Query改写器"""

    def rewrite(self, current_query: str, conversation_history: str = "") -> str:
        """将依赖上下文的查询改写为独立完整的查询"""
        instruction = """
        你是一个智能的查询优化助手。
        【任务】:分析用户的当前问题是否依赖于对话历史，如果依赖则补全信息。
        【识别标志】:
        - "还有"、"其他"、"更多"需要知道相对于什么
        - "也"、"另外"需要补充主体信息
        - 问题很短但需要上下文才能理解
        【改写步骤】:
        1. 从对话历史中提取：主题、对象、地点、已提到的内容
        2. 将这些信息补充到当前问题中
        3. 确保改写后的问题完全独立，不需要任何上下文即可理解
        """
        prompt = f"""
        ## 指令 ##
{instruction}

        ## 对话历史 ##
{conversation_history}

        ## 当前问题 ##
{current_query}

        ## 改写后的问题 ##
        """
        response = get_completion(prompt, self.model)
        return response.strip()