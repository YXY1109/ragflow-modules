# ===========
# 模糊指代型Query改写器
# ============
from .utils import BaseQueryRewriter, get_completion, QueryRewriterConfig


class ReferenceQueryRewriter(BaseQueryRewriter):
    """模糊指代型Query改写器

    解决的核心问题:
    - "它"指代什么？
    - "都"包括哪些？
    - "这个"具体是什么？

    技术难点:
    1. 指代词识别（NLP任务）
    2. 指代对象定位（需要理解上下文）
    3. 多义性消解（可能有多个候选对象）
    """

    def __init__(self, model: str = None, config: QueryRewriterConfig = None):
        self.model = model or (config.model if config else "z-ai/glm-4.5-air:free")
        self.config = config or QueryRewriterConfig()

    def rewrite(self, current_query: str, conversation_history: str = "") -> str:
        """消除模糊指代，生成明确的Query

        工作流程:
        1. 识别Query中的所有指代词
        2. 在对话历史中定位指代对象
        3. 用明确的名词替换指代词
        4. 验证改写后的Query语义完整

        参数:
            current_query: 当前查询（包含指代词）
            conversation_history: 对话历史

        返回:
            消歧后的明确查询
        """

        instruction = """你是一个语言歧义消除专家，专门处理指代词问题。

【常见指代词分类】:
1. 单数指代（指代单个对象）:
   • "它"、"他"、"她" → 具体的人或物
   • "这个"、"那个" → 前文提到的事物
   • "该" → 正在讨论的对象

2. 复数指代（指代多个对象）:
   • "都"、"全部"、"所有" → 前文提到的多个对象
   • "它们"、"他们"、"这些"、"那些" → 一组对象

3. 时间指代（指代前文内容）:
   • "刚才"、"之前"、"前面说的" → 前文的话题或对象

【消歧步骤】:
步骤1 - 识别指代词:
扫描Query，标注所有指代词及其位置

步骤2 - 定位指代对象
• 对于单数指代: 找最近提到的同类名词
• 对于复数指代: 找所有相关的名词并列举
• 对于时间指代: 定位到具体的对话内容

步骤3 - 验证指代关系
• 检查名词类型是否匹配（项目/园区/服务等）
• 检查语义是否合理
• 处理多义性（如果有多个候选对象，选择最可能的）

步骤4 - 执行替换
• 用明确的名词替换指代词
• 保持句子的流畅性
• 确保语义完整无歧义

【示例1 - 单数指代】:
对话历史:
  用户: "创极速光轮是什么项目？"
  AI: "创极速光轮是明日世界主题园区的过山车项目，是全球最快的迪士尼过山车。"

当前Query: "它有身高要求吗？"

分析:
  • 指代词: "它"
  • 指代对象: "创极速光轮"（最近提到的项目名词）
  • 验证: ✅ 类型匹配（都是游乐项目）

改写: "创极速光轮过山车项目有身高要求吗？"

【示例2 - 复杂指代】:
对话历史:
  用户: "我看到有个海盗船的项目"
  AI: "您说的应该是宝藏湾的'加勒比海盗：战争之潮'，这是全球首个海盗主题迪士尼项目。"
  用户: "这个项目好玩吗？"
  AI: "非常好玩！采用了先进的机器人和投影技术。"

当前Query: "那个适合小朋友吗？"

分析:
  • 指代词: "那个"
  • 需要追溯: 对话已经进行了3轮
  • 指代对象: "加勒比海盗：战争之潮"（初始提到的项目）

改写: "宝藏湾的加勒比海盗：战争之潮项目适合小朋友吗？"""

        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前查询（包含指代词）###
{current_query}

### 消歧后的查询 ###
"""
        response = get_completion(prompt, self.model)
        return response.strip()
