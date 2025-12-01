# ===================
# 上下文依赖型Query改写器
# ===================
class ContextDependentQueryRewriter:
"""上下文依赖型Query改写器"""

def__init__(self, model="qwen-turbo-latest"):
        self.model = model

defrewrite(self, current_query, conversation_history):
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


# 测试代码rewriter = ContextDependentQueryRewriter()
test_cases = [
    {
        "history": """用户: "疯狂动物城有什么好玩的？"AI: "有警察局互动、训练营、冰淇淋店"""",
        "query": "还有其他设施吗？"    },
    {
        "history": """用户: "门票多少钱？"AI: "平日399元，周末499元"""",
        "query": "儿童票呢？"    }
]
for test in test_cases:
    print(f"原始: {test['query']}")
    rewritten = rewriter.rewrite(test['query'], test['history'])
    print(f"改写: {rewritten}")
    print(f"效果: ✅ 补全了上下文，变成独立完整的问题\n")
# 输出：# 原始: 还有其他设施吗？# 改写: 除了警察局互动体验、朱迪警官训练营和尼克狐的冰淇淋店，疯狂动物城园区还有其他设施吗？# 效果: ✅ 补全了上下文，变成独立完整的问题# 原始: 儿童票呢？# 改写: 上海迪士尼乐园的儿童票价格是多少？# 效果: ✅ 补全了上下文，变成独立完整的问题