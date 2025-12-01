#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query改写系统综合测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_rewrite import AutoQueryRewriter

def test_query_rewrite_system():
    """测试Query改写系统"""
    print("="*80)
    print("Query改写系统测试")
    print("="*80)

    # 创建改写器实例
    rewriter = AutoQueryRewriter()

    # 测试案例
    test_cases = [
        {
            "name": "反问型 + 模糊指代型",
            "history": "用户: \"创极速光轮是什么项目？\" AI: \"创极速光轮是明日世界的过山车，全球最快\"",
            "query": "它不会要排队2小时吧？",
            "expected": ["rhetorical", "reference"]
        },
        {
            "name": "多意图型查询",
            "history": "用户第一次去迪士尼",
            "query": "门票价格？开放时间？怎么去？",
            "expected": ["multi_intent"]
        },
        {
            "name": "上下文依赖型 + 对比型",
            "history": "用户: \"我想带孩子去迪士尼\" AI: \"推荐疯狂动物城和宝藏湾\"",
            "query": "还有其他适合的园区吗？哪个更好？",
            "expected": ["context_dependent", "comparison"]
        },
        {
            "name": "简单查询",
            "history": "",
            "query": "迪士尼门票多少钱？",
            "expected": []
        }
    ]

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {test['name']}")
        print("-" * 50)
        print(f"对话历史: {test['history']}")
        print(f"原始查询: {test['query']}")
        print(f"预期类型: {test['expected']}")

        # 执行改写
        result = rewriter.rewrite(test['query'], test['history'])

        # 显示结果
        print(f"最终改写: {result['final_query']}")
        print(f"识别类型: {result['query_types']}")
        print(f"情绪分析: {result['emotion']['emotion']} ({result['emotion']['emotion_intensity']})")

        # 验证结果
        if set(result['query_types']) == set(test['expected']):
            print("[PASS] 测试通过")
        else:
            print("[FAIL] 测试失败")
            all_passed = False

        # 显示改写步骤
        if result['rewrite_steps']:
            print("改写步骤:")
            for step in result['rewrite_steps']:
                print(f"  -> {step['type']}")

        # 显示子查询（多意图型）
        if result['sub_queries']:
            print("拆分的子查询:")
            for j, sub in enumerate(result['sub_queries'], 1):
                print(f"  [{j}] {sub['final_query']} (优先级: {sub.get('priority', 'unknown')})")

    print("\n" + "="*80)
    if all_passed:
        print("[SUCCESS] 所有测试通过！")
        print("Query改写系统功能正常。")
    else:
        print("[ERROR] 部分测试失败，请检查系统配置。")
    print("="*80)

    return all_passed

if __name__ == "__main__":
    test_query_rewrite_system()