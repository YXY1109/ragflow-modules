#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query改写系统完整测试和示例
基于OpenRouter API的真实LLM实现
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_rewrite import AutoQueryRewriter, QueryRewriterConfig


def test_basic_query_examples():
    """测试基础查询改写示例"""
    print("=" * 80)
    print("🧪 基础查询改写测试")
    print("=" * 80)

    # 创建配置
    config = QueryRewriterConfig(
        model="glm-4.6",
        timeout_read=60.0,  # 增加读取超时
        timeout_connect=20.0
    )

    # 创建改写器
    rewriter = AutoQueryRewriter(config=config)

    # 测试用例
    test_cases = [
        {
            "query": "它不会要排队2小时吧？",
            "context": "用户: \"创极速光轮是什么项目？\" AI: \"是明日世界的过山车，全球最快\"",
            "expected_types": ["rhetorical", "reference"],
            "description": "反问句 + 模糊指代"
        },
        {
            "query": "上海迪士尼门票多少钱？开放时间是什么？怎么去？",
            "context": "",
            "expected_types": ["multi_intent"],
            "description": "多意图查询"
        },
        {
            "query": "疯狂动物城和宝藏湾哪个更好玩？",
            "context": "用户正在计划上海迪士尼之旅",
            "expected_types": ["comparison"],
            "description": "对比型查询"
        },
        {
            "query": "还有其他适合小孩的项目吗？",
            "context": "用户: \"我们已经玩过创极速光轮和加勒比海盗了\"",
            "expected_types": ["context_dependent"],
            "description": "上下文依赖查询"
        },
        {
            "query": "上海迪士尼乐园门票价格是多少？",
            "context": "",
            "expected_types": [],
            "description": "普通查询（无需改写）"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📍 测试案例 {i}: {test_case['description']}")
        print(f"原始查询: {test_case['query']}")
        if test_case['context']:
            print(f"上下文: {test_case['context']}")

        try:
            # 执行改写
            result = rewriter.rewrite(
                query=test_case['query'],
                conversation_history=test_case['context']
            )

            # 显示结果
            print(f"✅ 识别类型: {', '.join(result.get('query_types', []))}")
            print(f"📝 最终查询: {result.get('final_query', '无')}")
            print(f"🎭 情绪状态: {result.get('emotion', {}).get('emotion', 'neutral')} "
                  f"({result.get('emotion', {}).get('emotion_intensity', 'none')})")

            # 检查是否正确识别了类型
            expected_set = set(test_case['expected_types'])
            actual_set = set(result.get('query_types', []))

            if expected_set == actual_set:
                print("✅ 类型识别正确")
            else:
                print(f"⚠️  类型识别差异 - 期望: {test_case['expected_types']}, 实际: {result.get('query_types', [])}")

            # 显示改写步骤
            if result.get('rewrite_steps'):
                print("🔧 改写步骤:")
                for step in result['rewrite_steps']:
                    print(f"   - {step['type']}: {step.get('rewrite', 'N/A')}")

            # 显示子查询
            if result.get('sub_queries'):
                print("📋 子查询:")
                for j, sub in enumerate(result['sub_queries'], 1):
                    print(f"   [{j}] ({sub.get('priority', 'medium')}) {sub.get('final_query', 'N/A')}")

        except Exception as e:
            print(f"❌ 处理失败: {e}")

        print("-" * 60)


def test_individual_rewriters():
    """测试各个独立的改写器"""
    print("\n" + "=" * 80)
    print("🔧 独立改写器测试")
    print("=" * 80)

    config = QueryRewriterConfig(
        model="glm-4.6",
        timeout_read=45.0
    )

    from query_rewrite import (
        MultiIntentQueryRewriter,
        ReferenceQueryRewriter,
        ComparisonQueryRewriter,
        ContextDependentQueryRewriter,
        RhetoricalQueryRewriter
    )

    # 测试多意图改写器
    print("\n🎯 多意图改写器测试:")
    multi_rewriter = MultiIntentQueryRewriter(config=config)
    multi_result = multi_rewriter.rewrite(
        query="门票价格？开放时间？交通方式？",
        context="用户计划去上海迪士尼"
    )
    print(f"查询: 门票价格？开放时间？交通方式？")
    print(f"意图数量: {multi_result.get('intent_count', 0)}")
    for i, sub in enumerate(multi_result.get('sub_queries', []), 1):
        print(f"  子查询{i} ({sub.get('priority', 'medium')}): {sub.get('query', 'N/A')}")

    # 测试反问改写器
    print("\n💭 反问改写器测试:")
    rhetorical_rewriter = RhetoricalQueryRewriter(config=config)
    rhetorical_result = rhetorical_rewriter.rewrite(
        query="这个项目不会要等很久吧？",
        context="用户在询问排队时间"
    )
    print(f"查询: 这个项目不会要等很久吧？")
    print(f"是否反问: {rhetorical_result.get('is_rhetorical', False)}")
    print(f"情绪: {rhetorical_result.get('emotion', 'neutral')}")
    print(f"改写后: {rhetorical_result.get('rewritten_query', 'N/A')}")

    # 测试指代改写器
    print("\n👉 指代改写器测试:")
    reference_rewriter = ReferenceQueryRewriter(config=config)
    reference_result = reference_rewriter.rewrite(
        current_query="它有多高？",
        conversation_history="用户刚才在询问创极速光轮过山车"
    )
    print(f"查询: 它有多高？")
    print(f"改写后: {reference_result}")

    # 测试对比改写器
    print("\n⚖️  对比改写器测试:")
    comparison_rewriter = ComparisonQueryRewriter(config=config)
    comparison_result = comparison_rewriter.rewrite(
        query="这两个项目哪个更刺激？",
        context_info="用户在比较创极速光轮和加勒比海盗"
    )
    print(f"查询: 这两个项目哪个更刺激？")
    print(f"改写后: {comparison_result}")

    # 测试上下文改写器
    print("\n📚 上下文改写器测试:")
    context_rewriter = ContextDependentQueryRewriter(config=config)
    context_result = context_rewriter.rewrite(
        current_query="还有其他项目吗？",
        conversation_history="用户已经玩了创极速光轮，想了解其他项目"
    )
    print(f"查询: 还有其他项目吗？")
    print(f"改写后: {context_result}")


def test_configuration_options():
    """测试不同配置选项"""
    print("\n" + "=" * 80)
    print("⚙️  配置选项测试")
    print("=" * 80)

    # 测试不同模型
    models_to_test = [
        "glm-4.6",
        # 可以添加其他免费模型进行测试
    ]

    for model in models_to_test:
        print(f"\n🤖 测试模型: {model}")
        try:
            config = QueryRewriterConfig(
                model=model,
                timeout_read=30.0
            )

            rewriter = AutoQueryRewriter(config=config)

            # 简单测试
            result = rewriter.rewrite(
                query="它要排队很久吗？",
                conversation_history="用户在询问创极速光轮项目"
            )

            print(f"  ✅ 模型 {model} 测试成功")
            print(f"  📝 结果: {result.get('final_query', 'N/A')}")

        except Exception as e:
            print(f"  ❌ 模型 {model} 测试失败: {e}")


def demo_usage_patterns():
    """演示不同的使用模式"""
    print("\n" + "=" * 80)
    print("🎭 使用模式演示")
    print("=" * 80)

    # 基本使用
    print("\n1️⃣ 基本使用模式:")
    print("""
from query_rewrite import AutoQueryRewriter

# 创建改写器（使用默认配置）
rewriter = AutoQueryRewriter()

# 改写查询
result = rewriter.rewrite(
    query="它不会要排队2小时吧？",
    conversation_history="用户在询问创极速光轮项目"
)

print(f"改写结果: {result['final_query']}")
""")

    # 自定义配置
    print("2️⃣ 自定义配置模式:")
    print("""
from query_rewrite import AutoQueryRewriter, QueryRewriterConfig

# 自定义配置
config = QueryRewriterConfig(
    model="glm-4.6",
    timeout_read=120.0,  # 增加超时时间
    temperature=0.3      # 降低随机性
)

# 使用自定义配置
rewriter = AutoQueryRewriter(config=config)
""")

    # 批量处理
    print("3️⃣ 批量处理模式:")
    print("""
queries = [
    {"query": "门票价格？时间？", "context": ""},
    {"query": "它好玩吗？", "context": "用户在询问某个项目"},
    {"query": "哪个更好？", "context": "用户在比较两个选项"}
]

for item in queries:
    result = rewriter.rewrite(**item)
    print(f"原: {item['query']} -> 改: {result['final_query']}")
""")

    # 自定义改写器
    print("4️⃣ 自定义改写器模式:")
    print("""
from query_rewrite import AutoQueryRewriter, RhetoricalQueryRewriter, QueryRewriterConfig

# 创建自定义改写器
custom_rhetorical = RhetoricalQueryRewriter(
    model="your-preferred-model",
    config=config
)

# 使用自定义改写器
rewriter = AutoQueryRewriter(
    config=config,
    rhetorical_rewriter=custom_rhetorical
)
""")


def main():
    """主函数"""
    print("🚀 Query改写系统完整测试")
    print("基于OpenRouter API + 真实LLM")

    try:
        # 基础测试
        test_basic_query_examples()

        # 独立改写器测试
        test_individual_rewriters()

        # 配置选项测试
        test_configuration_options()

        # 使用模式演示
        demo_usage_patterns()

        print("\n" + "=" * 80)
        print("✅ 所有测试完成！")
        print("=" * 80)

        print("\n📚 更多信息:")
        print("• 查看query_rewrite/README.md了解详细文档")
        print("• 修改QueryRewriterConfig自定义配置")
        print("• 扩展改写器实现更多功能")

    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生错误: {e}")
        print("请检查网络连接和API配置")


if __name__ == "__main__":
    main()
