# -*- coding: utf-8 -*-
import json
import sys
import os



from parser.json_parser.json_parser import RAGFlowJsonParser


def test_json_parser():
    """测试 JSON 解析器的各种功能"""

    # 创建 JSON 解析器实例，最大块大小为 2000 字符
    json_parser = RAGFlowJsonParser(max_chunk_size=2000)

    print("=== JSON 解析器测试 ===\n")

    # 测试 1: 基本对象解析
    print("1. 测试基本 JSON 对象:")
    simple_json = {
        "name": "张三",
        "age": 30,
        "city": "北京",
        "skills": ["Python", "JavaScript", "机器学习"]
    }

    json_str = json.dumps(simple_json, ensure_ascii=False)
    chunks = json_parser.split_text(json_data=simple_json, convert_lists=True)

    for i, chunk in enumerate(chunks):
        print(f"  块 {i+1}: {chunk}")
    print()

    # 测试 2: 复杂嵌套对象
    print("2. 测试复杂嵌套 JSON:")
    complex_json = {
        "company": "科技公司",
        "employees": [
            {
                "id": 1,
                "name": "李四",
                "department": "研发部",
                "projects": [
                    {"name": "项目A", "status": "进行中"},
                    {"name": "项目B", "status": "已完成"}
                ]
            },
            {
                "id": 2,
                "name": "王五",
                "department": "市场部",
                "projects": [
                    {"name": "营销活动", "status": "计划中"}
                ]
            }
        ],
        "metadata": {
            "created": "2025-01-01",
            "version": "1.0"
        }
    }

    chunks = json_parser.split_text(json_data=complex_json, convert_lists=True)
    for i, chunk in enumerate(chunks):
        print(f"  块 {i+1}: {chunk}")
    print()

    # 测试 3: 使用二进制输入
    print("3. 测试二进制输入:")
    test_data = {
        "title": "产品文档",
        "content": "这是一个产品说明文档",
        "tags": ["文档", "产品", "说明"],
        "sections": [
            {"title": "介绍", "content": "产品介绍内容"},
            {"title": "使用方法", "content": "详细使用说明"}
        ]
    }

    # 转换为二进制
    json_bytes = json.dumps(test_data, ensure_ascii=False).encode('utf-8')

    # 使用 __call__ 方法解析
    result_chunks = json_parser(json_bytes)
    for i, chunk in enumerate(result_chunks):
        print(f"  块 {i+1}: {chunk}")
    print()

    # 测试 4: JSONL 格式测试
    print("4. 测试 JSONL 格式:")
    jsonl_content = '''{"id": 1, "name": "用户1", "action": "登录"}
{"id": 2, "name": "用户2", "action": "浏览"}
{"id": 3, "name": "用户1", "action": "购买"}'''

    jsonl_bytes = jsonl_content.encode('utf-8')
    jsonl_chunks = json_parser(jsonl_bytes)
    for i, chunk in enumerate(jsonl_chunks):
        print(f"  块 {i+1}: {chunk}")
    print()

    # 测试 5: 大数据分块测试
    print("5. 测试大数据分块:")
    large_data = {}
    for i in range(100):
        large_data[f"item_{i}"] = {
            "id": i,
            "name": f"项目{i}",
            "description": f"这是第{i}个项目的详细描述，包含足够的内容来测试分块功能",
            "tags": [f"标签{i}", f"类别{i%5}", f"类型{i%3}"],
            "metadata": {
                "created": f"2025-01-{(i%28)+1:02d}",
                "updated": f"2025-01-{(i%28)+1:02d}",
                "priority": i % 4
            }
        }

    large_chunks = json_parser.split_text(json_data=large_data, convert_lists=True)
    print(f"  总共分割成 {len(large_chunks)} 个块")
    for i, chunk in enumerate(large_chunks[:3]):  # 只显示前3个块
        print(f"  块 {i+1} (长度: {len(chunk)}): {chunk[:100]}...")
    print(f"  最后一个块 (长度: {len(large_chunks[-1])}): {large_chunks[-1][:100]}...")
    print()


def demo_json_file_processing():
    """演示处理 JSON 文件的完整流程"""

    print("=== JSON 文件处理演示 ===\n")

    # 创建示例 JSON 数据
    sample_data = {
        "文档信息": {
            "标题": "RAGFlow 模块学习指南",
            "作者": "开发者",
            "版本": "1.0.0",
            "创建日期": "2025-01-26"
        },
        "模块列表": [
            {
                "名称": "文件解析器",
                "功能": ["PDF解析", "Word解析", "Markdown解析", "Excel解析"],
                "状态": "已完成"
            },
            {
                "名称": "NLP处理器",
                "功能": ["中文分词", "Token计算", "文本合并"],
                "状态": "开发中"
            },
            {
                "名称": "动态注册系统",
                "功能": ["模型注册", "处理器注册", "自动发现"],
                "状态": "已完成"
            }
        ],
        "使用统计": {
            "总模块数": 3,
            "已完成": 2,
            "开发中": 1,
            "测试覆盖率": "85%"
        }
    }

    # 将数据写入文件
    json_file_path = "sample_data.json"
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"创建示例 JSON 文件: {json_file_path}")

    # 读取并解析文件
    with open(json_file_path, 'rb') as f:
        binary_data = f.read()

    # 创建解析器并处理
    parser = RAGFlowJsonParser(max_chunk_size=150)
    chunks = parser(binary_data)

    print(f"\n解析结果 (共 {len(chunks)} 个块):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- 块 {i} ---")
        print(chunk)

    # 清理文件
    # import os
    # os.remove(json_file_path)
    # print(f"\n已清理临时文件: {json_file_path}")


if __name__ == '__main__':
    # 运行基本测试
    test_json_parser()

    print("\n" + "="*50 + "\n")

    # 运行文件处理演示
    demo_json_file_processing()
