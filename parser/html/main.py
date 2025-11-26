# -*- coding: utf-8 -*-
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from parser.html.html_parser import RAGFlowHtmlParser


def test_html_parser():
    """测试 HTML 解析器的各种功能"""

    # 创建 HTML 解析器实例
    html_parser = RAGFlowHtmlParser()

    print("=== HTML 解析器测试 ===\n")

    # 测试 1: 基本 HTML 解析
    print("1. 测试基本 HTML 文档:")
    simple_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>测试文档</title>
        <style>
            body { font-family: Arial; }
            .test { color: red; }
        </style>
        <script>
            console.log("这个脚本应该被删除");
        </script>
    </head>
    <body>
        <h1>主标题</h1>
        <p>这是一个段落。</p>
        <div>
            <h2>子标题</h2>
            <p>这是另一个段落，包含<strong>粗体文本</strong>和<em>斜体文本</em>。</p>
        </div>
        <!-- 这是HTML注释，应该被删除 -->
    </body>
    </html>
    """

    chunks = html_parser.parser_txt(simple_html, chunk_token_num=100)
    print(f"  解析得到 {len(chunks)} 个块:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  块 {i}: {chunk}")
    print()

    # 测试 2: 包含表格的 HTML
    print("2. 测试包含表格的 HTML:")
    table_html = """
    <html>
    <body>
        <h1>销售报告</h1>
        <p>以下是2024年的销售数据：</p>
        <table border="1">
            <tr>
                <th>产品</th>
                <th>销量</th>
                <th>收入</th>
            </tr>
            <tr>
                <td>产品A</td>
                <td>1000</td>
                <td>¥10,000</td>
            </tr>
            <tr>
                <td>产品B</td>
                <td>1500</td>
                <td>¥15,000</td>
            </tr>
            <tr>
                <td>产品C</td>
                <td>800</td>
                <td>¥8,000</td>
            </tr>
        </table>
        <p>总结：总销售额为¥33,000。</p>
    </body>
    </html>
    """

    chunks = html_parser.parser_txt(table_html, chunk_token_num=150)
    print(f"  解析得到 {len(chunks)} 个块:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  块 {i}: {chunk}")
    print()

    # 测试 3: 复杂嵌套结构
    print("3. 测试复杂嵌套结构:")
    complex_html = """
    <html>
    <body>
        <article>
            <h1>技术文档</h1>
            <section>
                <h2>第一章：简介</h2>
                <p>这是一个技术文档的简介部分。</p>
                <blockquote>
                    "好的代码是最好的文档。"
                </blockquote>
            </section>
            <section>
                <h2>第二章：安装</h2>
                <p>安装步骤如下：</p>
                <ol>
                    <li>下载安装包</li>
                    <li>运行安装程序</li>
                    <li>配置环境</li>
                </ol>
                <pre><code>
# 示例代码
import html_parser
parser = RAGFlowHtmlParser()
result = parser("file.html")
                </code></pre>
            </section>
        </article>
    </body>
    </html>
    """

    chunks = html_parser.parser_txt(complex_html, chunk_token_num=200)
    print(f"  解析得到 {len(chunks)} 个块:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  块 {i}: {chunk}")
    print()


def demo_html_file_processing():
    """演示处理 HTML 文件的完整流程"""

    print("=== HTML 文件处理演示 ===\n")

    # 创建示例 HTML 文件内容
    sample_html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>RAGFlow 学习指南</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .feature { background: #e8f4fd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #007bff; color: white; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>RAGFlow 模块学习指南</h1>

    <section id="modules">
        <h2>核心模块</h2>

        <h3>1. 文件解析器模块</h3>
        <div class="feature">
            <p>支持多种文档格式的解析和处理：</p>
            <ul>
                <li><code>TXT</code> - 纯文本和邮件格式</li>
                <li><code>PDF</code> - PDF文档解析</li>
                <li><code>Word</code> - Word文档处理</li>
                <li><code>Excel</code> - 表格数据提取</li>
                <li><code>Markdown</code> - 结构化文档解析</li>
                <li><code>HTML</code> - 网页内容提取</li>
                <li><code>JSON</code> - 数据格式解析</li>
            </ul>
        </div>

        <h3>2. NLP处理模块</h3>
        <div class="feature">
            <p>提供强大的自然语言处理能力：</p>
            <ul>
                <li>基于HUQIE的中英文分词器</li>
                <li>Token计算和分析</li>
                <li>文本合并和预处理</li>
                <li>词频统计和词性标注</li>
            </ul>
        </div>
    </section>

    <section id="comparison">
        <h2>性能对比</h2>
        <p>以下是不同解析器模块的性能对比：</p>

        <table>
            <thead>
                <tr>
                    <th>解析器</th>
                    <th>支持格式</th>
                    <th>处理速度</th>
                    <th>内存占用</th>
                    <th>稳定性</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>TXT解析器</td>
                    <td>文本文件</td>
                    <td>极快</td>
                    <td>低</td>
                    <td>稳定</td>
                </tr>
                <tr>
                    <td>Markdown解析器</td>
                    <td>MD文件</td>
                    <td>快速</td>
                    <td>中</td>
                    <td>稳定</td>
                </tr>
                <tr>
                    <td>JSON解析器</td>
                    <td>JSON/JSONL</td>
                    <td>快速</td>
                    <td>中</td>
                    <td>稳定</td>
                </tr>
                <tr>
                    <td>HTML解析器</td>
                    <td>网页文件</td>
                    <td>极快</td>
                    <td>中高</td>
                    <td>稳定</td>
                </tr>
                <tr>
                    <td>PPT解析器</td>
                    <td>演示文件</td>
                    <td>较慢</td>
                    <td>高</td>
                    <td>开发中</td>
                </tr>
            </tbody>
        </table>
    </section>

    <footer>
        <hr>
        <p>© 2025 RAGFlow模块学习项目. 本项目遵循 Apache 2.0 许可证。</p>
    </footer>
</body>
</html>"""

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(sample_html)
        temp_file_path = f.name

    try:
        print(f"创建示例 HTML 文件: {temp_file_path}")

        # 使用HTML解析器处理文件
        html_parser = RAGFlowHtmlParser()
        chunks = html_parser(temp_file_path, chunk_token_num=300)

        print(f"\n解析结果 (共 {len(chunks)} 个块):")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- 块 {i} ---")
            print(chunk)

        # 测试二进制输入方式
        print(f"\n{'=' * 50}")
        print("测试二进制输入方式:")

        with open(temp_file_path, 'rb') as f:
            binary_data = f.read()

        binary_chunks = html_parser(None, binary=binary_data, chunk_token_num=300)
        print(f"通过二进制输入解析得到 {len(binary_chunks)} 个块")

        # 验证两种方式的结果是否一致
        if len(chunks) == len(binary_chunks):
            print("✅ 两种解析方式的结果一致")
        else:
            print("⚠️ 两种解析方式的结果块数不同")

    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"\n已清理临时文件: {temp_file_path}")


def test_table_splitting():
    """专门测试表格分割功能"""

    print("\n=== 表格分割测试 ===\n")

    # 创建一个大表格用于测试分割功能
    large_table = """<table border="1">
        <tr><th>ID</th><th>姓名</th><th>部门</th><th>职位</th></tr>"""

    # 添加20行数据
    for i in range(1, 21):
        large_table += f"""
        <tr>
            <td>{i}</td>
            <td>员工{i}</td>
            <td>{'技术' if i % 2 == 0 else '市场'}部</td>
            <td>{'工程师' if i % 3 == 0 else '专员'}</td>
        </tr>"""

    large_table += "</table>"

    html_parser = RAGFlowHtmlParser()
    table_chunks = html_parser.split_table(large_table, chunk_token_num=100)

    print(f"大表格被分割成 {len(table_chunks)} 个小表格:")
    for i, table in enumerate(table_chunks, 1):
        print(f"\n表格片段 {i}:")
        print(table[:200] + "..." if len(table) > 200 else table)


if __name__ == '__main__':
    """
    都是直接使用，没有继承关系
    """

    # 运行基本测试
    test_html_parser()

    print("\n" + "=" * 50 + "\n")

    # 运行文件处理演示
    demo_html_file_processing()

    print("\n" + "=" * 50 + "\n")

    # 运行表格分割测试
    test_table_splitting()

    print("\n✅ HTML解析器测试完成！")
