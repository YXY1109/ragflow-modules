from excel_parser import RAGFlowExcelParser
from qa_table import QAExcel, TableExcel
import pandas as pd
from io import BytesIO
import os


def test_ragflow_excel_parser():
    """测试基础Excel解析器功能"""
    print("=" * 50)
    print("测试 RAGFlowExcelParser 功能")
    print("=" * 50)

    parser = RAGFlowExcelParser()

    # 创建示例数据
    data = {
        '姓名': ['张三', '李四', '王五'],
        '年龄': [25, 30, 28],
        '城市': ['北京', '上海', '广州']
    }
    df = pd.DataFrame(data)

    # 测试HTML输出
    print("\n1. 测试HTML输出功能:")
    excel_data = BytesIO()
    with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='员工信息', index=False)

    excel_data.seek(0)
    html_chunks = parser.html(excel_data, chunk_rows=10)
    for i, chunk in enumerate(html_chunks):
        print(f"HTML Chunk {i+1} (前200字符): {chunk[:200]}...")

    # 测试Markdown输出
    print("\n2. 测试Markdown输出功能:")
    excel_data.seek(0)
    markdown_output = parser.markdown(excel_data)
    print("Markdown输出:")
    print(markdown_output)

    # 测试基本解析
    print("\n3. 测试基本解析功能:")
    excel_data.seek(0)
    parsed_content = parser(excel_data)
    print("解析结果:")
    for line in parsed_content:
        print(f"  {line}")


def test_qa_excel():
    """测试QA Excel解析器功能"""
    print("\n" + "=" * 50)
    print("测试 QAExcel 功能")
    print("=" * 50)

    qa_parser = QAExcel()

    # 创建问答对数据
    qa_data = [
        ['问题', '答案'],
        ['什么是人工智能？', '人工智能是研究如何让计算机模拟人类智能的技术'],
        ['机器学习是什么？', '机器学习是人工智能的一个分支，让计算机从数据中学习'],
        ['深度学习的优势？', '深度学习可以处理复杂的非线性问题']
    ]

    df_qa = pd.DataFrame(qa_data[1:], columns=qa_data[0])

    # 测试QA提取
    print("\n1. 测试问答对提取:")
    def progress_callback(progress, message):
        print(f"  进度: {progress:.1%} - {message}")

    excel_data = BytesIO()
    with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
        df_qa.to_excel(writer, sheet_name='问答库', index=False)

    excel_data.seek(0)
    qa_pairs = qa_parser(excel_data, callback=progress_callback)

    print(f"\n提取到 {len(qa_pairs)} 个问答对:")
    for i, (q, a) in enumerate(qa_pairs, 1):
        print(f"  Q{i}: {q}")
        print(f"  A{i}: {a}")
        print()


def test_table_excel():
    """测试Table Excel解析器功能"""
    print("\n" + "=" * 50)
    print("测试 TableExcel 功能")
    print("=" * 50)

    table_parser = TableExcel()

    # 创建表格数据
    table_data = [
        ['产品名称', '价格', '库存', '类别'],
        ['iPhone 15', '5999', '100', '手机'],
        ['MacBook Pro', '12999', '50', '电脑'],
        ['iPad Air', '4799', '80', '平板'],
        ['AirPods Pro', '1999', '200', '耳机']
    ]

    df_table = pd.DataFrame(table_data[1:], columns=table_data[0])

    print("\n1. 测试表格数据提取:")
    def progress_callback(progress, message):
        print(f"  进度: {progress:.1%} - {message}")

    excel_data = BytesIO()
    with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
        df_table.to_excel(writer, sheet_name='产品信息', index=False)

    excel_data.seek(0)
    dataframes = table_parser(excel_data, callback=progress_callback)

    print(f"\n提取到 {len(dataframes)} 个表格:")
    for i, df in enumerate(dataframes, 1):
        print(f"\n表格 {i}:")
        print(df.to_string(index=False))
        print(f"行数: {len(df)}, 列数: {len(df.columns)}")


def create_sample_files():
    """创建示例Excel文件用于测试"""
    print("创建示例Excel文件...")

    # QA数据文件
    qa_data = [
        ['问题', '答案'],
        ['Python是什么？', 'Python是一种高级编程语言'],
        ['什么是机器学习？', '机器学习是让计算机从数据中学习的技术'],
        ['深度学习与机器学习的区别？', '深度学习是机器学习的一个子领域']
    ]
    df_qa = pd.DataFrame(qa_data[1:], columns=qa_data[0])

    # 表格数据文件
    table_data = [
        ['学生姓名', '数学成绩', '英语成绩', '班级'],
        ['张小明', '95', '88', '一班'],
        ['李小红', '87', '92', '一班'],
        ['王小刚', '78', '85', '二班'],
        ['刘小美', '92', '90', '二班']
    ]
    df_table = pd.DataFrame(table_data[1:], columns=table_data[0])

    # 保存文件
    with pd.ExcelWriter('sample_qa.xlsx', engine='openpyxl') as writer:
        df_qa.to_excel(writer, sheet_name='问答库', index=False)

    with pd.ExcelWriter('sample_table.xlsx', engine='openpyxl') as writer:
        df_table.to_excel(writer, sheet_name='成绩单', index=False)

    print("示例文件已创建: sample_qa.xlsx, sample_table.xlsx")


def test_with_files():
    """使用文件进行测试"""
    print("\n" + "=" * 50)
    print("使用文件进行测试")
    print("=" * 50)

    # 如果文件不存在，先创建
    if not os.path.exists('sample_qa.xlsx') or not os.path.exists('sample_table.xlsx'):
        create_sample_files()

    qa_parser = QAExcel()
    table_parser = TableExcel()
    base_parser = RAGFlowExcelParser()

    print("\n1. 从文件测试QA解析:")
    def progress_callback(progress, message):
        print(f"  {message}")

    try:
        with open('sample_qa.xlsx', 'rb') as f:
            qa_result = qa_parser('sample_qa.xlsx', binary=f.read(), callback=progress_callback)
        print(f"  成功提取 {len(qa_result)} 个问答对")
    except Exception as e:
        print(f"  QA解析失败: {e}")

    print("\n2. 从文件测试表格解析:")
    try:
        with open('sample_table.xlsx', 'rb') as f:
            table_result = table_parser('sample_table.xlsx', binary=f.read(), callback=progress_callback)
        print(f"  成功提取 {len(table_result)} 个表格")
        for i, df in enumerate(table_result):
            print(f"    表格{i+1}: {len(df)}行 x {len(df.columns)}列")
    except Exception as e:
        print(f"  表格解析失败: {e}")

    print("\n3. 从文件测试基础解析:")
    try:
        with open('sample_table.xlsx', 'rb') as f:
            base_result = base_parser(f.read())
        print(f"  成功解析，共 {len(base_result)} 行数据")
        for line in base_result[:3]:  # 只显示前3行
            print(f"    {line}")
    except Exception as e:
        print(f"  基础解析失败: {e}")


if __name__ == '__main__':
    """
    测试Excel解析器的三种功能:
    1. RAGFlowExcelParser - 基础解析器，支持HTML/Markdown输出
    2. QAExcel - 问答对提取器，从Excel中提取问题和答案
    3. TableExcel - 表格数据提取器，将Excel转换为DataFrame

    one和naive是直接使用的
    qa和table有继承关系
    """

    print("开始测试Excel解析器的功能模块")

    try:
        # 测试基础解析器
        test_ragflow_excel_parser()

        # 测试QA解析器
        test_qa_excel()

        # 测试表格解析器
        test_table_excel()

        # 使用文件测试
        test_with_files()

        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("=" * 50)

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
