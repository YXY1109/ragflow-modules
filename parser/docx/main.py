from pathlib import Path


def test_docx_parser():
    """测试基础docx_parser.py解析器"""
    print("\n" + "=" * 60)
    print("测试 RAGFlowDocxParser (docx_parser.py)")
    print("=" * 60)

    from parser.docx.docx_parser import RAGFlowDocxParser

    # 获取测试文件
    test_files_dir = Path("files")
    test_files = list(test_files_dir.glob("*.docx"))

    if not test_files:
        print("X 没有找到测试文件")
        return

    parser = RAGFlowDocxParser()

    for file_path in test_files:
        print(f"\n* 测试文件: {file_path.name}")
        try:
            sections, tables = parser(str(file_path))
            print(f"+ 解析成功")
            print(f"   - 段落数量: {len(sections)}")
            print(f"   - 表格数量: {len(tables)}")

            if sections:
                print(f"   - 前3个段落内容:")
                for i, (text, style) in enumerate(sections[:3]):
                    print(f"     {i + 1}. [{style}] {text[:50]}...")

            if tables:
                print(f"   - 表格内容:")
                for i, table in enumerate(tables[:2]):  # 只显示前2个表格
                    print(f"     表格{i + 1}: {table[:100]}...")

        except Exception as e:
            print(f"X 解析失败: {e}")


def test_laws_parser():
    """测试laws.py解析器"""
    print("\n" + "=" * 60)
    print("测试 Laws Parser")
    print("=" * 60)

    from parser.docx.laws import Docx

    test_files_dir = Path("files")
    test_files = list(test_files_dir.glob("*.docx"))

    if not test_files:
        print("X 没有找到测试文件")
        return

    parser = Docx()

    for file_path in test_files:
        print(f"\n* 测试文件: {file_path.name}")
        try:
            result = parser(str(file_path))
            print(f"+ 解析成功")
            print(f"   - 结果类型: {type(result)}")
            if isinstance(result, list):
                print(f"   - 元素数量: {len(result)}")
                for i, item in enumerate(result[:3]):  # 显示前3个元素
                    print(f"     {i + 1}. {str(item)[:100]}...")

        except Exception as e:
            print(f"X 解析失败: {e}")


def test_manual_parser():
    """测试manual.py解析器"""
    print("\n" + "=" * 60)
    print("测试 Manual Parser")
    print("=" * 60)

    from parser.docx.manual import Docx

    test_files_dir = Path("files")
    test_files = list(test_files_dir.glob("*.docx"))

    if not test_files:
        print("X 没有找到测试文件")
        return

    parser = Docx()

    for file_path in test_files:
        print(f"\n* 测试文件: {file_path.name}")
        try:
            text_image_list, tables = parser(str(file_path))
            print(f"+ 解析成功")
            print(f"   - 文本+图片对数量: {len(text_image_list)}")
            print(f"   - 表格数量: {len(tables)}")

            for i, (text, image) in enumerate(text_image_list[:2]):  # 显示前2个
                print(f"     {i + 1}. 文本: {text[:80]}...")
                print(f"        图片: {'有' if image else '无'}")

        except Exception as e:
            print(f"X 解析失败: {e}")


def test_naive_parser():
    """测试naive.py解析器"""
    print("\n" + "=" * 60)
    print("测试 Naive Parser")
    print("=" * 60)

    from parser.docx.naive import Docx

    test_files_dir = Path("files")
    test_files = list(test_files_dir.glob("*.docx"))

    if not test_files:
        print("X 没有找到测试文件")
        return

    parser = Docx()

    for file_path in test_files:
        print(f"\n* 测试文件: {file_path.name}")
        try:
            sections, tables = parser(str(file_path))
            print(f"+ 解析成功")
            print(f"   - 段落数量: {len(sections)}")
            print(f"   - 表格数量: {len(tables)}")

            for i, (text, image, style) in enumerate(sections[:3]):  # 显示前3个
                print(f"     {i + 1}. [{style}] {text[:60]}...")
                print(f"        图片: {'有' if image else '无'}")

        except Exception as e:
            print(f"X 解析失败: {e}")


def test_qa_parser():
    """测试qa.py解析器"""
    print("\n" + "=" * 60)
    print("测试 QA Parser")
    print("=" * 60)

    from parser.docx.qa import Docx

    test_files_dir = Path("files")
    test_files = list(test_files_dir.glob("*.docx"))

    if not test_files:
        print("X 没有找到测试文件")
        return

    parser = Docx()

    for file_path in test_files:
        print(f"\n* 测试文件: {file_path.name}")
        try:
            qai_list, tables = parser(str(file_path))
            print(f"+ 解析成功")
            print(f"   - 问答对数量: {len(qai_list)}")
            print(f"   - 表格数量: {len(tables)}")

            for i, (question, answer, image) in enumerate(qai_list[:3]):  # 显示前3个
                print(f"     {i + 1}. 问题: {question[:60]}...")
                print(f"        答案: {answer[:60]}...")
                print(f"        图片: {'有' if image else '无'}")

        except Exception as e:
            print(f"X 解析失败: {e}")


def main():
    """主测试函数"""
    print("* 开始测试DOCX解析器")

    # 测试各个解析器
    test_docx_parser()
    test_laws_parser()
    test_manual_parser()
    test_naive_parser()
    test_qa_parser()

    print("\n" + "=" * 60)
    print("+ 所有解析器测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
