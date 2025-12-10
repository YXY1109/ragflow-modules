from parser.pdf.vision.llm_service import LLMBundle
from parser.pdf.vision.tenant_llm_service import LLMType
from parser.pdf.vision.vision_parser import VisionParser


def demo_file():
    # 创建vision_model实例
    vision_model = LLMBundle("tenant_id", LLMType.IMAGE2TEXT.value)

    # 创建PDF解析器
    pdf_parser = VisionParser(vision_model=vision_model)

    # 设置PDF文件路径
    filename = r"D:\PycharmProjects\ragflow-modules\files\test.pdf"

    if not filename:
        print("请设置PDF文件路径")
        return

    print(f"开始解析PDF文件: {filename}")

    try:
        # 解析PDF（从第0页到第100页）
        sections, tables = pdf_parser(filename, from_page=0, to_page=100)

        print(f"\n解析完成！")
        print(f"解析到 {len(sections)} 个段落")
        print(f"解析到 {len(tables)} 个表格")

        # 打印前几个段落的内容
        print("\n前5个段落内容预览：")
        for i, section in enumerate(sections[:5]):
            print(f"\n段落 {i + 1}:")
            print(f"内容: {section[:200]}...")

    except Exception as e:
        print(f"解析PDF时出错: {e}")


if __name__ == '__main__':
    demo_file()
