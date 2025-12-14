from parser.pdf.plain.plain_parser import PlainParser

# 设置PDF文件路径
filename = r"D:\PycharmProjects\ragflow-modules\files\wenben.pdf"

print(f"开始解析PDF文件: {filename}")

pdf_parser = PlainParser()
# 解析PDF（从第0页到第100页）
sections, tables = pdf_parser(filename, from_page=0, to_page=100)

print(f"\n解析完成！")
print(f"解析到 {len(sections)} 个段落")
print(f"解析到 {len(tables)} 个表格")

# 打印前几个段落的内容
print("\n前5个段落内容预览：")
for i, section in enumerate(sections[:10]):
    print(f"\n段落 {i + 1}:")
    print(f"内容: {section[:200]}...")
