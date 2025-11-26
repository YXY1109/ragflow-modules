from parser.txt_parser.txt_parser import RAGFlowTxtParser

if __name__ == '__main__':
    # 没有继承关系，直接使用的
    parser = RAGFlowTxtParser()

    sections1 = parser("demo.txt", chunk_token_num=32)
    txt = "这是一个纯文本解析示例。"
    sections2 = parser.parser_txt(txt)
    print(sections1)
    print(sections2)
