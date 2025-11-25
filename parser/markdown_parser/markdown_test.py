from parser.markdown_parser.markdown_base import Markdown


def main_chunk():
    """
    源码路径：/Users/cj/PycharmProjects/ragflow/rag/app/naive.py 499
    :return:
    """
    filename = r"/Users/cj/PycharmProjects/ragflow-modules/files/myself/demo1.md"
    filename = r"D:\PycharmProjects\ragflow-modules\files\markdown\myself\demo1.md"
    filename = r"D:\PycharmProjects\ragflow-modules\files\markdown\myself\格力2023年年报.md"
    markdown_parser = Markdown(128)
    sections, tables = markdown_parser(filename, separate_tables=False,
                                       delimiter="\n!?;。；！？")
    print(sections)
    print(tables)
    print(111)


if __name__ == '__main__':
    main_chunk()
