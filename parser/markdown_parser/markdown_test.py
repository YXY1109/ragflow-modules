import re
from functools import reduce
from parser.markdown_parser import concat_img, tokenize_table
from parser.markdown_parser.markdown_parser import MarkdownBase


def main():
    """
    源码路径：/Users/cj/PycharmProjects/ragflow/rag/app/naive.py 499
    :return:
    """
    filename = r"/Users/cj/PycharmProjects/ragflow-modules/files/myself/demo1.md"
    markdown_parser = MarkdownBase(128)
    sections, tables = markdown_parser(filename, None, separate_tables=False)

    # 一些初始化
    is_english = True  # is_english(cks)
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])

    # Process images for each section
    section_images = []
    for section_text, _ in sections:
        images = markdown_parser.get_pictures(section_text) if section_text else None
        print(f"images:{images}")
        if images:
            # If multiple images found, combine them using concat_img
            combined_image = reduce(concat_img, images) if len(images) > 1 else images[0]
            section_images.append(combined_image)
            section_images.append(images)
        else:
            section_images.append(None)

    res = tokenize_table(tables, doc, is_english)
    print(f"res:{res}")
    print("Finish parsing.")


if __name__ == '__main__':
    main()
