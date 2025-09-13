import re
from functools import reduce
from timeit import default_timer as timer

from nlp import rag_tokenizer
from nlp.merge import naive_merge_with_images, tokenize_chunks_with_images, naive_merge, tokenize_chunks
from parser.markdown_parser import concat_img, tokenize_table
from parser.markdown_parser.markdown_parser import MarkdownBase


def main_chunk():
    """
    源码路径：/Users/cj/PycharmProjects/ragflow/rag/app/naive.py 499
    :return:
    """
    filename = r"/Users/cj/PycharmProjects/ragflow-modules/files/myself/demo1.md"
    filename = r"D:\PycharmProjects\ragflow-modules\files\markdown\myself\demo1.md"
    filename = r"D:\PycharmProjects\ragflow-modules\files\markdown\myself\格力2023年年报.md"
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
        # print(f"images:{images}")
        if images:
            # If multiple images found, combine them using concat_img
            combined_image = reduce(concat_img, images) if len(images) > 1 else images[0]
            section_images.append(combined_image)
        else:
            section_images.append(None)

    # 表格处理
    res = tokenize_table(tables, doc, is_english)
    print(f"表格处理结果:{res}")

    st = timer()
    if section_images:
        # if all images are None, set section_images to None
        if all(image is None for image in section_images):
            section_images = None

    if section_images:  # 图片合并
        chunks, images = naive_merge_with_images(sections, section_images,
                                                 int(128), "\n!?。；！？")
        # if kwargs.get("section_only", False):
        #     return chunks
        res.extend(tokenize_chunks_with_images(chunks, doc, is_english, images))
    else:  # 文本处理无图片
        chunks = naive_merge(
            sections, int(128), "\n!?。；！？")
        # if kwargs.get("section_only", False):
        #     return chunks
        res.extend(tokenize_chunks(chunks, doc, is_english))

    print("naive_merge({}): {}".format(filename, timer() - st))

    print(f"所有数据的结果:{res}")
    print("Finish parsing.")


if __name__ == '__main__':
    main_chunk()
