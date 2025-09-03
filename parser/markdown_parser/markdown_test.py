from parser.markdown_parser.markdown_parser import MarkdownBase


def main():
    filename = r"D:\PycharmProjects\ragflow-modules\files\markdown\demo1.md"
    filename = r"/Users/cj/PycharmProjects/ragflow-modules/files/myself/demo1.md"
    markdown_parser = MarkdownBase(128)
    sections, tables = markdown_parser(filename, None, separate_tables=False)

    # Process images for each section
    section_images = []
    for section_text, _ in sections:
        images = markdown_parser.get_pictures(section_text) if section_text else None
        print(f"images:{images}")
        if images:
            # If multiple images found, combine them using concat_img
            pass
            # combined_image = reduce(concat_img, images) if len(images) > 1 else images[0]
            # section_images.append(combined_image)
            section_images.append(images)
        else:
            section_images.append(None)

    # res = tokenize_table(tables, doc, is_english)
    print(f"section_images:{section_images}")
    print("123")


if __name__ == '__main__':
    main()
