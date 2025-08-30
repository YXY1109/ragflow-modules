from parser.mark_down.mark_down import Markdown


def main():
    filename=r""
    markdown_parser = Markdown(128)
    sections, tables = markdown_parser(filename, None, separate_tables=False)

    # Process images for each section
    section_images = []
    for section_text, _ in sections:
        images = markdown_parser.get_pictures(section_text) if section_text else None
        if images:
            # If multiple images found, combine them using concat_img
            combined_image = reduce(concat_img, images) if len(images) > 1 else images[0]
            section_images.append(combined_image)
        else:
            section_images.append(None)

    res = tokenize_table(tables, doc, is_english)


if __name__ == '__main__':
    main()