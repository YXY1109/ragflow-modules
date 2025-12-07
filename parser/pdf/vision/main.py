


def demo_file():
    pdf_parser = VisionParser(vision_model=vision_model)
    filename = r""
    sections, tables = pdf_parser(filename, from_page=0, to_page=100)

if __name__ == '__main__':
    demo_file()
