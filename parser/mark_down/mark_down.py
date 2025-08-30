from io import BytesIO
from parser.mark_down import find_codec
from parser.mark_down.markdown_parser import (MarkdownElementExtractor,
                                              RAGFlowMarkdownParser)

from loguru import logger
from markdown import markdown
from PIL import Image


class Markdown(RAGFlowMarkdownParser):
    def get_picture_urls(self, sections):
        if not sections:
            return []
        if isinstance(sections, type("")):
            text = sections
        elif isinstance(sections[0], type("")):
            text = sections[0]
        else:
            return []

        from bs4 import BeautifulSoup
        html_content = markdown(text)
        soup = BeautifulSoup(html_content, 'html.parser')
        html_images = [img.get('src') for img in soup.find_all('img') if img.get('src')]
        return html_images

    def get_pictures(self, text):
        """Download and open all images from markdown text."""
        import requests
        image_urls = self.get_picture_urls(text)
        images = []
        # Find all image URLs in text
        for url in image_urls:
            try:
                # check if the url is a local file or a remote URL
                if url.startswith(('http://', 'https://')):
                    # For remote URLs, download the image
                    response = requests.get(url, stream=True, timeout=30)
                    if response.status_code == 200 and response.headers['Content-Type'].startswith('image/'):
                        img = Image.open(BytesIO(response.content)).convert('RGB')
                        images.append(img)
                else:
                    # For local file paths, open the image directly
                    from pathlib import Path
                    local_path = Path(url)
                    if not local_path.exists():
                        logger.warning(f"Local image file not found: {url}")
                        continue
                    img = Image.open(url).convert('RGB')
                    images.append(img)
            except Exception as e:
                logger.error(f"Failed to download/open image from {url}: {e}")
                continue

        return images if images else None

    def __call__(self, filename, binary=None, separate_tables=True):
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(filename, "r") as f:
                txt = f.read()

        remainder, tables = self.extract_tables_and_remainder(f'{txt}\n', separate_tables=separate_tables)

        extractor = MarkdownElementExtractor(txt)
        element_sections = extractor.extract_elements()
        sections = [(element, "") for element in element_sections]

        tbls = []
        for table in tables:
            tbls.append(((None, markdown(table, extensions=['markdown.extensions.tables'])), ""))
        return sections, tbls
