import logging
from io import BytesIO

from PIL import Image
from markdown import markdown

from markdown_parser import RAGFlowMarkdownParser as MarkdownParser, MarkdownElementExtractor
from parser.markdown_parser import find_codec


class Markdown(MarkdownParser):
    def md_to_html(self, sections):
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
        return soup

    def get_picture_urls(self, soup):
        if soup:
            return [img.get('src') for img in soup.find_all('img') if img.get('src')]
        return []

    def get_hyperlink_urls(self, soup):
        if soup:
            return set([a.get('href') for a in soup.find_all('a') if a.get('href')])
        return []

    def get_pictures(self, text):
        """Download and open all images from markdown text."""
        import requests
        soup = self.md_to_html(text)
        image_urls = self.get_picture_urls(soup)
        images = []
        # Find all image URLs in text
        for url in image_urls:
            if not url:
                continue
            try:
                # check if the url is a local file or a remote URL
                if url.startswith(('http://', 'https://')):
                    # For remote URLs, download the image
                    response = requests.get(url, stream=True, timeout=30)
                    if response.status_code == 200 and response.headers['Content-Type'] and response.headers[
                        'Content-Type'].startswith('image/'):
                        img = Image.open(BytesIO(response.content)).convert('RGB')
                        images.append(img)
                else:
                    # For local file paths, open the image directly
                    from pathlib import Path
                    local_path = Path(url)
                    if not local_path.exists():
                        logging.warning(f"Local image file not found: {url}")
                        continue
                    img = Image.open(url).convert('RGB')
                    images.append(img)
            except Exception as e:
                logging.error(f"Failed to download/open image from {url}: {e}")
                continue

        return images if images else None

    def __call__(self, filename, binary=None, separate_tables=True, delimiter=None):
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(filename, "r",encoding="utf-8") as f:
                txt = f.read()

        remainder, tables = self.extract_tables_and_remainder(f'{txt}\n', separate_tables=separate_tables)
        # To eliminate duplicate tables in chunking result, uncomment code below and set separate_tables to True in line 410.
        # extractor = MarkdownElementExtractor(remainder)
        extractor = MarkdownElementExtractor(txt)
        element_sections = extractor.extract_elements(delimiter)
        sections = [(element, "") for element in element_sections]
        tbls = []
        for table in tables:
            tbls.append(((None, markdown(table, extensions=['markdown.extensions.tables'])), ""))
        return sections, tbls