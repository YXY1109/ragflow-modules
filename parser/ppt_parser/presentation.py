#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from io import BytesIO

from PIL import Image

from nlp import is_english
from parser.ppt_parser.ppt_parser import RAGFlowPptParser as PptParser


class Ppt(PptParser):
    def __call__(self, fnm, from_page, to_page, callback=None):
        txts = super().__call__(fnm, from_page, to_page)

        callback(0.5, "Text extraction finished.")
        import aspose.slides as slides
        import aspose.pydrawing as drawing
        imgs = []
        with slides.Presentation(BytesIO(fnm)) as presentation:
            for i, slide in enumerate(presentation.slides[from_page: to_page]):
                try:
                    with BytesIO() as buffered:
                        slide.get_thumbnail(
                            0.1, 0.1).save(
                            buffered, drawing.imaging.ImageFormat.jpeg)
                        buffered.seek(0)
                        imgs.append(Image.open(buffered).copy())
                except RuntimeError as e:
                    raise RuntimeError(f'ppt parse error at page {i + 1}, original error: {str(e)}') from e
        assert len(imgs) == len(
            txts), "Slides text and image do not match: {} vs. {}".format(len(imgs), len(txts))
        callback(0.9, "Image extraction finished")
        self.is_english = is_english(txts)
        return [(txts[i], imgs[i]) for i in range(len(txts))]


if __name__ == "__main__":
    import os

    # 创建PPT解析器实例
    ppt_parser = Ppt()

    # 指定要解析的PPTX文件路径
    pptx_file = "20251755499694183.pptx"

    if os.path.exists(pptx_file):
        print(f"开始解析PPTX文件: {pptx_file}")

        try:
            # 读取PPTX文件内容
            with open(pptx_file, "rb") as f:
                pptx_content = f.read()

            # 定义回调函数用于显示进度
            def progress_callback(progress, message):
                print(f"进度: {progress*100:.1f}% - {message}")

            # 解析PPTX文件（从第1页到最后一页）
            result = ppt_parser(pptx_content, from_page=0, to_page=20, callback=progress_callback)

            # 输出解析结果
            print(f"\n解析完成！共解析出 {len(result)} 页幻灯片")
            print("=" * 50)

            for i, (text, image) in enumerate(result, 1):
                print(f"\n--- 第 {i} 页 ---")
                print(f"文本内容:\n{text.strip() if text else '(无文本内容)'}")
                print(f"图片尺寸: {image.size if image else '无图片'}")

        except Exception as e:
            print(f"解析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"错误: 找不到文件 {pptx_file}")
        print("请确保文件存在于当前目录下")
