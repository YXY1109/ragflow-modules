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
import os
import sys
from io import BytesIO

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nlp import is_english
from parser.ppt_parser.ppt_parser import RAGFlowPptParser as PptParser


class Ppt(PptParser):
    def __call__(self, fnm, from_page, to_page, callback=None):
        txts = super().__call__(fnm, from_page, to_page)

        callback(0.5, "Text extraction finished.")

        # 尝试使用aspose.slides生成缩略图
        imgs = []
        try:
            import aspose.slides as slides
            import aspose.pydrawing as drawing

            with slides.Presentation(BytesIO(fnm)) as presentation:
                for i, slide in enumerate(presentation.slides[from_page: to_page]):
                    try:
                        # 获取幻灯片图像
                        slide_image = slide.get_image(0.2, 0.2)  # 使用全尺寸

                        # 创建临时文件
                        temp_path = f"temp_slide_{i}.png"

                        # 尝试不同的保存方法
                        try:
                            # 方法1：直接保存到文件
                            slide_image.save(temp_path)
                            img = Image.open(temp_path)
                            imgs.append(img.copy())
                            img.close()
                        except:
                            # 方法2：使用格式参数
                            slide_image.save(temp_path, drawing.imaging.ImageFormat.png)
                            img = Image.open(temp_path)
                            imgs.append(img.copy())
                            img.close()

                        # 清理临时文件
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                    except Exception as e:
                        print(f"警告：无法生成第{i + 1}页缩略图: {str(e)}")
                        # 使用占位图像
                        img = Image.new('RGB', (800, 600), color='lightgray')
                        imgs.append(img)

        except ImportError:
            print("警告：未安装aspose.slides，使用占位图像")
            # 如果没有安装aspose.slides，使用占位图像
            for _ in txts:
                img = Image.new('RGB', (800, 600), color='lightgray')
                imgs.append(img)
        except Exception as e:
            print(f"警告：图像生成失败: {str(e)}")
            # 使用占位图像
            for _ in txts:
                img = Image.new('RGB', (800, 600), color='lightgray')
                imgs.append(img)

        callback(0.9, "Image extraction finished")
        self.is_english = is_english(txts)
        return [(txts[i], imgs[i]) for i in range(len(txts))]
