import os

from parser.ppt_parser.presentation import Ppt

if __name__ == "__main__":
    # 没有继承关系，直接使用的

    # 创建PPT解析器实例
    ppt_parser = Ppt()

    # 获取当前目录下的所有PPTX文件
    pptx_dir = os.path.dirname(__file__)
    pptx_files = [f for f in os.listdir(pptx_dir) if f.endswith('.pptx')]

    if not pptx_files:
        print("错误: 当前目录下没有找到PPTX文件")
        print("请确保PPTX文件存在于当前目录下")
        exit(1)

    # 解析所有找到的PPTX文件
    for pptx_filename in pptx_files:
        pptx_file = os.path.join(pptx_dir, pptx_filename)

        print(f"\n{'=' * 60}")
        print(f"开始解析PPTX文件: {pptx_filename}")
        print(f"{'=' * 60}")

        try:
            # 读取PPTX文件内容
            with open(pptx_file, "rb") as f:
                pptx_content = f.read()


            # 定义回调函数用于显示进度
            def progress_callback(progress, message):
                print(f"进度: {progress * 100:.1f}% - {message}")


            # 解析PPTX文件（从第1页到最后一页）
            result = ppt_parser(pptx_content, from_page=0, to_page=20, callback=progress_callback)

            # 输出解析结果
            print(f"\n解析完成！共解析出 {len(result)} 页幻灯片")
            print("-" * 40)

            # 创建保存图片的目录
            output_dir = os.path.join(pptx_dir, "slide_images")
            os.makedirs(output_dir, exist_ok=True)

            print(f"图片将保存到: {output_dir}")

            for i, (text, image) in enumerate(result, 1):
                print(f"\n--- 第 {i} 页 ---")
                text_preview = text.strip()[:100] + "..." if len(text.strip()) > 100 else text.strip()
                print(f"文本内容:\n{text_preview if text else '(无文本内容)'}")
                print(f"图片尺寸: {image.size if image else '无图片'}")

                # 保存图片到本地
                if image:
                    # 生成图片文件名（基于PPT文件名）
                    pptx_basename = os.path.splitext(os.path.basename(pptx_file))[0]
                    image_filename = f"{pptx_basename}_slide_{i:02d}.png"
                    image_path = os.path.join(output_dir, image_filename)

                    try:
                        image.save(image_path, "PNG")
                        print(f"图片已保存: {image_path}")
                    except Exception as save_error:
                        print(f"保存图片失败: {str(save_error)}")
                else:
                    print("无图片数据，跳过保存")

            print(f"\n{pptx_filename} 处理完成！")

        except Exception as e:
            print(f"解析过程中出现错误: {str(e)}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("所有PPTX文件处理完成！")
    print(f"{'=' * 60}")
