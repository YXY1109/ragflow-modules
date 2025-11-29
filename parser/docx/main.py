# -*- coding: utf-8 -*-
"""
DOCX文档解析器模块
提供多种DOCX文档解析功能，包括基础文本提取、表格提取、图片提取、样式保持和结构化解析
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.shared import RGBColor
except ImportError:
    print("请安装python-docx库: pip install python-docx")
    exit(1)

try:
    from PIL import Image
    import zipfile
except ImportError:
    print("请安装Pillow库: pip install Pillow")
    exit(1)


@dataclass
class DocumentElement:
    """文档元素数据结构"""
    element_type: str  # text, table, image, heading, list
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['DocumentElement'] = field(default_factory=list)
    level: int = 0  # 用于标题层级、列表层级等


class BaseDocxParser(ABC):
    """DOCX解析器基类"""

    def __init__(self):
        self.supported_extensions = ['.docx']

    @abstractmethod
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        解析DOCX文件

        Args:
            file_path: DOCX文件路径

        Returns:
            解析结果字典
        """
        pass

    def validate_file(self, file_path: str) -> bool:
        """验证文件是否支持"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions


class BasicDocxParser(BaseDocxParser):
    """基础DOCX解析器 - 提取纯文本内容"""

    def parse(self, file_path: str) -> Dict[str, Any]:
        """解析DOCX文件，提取基础文本内容"""
        if not self.validate_file(file_path):
            raise ValueError(f"不支持的文件格式: {file_path}")

        try:
            doc = Document(file_path)

            # 提取段落文本
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append({
                        'text': paragraph.text,
                        'style': paragraph.style.name if paragraph.style else 'Normal'
                    })

            # 提取表格文本
            tables = []
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = []
                    for cell in row.cells:
                        row_data.append(cell.text.strip())
                    table_data.append(row_data)
                tables.append(table_data)

            return {
                'parser_type': 'BasicDocxParser',
                'file_path': file_path,
                'paragraphs': paragraphs,
                'tables': tables,
                'total_paragraphs': len(paragraphs),
                'total_tables': len(tables),
                'raw_text': '\n'.join([p['text'] for p in paragraphs])
            }

        except Exception as e:
            raise RuntimeError(f"解析DOCX文件失败: {str(e)}")


class TableExtractorParser(BaseDocxParser):
    """表格提取解析器 - 专门提取和处理表格数据"""

    def parse(self, file_path: str) -> Dict[str, Any]:
        """解析DOCX文件，重点提取表格结构"""
        if not self.validate_file(file_path):
            raise ValueError(f"不支持的文件格式: {file_path}")

        try:
            doc = Document(file_path)

            # 提取表格详细信息
            tables = []
            for table_idx, table in enumerate(doc.tables):
                table_info = {
                    'table_index': table_idx,
                    'rows': len(table.rows),
                    'columns': len(table.columns) if table.rows else 0,
                    'cells': [],
                    'structure': []
                }

                # 提取单元格数据
                for row_idx, row in enumerate(table.rows):
                    row_cells = []
                    for col_idx, cell in enumerate(row.cells):
                        cell_info = {
                            'row': row_idx,
                            'column': col_idx,
                            'text': cell.text.strip(),
                            'paragraphs_count': len(cell.paragraphs)
                        }

                        # 提取单元格内的段落格式
                        cell_paragraphs = []
                        for para in cell.paragraphs:
                            if para.text.strip():
                                cell_paragraphs.append({
                                    'text': para.text,
                                    'style': para.style.name if para.style else 'Normal'
                                })
                        cell_info['paragraphs'] = cell_paragraphs
                        row_cells.append(cell_info)

                    table_info['cells'].append(row_cells)
                    table_info['structure'].append([cell['text'] for cell in row_cells])

                tables.append(table_info)

            # 提取普通文本
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text.strip())

            return {
                'parser_type': 'TableExtractorParser',
                'file_path': file_path,
                'tables': tables,
                'paragraphs': paragraphs,
                'total_tables': len(tables),
                'total_paragraphs': len(paragraphs),
                'summary': {
                    'total_cells': sum(len(table['cells']) * (len(table['cells'][0]) if table['cells'] else 0) for table in tables),
                    'max_rows': max((table['rows'] for table in tables), default=0),
                    'max_cols': max((table['columns'] for table in tables), default=0)
                }
            }

        except Exception as e:
            raise RuntimeError(f"表格提取失败: {str(e)}")


class ImageExtractorParser(BaseDocxParser):
    """图片提取解析器 - 提取文档中的图片信息"""

    def parse(self, file_path: str) -> Dict[str, Any]:
        """解析DOCX文件，提取图片信息"""
        if not self.validate_file(file_path):
            raise ValueError(f"不支持的文件格式: {file_path}")

        try:
            doc = Document(file_path)

            # 提取文本内容
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text.strip())

            # 提取图片信息
            images = self._extract_images(file_path)

            # 统计信息
            return {
                'parser_type': 'ImageExtractorParser',
                'file_path': file_path,
                'paragraphs': paragraphs,
                'images': images,
                'total_paragraphs': len(paragraphs),
                'total_images': len(images),
                'summary': {
                    'has_images': len(images) > 0,
                    'image_formats': list(set(img['format'] for img in images)),
                    'total_size_mb': round(sum(img.get('size_bytes', 0) for img in images) / (1024 * 1024), 2)
                }
            }

        except Exception as e:
            raise RuntimeError(f"图片提取失败: {str(e)}")

    def _extract_images(self, docx_path: str) -> List[Dict[str, Any]]:
        """从DOCX文件中提取图片信息"""
        images = []

        try:
            with zipfile.ZipFile(docx_path, 'r') as zip_file:
                # 获取所有图片文件
                image_files = [f for f in zip_file.namelist()
                             if f.startswith('word/media/') and
                             f.split('/')[-1].lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.wmf'))]

                for img_idx, img_file in enumerate(image_files):
                    try:
                        # 获取图片文件信息
                        img_data = zip_file.read(img_file)
                        img_name = img_file.split('/')[-1]

                        # 尝试获取图片尺寸
                        try:
                            import io
                            img = Image.open(io.BytesIO(img_data))
                            width, height = img.size
                            format_name = img.format or img_name.split('.')[-1].upper()
                        except:
                            width, height, format_name = None, None, img_name.split('.')[-1].upper()

                        images.append({
                            'image_index': img_idx,
                            'filename': img_name,
                            'format': format_name,
                            'size_bytes': len(img_data),
                            'width': width,
                            'height': height,
                            'path_in_docx': img_file,
                            'size_mb': round(len(img_data) / (1024 * 1024), 2)
                        })

                    except Exception as e:
                        print(f"警告: 无法处理图片 {img_file}: {str(e)}")
                        continue

        except Exception as e:
            print(f"警告: 提取图片信息时出错: {str(e)}")

        return images


class StylePreservingParser(BaseDocxParser):
    """样式保持解析器 - 提取文本并保持格式信息"""

    def parse(self, file_path: str) -> Dict[str, Any]:
        """解析DOCX文件，保持文本样式信息"""
        if not self.validate_file(file_path):
            raise ValueError(f"不支持的文件格式: {file_path}")

        try:
            doc = Document(file_path)

            # 提取带样式的段落
            styled_paragraphs = []
            for para_idx, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    para_info = {
                        'paragraph_index': para_idx,
                        'text': paragraph.text,
                        'style': {
                            'name': paragraph.style.name if paragraph.style else 'Normal',
                            'alignment': self._get_alignment_name(paragraph.alignment) if paragraph.alignment else 'left'
                        }
                    }

                    # 提取运行(run)信息
                    runs = []
                    for run in paragraph.runs:
                        if run.text.strip():
                            run_info = {
                                'text': run.text,
                                'bold': run.bold,
                                'italic': run.italic,
                                'underline': run.underline,
                                'font_name': run.font.name if run.font.name else None,
                                'font_size': run.font.size.pt if run.font.size else None,
                                'color': self._get_color_rgb(run.font.color) if run.font.color else None
                            }
                            runs.append(run_info)

                    para_info['runs'] = runs
                    styled_paragraphs.append(para_info)

            # 提取表格样式信息
            styled_tables = []
            for table_idx, table in enumerate(doc.tables):
                table_info = {
                    'table_index': table_idx,
                    'rows': len(table.rows),
                    'columns': len(table.columns) if table.rows else 0,
                    'cells': []
                }

                for row_idx, row in enumerate(table.rows):
                    for col_idx, cell in enumerate(row.cells):
                        cell_paragraphs = []
                        for para in cell.paragraphs:
                            if para.text.strip():
                                cell_paragraphs.append({
                                    'text': para.text,
                                    'style': para.style.name if para.style else 'Normal'
                                })

                        table_info['cells'].append({
                            'row': row_idx,
                            'column': col_idx,
                            'text': cell.text.strip(),
                            'paragraphs': cell_paragraphs
                        })

                styled_tables.append(table_info)

            return {
                'parser_type': 'StylePreservingParser',
                'file_path': file_path,
                'paragraphs': styled_paragraphs,
                'tables': styled_tables,
                'total_paragraphs': len(styled_paragraphs),
                'total_tables': len(styled_tables),
                'style_summary': self._analyze_styles(styled_paragraphs)
            }

        except Exception as e:
            raise RuntimeError(f"样式保持解析失败: {str(e)}")

    def _get_alignment_name(self, alignment) -> str:
        """获取对齐方式名称"""
        alignment_map = {
            0: 'left',
            1: 'center',
            2: 'right',
            3: 'justify'
        }
        return alignment_map.get(alignment, 'left')

    def _get_color_rgb(self, color) -> str:
        """获取颜色RGB值"""
        try:
            if color.rgb:
                return f"#{color.rgb}"
            return None
        except:
            return None

    def _analyze_styles(self, paragraphs: List[Dict]) -> Dict[str, Any]:
        """分析文档样式统计信息"""
        styles_used = {}
        formatting_stats = {
            'bold_count': 0,
            'italic_count': 0,
            'underline_count': 0
        }

        for para in paragraphs:
            style_name = para['style']['name']
            styles_used[style_name] = styles_used.get(style_name, 0) + 1

            for run in para.get('runs', []):
                if run.get('bold'):
                    formatting_stats['bold_count'] += 1
                if run.get('italic'):
                    formatting_stats['italic_count'] += 1
                if run.get('underline'):
                    formatting_stats['underline_count'] += 1

        return {
            'styles_used': styles_used,
            'formatting_stats': formatting_stats,
            'total_runs': sum(len(para.get('runs', [])) for para in paragraphs)
        }


class StructuredDocxParser(BaseDocxParser):
    """结构化解析器 - 将文档内容解析为结构化元素"""

    def parse(self, file_path: str) -> Dict[str, Any]:
        """解析DOCX文件，生成结构化文档树"""
        if not self.validate_file(file_path):
            raise ValueError(f"不支持的文件格式: {file_path}")

        try:
            doc = Document(file_path)

            # 构建文档元素树
            document_elements = []

            # 处理段落
            for element in doc.element.body:
                if element.tag.endswith('p'):  # 段落
                    para_element = self._parse_paragraph(element)
                    if para_element:
                        document_elements.append(para_element)
                elif element.tag.endswith('tbl'):  # 表格
                    table_element = self._parse_table(element)
                    if table_element:
                        document_elements.append(table_element)

            # 分析文档结构
            structure_info = self._analyze_structure(document_elements)

            return {
                'parser_type': 'StructuredDocxParser',
                'file_path': file_path,
                'document_elements': [self._element_to_dict(elem) for elem in document_elements],
                'structure_info': structure_info,
                'total_elements': len(document_elements),
                'content_types': list(set(elem.element_type for elem in document_elements))
            }

        except Exception as e:
            raise RuntimeError(f"结构化解析失败: {str(e)}")

    def _parse_paragraph(self, paragraph_element) -> Optional[DocumentElement]:
        """解析段落元素"""
        try:
            from docx.text.paragraph import Paragraph
            para = Paragraph(paragraph_element, None)

            if not para.text.strip():
                return None

            # 判断元素类型
            element_type = 'text'
            level = 0

            # 检查是否为标题
            style_name = para.style.name if para.style else 'Normal'
            if 'Heading' in style_name:
                element_type = 'heading'
                try:
                    level = int(style_name.split()[-1])
                except:
                    level = 1
            # 检查是否为列表项
            elif para.text.strip().startswith(('•', '-', '*', '1.', '2.', '3.')):
                element_type = 'list'
                level = self._detect_list_level(para.text)

            return DocumentElement(
                element_type=element_type,
                content=para.text.strip(),
                metadata={
                    'style': style_name,
                    'alignment': self._get_alignment_name(para.alignment) if para.alignment else 'left'
                },
                level=level
            )

        except Exception as e:
            print(f"警告: 解析段落失败: {str(e)}")
            return None

    def _parse_table(self, table_element) -> Optional[DocumentElement]:
        """解析表格元素"""
        try:
            from docx.table import Table
            table = Table(table_element, None)

            table_data = []
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    cell_text = cell.text.strip().replace('\n', ' ')
                    row_data.append(cell_text)
                if any(row_data):  # 只添加非空行
                    table_data.append(row_data)

            if not table_data:
                return None

            return DocumentElement(
                element_type='table',
                content=json.dumps(table_data, ensure_ascii=False),
                metadata={
                    'rows': len(table.rows),
                    'columns': len(table.columns) if table.rows else 0
                },
                children=[]
            )

        except Exception as e:
            print(f"警告: 解析表格失败: {str(e)}")
            return None

    def _detect_list_level(self, text: str) -> int:
        """检测列表级别"""
        stripped = text.lstrip()
        if stripped.startswith('        '):
            return 3
        elif stripped.startswith('    '):
            return 2
        elif stripped.startswith('  '):
            return 1
        return 0

    def _analyze_structure(self, elements: List[DocumentElement]) -> Dict[str, Any]:
        """分析文档结构"""
        structure_stats = {
            'headings': [],
            'lists': [],
            'tables': 0,
            'regular_paragraphs': 0
        }

        for elem in elements:
            if elem.element_type == 'heading':
                structure_stats['headings'].append({
                    'level': elem.level,
                    'content': elem.content[:50] + '...' if len(elem.content) > 50 else elem.content
                })
            elif elem.element_type == 'list':
                structure_stats['lists'].append({
                    'level': elem.level,
                    'content': elem.content[:50] + '...' if len(elem.content) > 50 else elem.content
                })
            elif elem.element_type == 'table':
                structure_stats['tables'] += 1
            else:
                structure_stats['regular_paragraphs'] += 1

        return structure_stats

    def _element_to_dict(self, element: DocumentElement) -> Dict[str, Any]:
        """将DocumentElement转换为字典"""
        return {
            'element_type': element.element_type,
            'content': element.content,
            'metadata': element.metadata,
            'level': element.level,
            'children': [self._element_to_dict(child) for child in element.children]
        }


class DocxParserFactory:
    """DOCX解析器工厂类"""

    _parsers = {
        'basic': BasicDocxParser,
        'table': TableExtractorParser,
        'image': ImageExtractorParser,
        'style': StylePreservingParser,
        'structured': StructuredDocxParser
    }

    @classmethod
    def create_parser(cls, parser_type: str) -> BaseDocxParser:
        """创建解析器实例"""
        if parser_type not in cls._parsers:
            raise ValueError(f"不支持的解析器类型: {parser_type}. 支持的类型: {list(cls._parsers.keys())}")

        return cls._parsers[parser_type]()

    @classmethod
    def get_available_parsers(cls) -> List[str]:
        """获取可用的解析器列表"""
        return list(cls._parsers.keys())


def test_parsers():
    """测试所有解析器"""
    print("DOCX解析器测试程序")
    print("=" * 50)

    # 获取可用解析器
    available_parsers = DocxParserFactory.get_available_parsers()
    print(f"可用解析器: {', '.join(available_parsers)}")
    print()

    # 测试文件路径 (这里使用一个示例路径，实际使用时需要提供真实的docx文件)
    test_file = "test_document.docx"

    if not os.path.exists(test_file):
        print(f"测试文件 {test_file} 不存在")
        print("请将测试文件放在当前目录下，或修改test_file变量")
        return

    # 测试每种解析器
    for parser_type in available_parsers:
        print(f"\n测试 {parser_type} 解析器:")
        print("-" * 30)

        try:
            parser = DocxParserFactory.create_parser(parser_type)
            result = parser.parse(test_file)

            print(f"解析器类型: {result.get('parser_type')}")
            print(f"文件路径: {result.get('file_path')}")

            # 显示特定解析器的关键信息
            if parser_type == 'basic':
                print(f"段落数: {result.get('total_paragraphs')}")
                print(f"表格数: {result.get('total_tables')}")
                print(f"文本预览: {result.get('raw_text', '')[:100]}...")

            elif parser_type == 'table':
                print(f"表格数: {result.get('total_tables')}")
                summary = result.get('summary', {})
                print(f"总单元格数: {summary.get('total_cells', 0)}")
                print(f"最大行数: {summary.get('max_rows', 0)}")
                print(f"最大列数: {summary.get('max_cols', 0)}")

            elif parser_type == 'image':
                print(f"图片数: {result.get('total_images')}")
                summary = result.get('summary', {})
                print(f"包含图片: {summary.get('has_images', False)}")
                print(f"图片格式: {', '.join(summary.get('image_formats', []))}")
                print(f"总大小: {summary.get('total_size_mb', 0)} MB")

            elif parser_type == 'style':
                print(f"段落数: {result.get('total_paragraphs')}")
                style_summary = result.get('style_summary', {})
                print(f"使用的样式: {list(style_summary.get('styles_used', {}).keys())}")
                print(f"格式化统计: {style_summary.get('formatting_stats', {})}")

            elif parser_type == 'structured':
                print(f"元素总数: {result.get('total_elements')}")
                print(f"内容类型: {', '.join(result.get('content_types', []))}")
                structure = result.get('structure_info', {})
                print(f"标题数: {len(structure.get('headings', []))}")
                print(f"列表数: {len(structure.get('lists', []))}")
                print(f"表格数: {structure.get('tables', 0)}")

            print(f"✓ {parser_type} 解析器测试成功")

        except Exception as e:
            print(f"✗ {parser_type} 解析器测试失败: {str(e)}")


def save_results_to_file(results: Dict[str, Any], output_file: str):
    """将解析结果保存到JSON文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"解析结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")


def parse_docx_file(file_path: str, parser_type: str = 'basic', output_file: str = None) -> Dict[str, Any]:
    """
    解析DOCX文件的便捷函数

    Args:
        file_path: DOCX文件路径
        parser_type: 解析器类型 ('basic', 'table', 'image', 'style', 'structured')
        output_file: 输出文件路径 (可选)

    Returns:
        解析结果字典
    """
    parser = DocxParserFactory.create_parser(parser_type)
    result = parser.parse(file_path)

    if output_file:
        save_results_to_file(result, output_file)

    return result


def main():
    """主函数"""
    print("DOCX文档解析器")
    print("=" * 50)

    # 交互式解析
    file_path = input("请输入DOCX文件路径 (或直接回车运行测试): ").strip()

    if not file_path:
        # 运行测试
        test_parsers()
        return

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 选择解析器
    available_parsers = DocxParserFactory.get_available_parsers()
    print(f"\n可用解析器: {', '.join(available_parsers)}")

    parser_type = input("请选择解析器类型 (默认: basic): ").strip() or 'basic'

    if parser_type not in available_parsers:
        print(f"不支持的解析器类型: {parser_type}")
        return

    # 输出文件
    output_file = input("请输入输出JSON文件路径 (可选，直接回车跳过): ").strip()
    if not output_file:
        output_file = None

    print(f"\n正在使用 {parser_type} 解析器解析文件...")

    try:
        result = parse_docx_file(file_path, parser_type, output_file)

        print("解析完成!")
        print(f"解析器类型: {result.get('parser_type')}")
        print(f"文件路径: {result.get('file_path')}")

        # 显示简要结果
        if parser_type == 'basic':
            print(f"段落数: {result.get('total_paragraphs')}")
            print(f"表格数: {result.get('total_tables')}")
        elif parser_type == 'image':
            print(f"图片数: {result.get('total_images')}")

        if output_file:
            print(f"结果已保存到: {output_file}")

    except Exception as e:
        print(f"解析失败: {str(e)}")


if __name__ == "__main__":
    main()