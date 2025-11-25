import pytest
import tempfile
import os
from pathlib import Path
import sys

# 添加项目根目录到sys.path
sys.path.append(str(Path(__file__).parent.parent))

from parser.markdown_parser.markdown_parser import RAGFlowMarkdownParser, MarkdownElementExtractor


class TestRAGFlowMarkdownParser:
    """测试RAGFlow Markdown解析器"""

    @pytest.fixture
    def markdown_parser(self):
        """创建RAGFlowMarkdownParser实例"""
        return RAGFlowMarkdownParser(chunk_token_num=128)

    @pytest.fixture
    def sample_markdown_content(self):
        """提供示例Markdown内容"""
        return """# 测试标题

这是一个测试的Markdown文档。

## 二级标题

### 三级标题

这是一个包含**粗体**和*斜体*文本的段落。

- 列表项1
- 列表项2
  - 嵌套列表项1
  - 嵌套列表项2

1. 有序列表项1
2. 有序列表项2

```python
def hello_world():
    print("Hello, World!")
```

[链接文本](https://example.com)

![图片描述](image.jpg)

| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 数据1 | 数据2 | 数据3 |
| 数据4 | 数据5 | 数据6 |
"""

    @pytest.fixture
    def simple_markdown_content(self):
        """提供简单的Markdown内容"""
        return """# 简单标题

这是一个简单的段落。"""

    def test_parser_initialization(self):
        """测试解析器初始化"""
        parser1 = RAGFlowMarkdownParser()
        assert parser1.chunk_token_num == 128

        parser2 = RAGFlowMarkdownParser(chunk_token_num=64)
        assert parser2.chunk_token_num == 64

    def test_extract_tables_and_remainder_with_no_tables(self, markdown_parser, simple_markdown_content):
        """测试提取表格（无表格情况）"""
        result_text, tables = markdown_parser.extract_tables_and_remainder(
            simple_markdown_content, separate_tables=True
        )

        assert isinstance(result_text, str)
        assert isinstance(tables, list)
        assert len(tables) == 0
        assert len(result_text) > 0
        assert "简单标题" in result_text

    def test_extract_tables_and_remainder_with_tables(self, markdown_parser, sample_markdown_content):
        """测试提取表格（有表格情况）"""
        result_text, tables = markdown_parser.extract_tables_and_remainder(
            sample_markdown_content, separate_tables=True
        )

        assert isinstance(result_text, str)
        assert isinstance(tables, list)

        # 检查是否检测到了表格
        if len(tables) > 0:
            assert isinstance(tables[0], str)
            assert "|" in tables[0]  # 表格应该包含竖线

        # 结果文本应该包含其他内容
        assert len(result_text) > 0

    def test_extract_tables_with_html_tables(self, markdown_parser):
        """测试提取HTML表格"""
        html_table_content = """
# 标题

这是一些文本内容。

<table>
    <tr>
        <th>表头1</th>
        <th>表头2</th>
    </tr>
    <tr>
        <td>数据1</td>
        <td>数据2</td>
    </tr>
</table>

更多内容。
"""

        result_text, tables = markdown_parser.extract_tables_and_remainder(
            html_table_content, separate_tables=True
        )

        assert isinstance(result_text, str)
        assert isinstance(tables, list)

    def test_extract_tables_without_separation(self, markdown_parser, sample_markdown_content):
        """测试提取表格但不分离"""
        result_text, tables = markdown_parser.extract_tables_and_remainder(
            sample_markdown_content, separate_tables=False
        )

        assert isinstance(result_text, str)
        assert isinstance(tables, list)
        assert len(result_text) > 0


class TestMarkdownElementExtractor:
    """测试Markdown元素提取器"""

    @pytest.fixture
    def extractor(self):
        """创建MarkdownElementExtractor实例"""
        content = """# 主标题

这是一些介绍文本。

## 子标题1

这是第一个部分的内容。

- 列表项1
- 列表项2

## 子标题2

这是第二个部分的内容。

```python
def example():
    return "test"
```

> 这是一个引用块
> 包含多行内容

最后一段内容。
"""
        return MarkdownElementExtractor(content)

    def test_extractor_initialization(self, extractor):
        """测试提取器初始化"""
        assert extractor.markdown_content is not None
        assert len(extractor.markdown_content) > 0
        assert isinstance(extractor.lines, list)
        assert len(extractor.lines) > 0

    def test_extract_elements_without_delimiter(self, extractor):
        """测试提取元素（无分隔符）"""
        elements = extractor.extract_elements()

        assert isinstance(elements, list)
        assert len(elements) > 0

        # 检查是否包含各种类型的元素
        combined_text = " ".join(elements)
        assert "主标题" in combined_text or "子标题" in combined_text

    def test_extract_elements_with_delimiter(self, extractor):
        """测试使用分隔符提取元素"""
        # 使用句号作为分隔符
        elements = extractor.extract_elements(delimiter="`。`")

        assert isinstance(elements, list)
        # 结果取决于具体内容和分隔符

    def test_extract_elements_with_meta(self, extractor):
        """测试提取元素并包含元数据"""
        elements = extractor.extract_elements(include_meta=True)

        assert isinstance(elements, list)
        assert len(elements) > 0

        # 检查元数据结构
        for element in elements:
            assert isinstance(element, dict)
            assert "content" in element
            assert "start_line" in element
            assert "end_line" in element
            assert isinstance(element["content"], str)
            assert isinstance(element["start_line"], int)
            assert isinstance(element["end_line"], int)

    def test_get_delimiters(self):
        """测试获取分隔符"""
        extractor = MarkdownElementExtractor("测试内容")
        delimiters = extractor.get_delimiters("句号。逗号，感叹号！")

        assert isinstance(delimiters, str)
        # 结果取决于具体的分隔符处理逻辑

    def test_extract_elements_from_simple_content(self):
        """测试从简单内容提取元素"""
        simple_content = """# 标题

段落内容。

## 子标题

更多内容。
"""
        extractor = MarkdownElementExtractor(simple_content)
        elements = extractor.extract_elements()

        assert isinstance(elements, list)
        assert len(elements) > 0

    def test_extract_elements_from_code_heavy_content(self):
        """测试从包含大量代码的内容提取元素"""
        code_content = """# 代码示例

```python
def hello():
    print("Hello, World!")
```

```javascript
function test() {
    return "JavaScript";
}
```

一些解释文本。
"""
        extractor = MarkdownElementExtractor(code_content)
        elements = extractor.extract_elements()

        assert isinstance(elements, list)
        assert len(elements) > 0

        combined_text = " ".join(elements)
        assert "代码示例" in combined_text or "def hello" in combined_text

    def test_extract_elements_from_list_heavy_content(self):
        """测试从包含大量列表的内容提取元素"""
        list_content = """# 列表示例

- 第一项
- 第二项
  - 子项目1
  - 子项目2
- 第三项

1. 有序第一项
2. 有序第二项
3. 有序第三项

内容继续。
"""
        extractor = MarkdownElementExtractor(list_content)
        elements = extractor.extract_elements()

        assert isinstance(elements, list)
        assert len(elements) > 0

    def test_empty_content_handling(self):
        """测试空内容处理"""
        extractor = MarkdownElementExtractor("")
        elements = extractor.extract_elements()

        assert isinstance(elements, list)
        # 空内容的结果取决于具体实现

    def test_whitespace_only_content(self):
        """测试只包含空白字符的内容"""
        whitespace_content = "   \n\n  \t  \n\n   "
        extractor = MarkdownElementExtractor(whitespace_content)
        elements = extractor.extract_elements()

        assert isinstance(elements, list)