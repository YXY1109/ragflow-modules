import pytest
import tempfile
import os
import sys
from pathlib import Path

# 添加项目根目录到sys.path，确保所有测试都能找到项目模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def temp_dir():
    """创建临时目录fixture"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="session")
def sample_text():
    """提供示例文本fixture"""
    return {
        "chinese": "这是一个中文测试文本，包含多种词汇和标点符号。",
        "english": "This is an English test text with various words and punctuation.",
        "mixed": "这是 mixed 中英文 text 测试文本。",
        "unicode": "🚀 Unicode test 测试 🌍",
        "empty": "",
        "whitespace": "   \n\t   ",
        "long": "这是一个很长的测试文本。" * 100
    }


@pytest.fixture(scope="session")
def sample_markdown_content():
    """提供示例Markdown内容fixture"""
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


@pytest.fixture(scope="function")
def temp_text_file(temp_dir, sample_text):
    """创建临时文本文件fixture"""
    def _create_temp_file(content_key="chinese", suffix=".txt"):
        content = sample_text[content_key]
        temp_file = os.path.join(temp_dir, f"temp_{content_key}{suffix}")

        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return temp_file

    return _create_temp_file


@pytest.fixture(scope="function")
def temp_markdown_file(temp_dir, sample_markdown_content):
    """创建临时Markdown文件fixture"""
    temp_file = os.path.join(temp_dir, "temp_sample.md")

    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_markdown_content)

    return temp_file


def pytest_configure(config):
    """pytest配置钩子"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "parser: mark test as parser-related test"
    )
    config.addinivalue_line(
        "markers", "nlp: mark test as NLP-related test"
    )
    config.addinivalue_line(
        "markers", "registry: mark test as registry-related test"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试项的钩子"""
    # 为没有标记的测试自动添加unit标记
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """自动清理临时文件的fixture"""
    # 测试前执行
    yield
    # 测试后执行（如果需要清理逻辑）
    pass