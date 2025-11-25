import pytest
import tempfile
import os
from pathlib import Path
import sys

# 添加项目根目录到sys.path
sys.path.append(str(Path(__file__).parent.parent))

from parser.txt_parser.txt_parser import RAGFlowTxtParser


class TestRAGFlowTxtParser:
    """测试RAGFlow文本解析器"""

    @pytest.fixture
    def txt_parser(self):
        """创建RAGFlowTxtParser实例"""
        return RAGFlowTxtParser()

    @pytest.fixture
    def sample_content(self):
        """提供示例内容"""
        return "这是一个测试文档。\n包含多行文本。\n还有一些特殊字符：!@#$%^&*()\nEmail: test@example.com\n电话：123-456-7890"

    def test_parser_txt_class_method(self, txt_parser, sample_content):
        """测试parser_txt类方法"""
        result = RAGFlowTxtParser.parser_txt(sample_content, chunk_token_num=32)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

        # 检查返回格式：[[content, ""]]
        for chunk in result:
            assert isinstance(chunk, list)
            assert len(chunk) == 2
            assert isinstance(chunk[0], str)  # 文本内容
            assert isinstance(chunk[1], str)  # 元数据部分

    def test_parser_txt_with_empty_string(self):
        """测试解析空字符串"""
        result = RAGFlowTxtParser.parser_txt("", chunk_token_num=32)

        # 空字符串应该返回至少一个空块
        assert result is not None
        assert isinstance(result, list)

    def test_parser_txt_with_short_text(self):
        """测试解析短文本"""
        short_text = "短文本测试"
        result = RAGFlowTxtParser.parser_txt(short_text, chunk_token_num=128)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1

        # 验证内容是否正确包含
        combined_content = "".join(chunk[0] for chunk in result)
        assert "短文本测试" in combined_content

    def test_parser_txt_with_long_text(self):
        """测试解析长文本（会被切分）"""
        long_text = "这是一个很长的测试文本，用来验证文本切分功能。" * 50
        result = RAGFlowTxtParser.parser_txt(long_text, chunk_token_num=16)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1

        # 验证内容长度
        combined_content = "".join(chunk[0] for chunk in result)
        assert len(combined_content) > 0

    def test_parser_txt_with_different_delimiters(self):
        """测试使用不同分隔符"""
        text = "第一句。第二句！第三句？第四句；第五句："

        # 使用默认分隔符
        result1 = RAGFlowTxtParser.parser_txt(text, chunk_token_num=128)

        # 使用自定义分隔符
        result2 = RAGFlowTxtParser.parser_txt(text, chunk_token_num=128, delimiter="。！？")

        assert result1 is not None
        assert result2 is not None
        assert isinstance(result1, list)
        assert isinstance(result2, list)

    def test_parser_txt_with_unicode_content(self):
        """测试Unicode内容"""
        unicode_text = "🚀 测试中文内容\nEnglish content\nТест русского\n日本語テスト"
        result = RAGFlowTxtParser.parser_txt(unicode_text, chunk_token_num=32)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1

        combined_content = "".join(chunk[0] for chunk in result)
        assert "测试中文" in combined_content or "中文" in combined_content

    def test_parser_txt_with_numbers_and_special_chars(self):
        """测试包含数字和特殊字符的内容"""
        special_text = "2023年12月25日，价格是$99.99！\n联系电话：+86-123-4567-8900\n邮箱：user@domain.com"
        result = RAGFlowTxtParser.parser_txt(special_text, chunk_token_num=64)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1

        combined_content = "".join(chunk[0] for chunk in result)
        assert "2023" in combined_content or "$" in combined_content or "@" in combined_content

    def test_parser_txt_error_handling(self):
        """测试错误处理"""
        # 测试非字符串输入
        with pytest.raises(TypeError):
            RAGFlowTxtParser.parser_txt(123, chunk_token_num=32)

        with pytest.raises(TypeError):
            RAGFlowTxtParser.parser_txt(None, chunk_token_num=32)

        with pytest.raises(TypeError):
            RAGFlowTxtParser.parser_txt(["list", "input"], chunk_token_num=32)

    def test_call_method_with_temp_file(self, txt_parser, sample_content):
        """测试__call__方法（文件解析）"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(sample_content)
            temp_path = f.name

        try:
            # 使用__call__方法解析文件
            result = txt_parser(temp_path, chunk_token_num=32)

            assert result is not None
            assert isinstance(result, list)
            assert len(result) > 0

            # 验证内容
            combined_content = "".join(chunk[0] for chunk in result)
            assert "测试文档" in combined_content or "test@example.com" in combined_content

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_different_chunk_sizes(self):
        """测试不同的chunk大小"""
        text = "这是一个测试文本，用来验证不同chunk大小对结果的影响。" * 10

        # 测试不同的chunk大小
        for chunk_size in [8, 16, 32, 64, 128]:
            result = RAGFlowTxtParser.parser_txt(text, chunk_token_num=chunk_size)
            assert result is not None
            assert isinstance(result, list)
            assert len(result) >= 1

    def test_parser_consistency(self):
        """测试解析器的一致性"""
        text = "一致性测试文本。"

        # 多次解析相同文本应该得到相同结果
        result1 = RAGFlowTxtParser.parser_txt(text, chunk_token_num=32)
        result2 = RAGFlowTxtParser.parser_txt(text, chunk_token_num=32)

        assert result1 == result2

    def test_parser_with_whitespace_only(self):
        """测试只包含空白字符的文本"""
        whitespace_text = "   \n\t  \n   \t\n   "
        result = RAGFlowTxtParser.parser_txt(whitespace_text, chunk_token_num=32)

        assert result is not None
        assert isinstance(result, list)
        # 只包含空白字符的结果处理取决于具体实现