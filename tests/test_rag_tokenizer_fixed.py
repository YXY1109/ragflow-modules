import pytest
from pathlib import Path
import sys

# 添加项目根目录到sys.path
sys.path.append(str(Path(__file__).parent.parent))

from nlp.rag_tokenizer import RagTokenizer, tokenizer, tokenize, fine_grained_tokenize


class TestRagTokenizer:
    """测试RAG分词器"""

    @pytest.fixture
    def rag_tokenizer(self):
        """创建RagTokenizer实例"""
        return RagTokenizer()

    def test_tokenizer_initialization(self, rag_tokenizer):
        """测试分词器初始化"""
        assert rag_tokenizer is not None
        assert hasattr(rag_tokenizer, 'tokenize')
        assert hasattr(rag_tokenizer, 'fine_grained_tokenize')

    def test_tokenize_chinese_text(self, rag_tokenizer):
        """测试中文文本分词"""
        text = "这是一个中文分词测试"
        result = rag_tokenizer.tokenize(text)

        assert isinstance(result, str)
        assert len(result) > 0
        # 中文分词应该返回空格分隔的tokens
        tokens = result.split()
        assert isinstance(tokens, list)
        assert len(tokens) >= 3  # 至少应该切分出一些词

    def test_tokenize_english_text(self, rag_tokenizer):
        """测试英文文本分词"""
        text = "This is an English tokenization test"
        result = rag_tokenizer.tokenize(text)

        assert isinstance(result, str)
        assert len(result) > 0
        tokens = result.split()
        # 英文应该按空格分词，但可能会有词干提取和词形还原
        assert len(tokens) >= 4  # 至少应该有一些词
        # 检查是否包含相关的词汇（可能有词形变化）
        has_english_words = any(
            any(eng_word in token for eng_word in ["engli", "token", "test"])
            for token in tokens
        )
        assert has_english_words

    def test_tokenize_mixed_text(self, rag_tokenizer):
        """测试中英文混合文本分词"""
        text = "这是 mixed 中英文 text 测试"
        result = rag_tokenizer.tokenize(text)

        assert isinstance(result, str)
        assert len(result) > 0
        tokens = result.split()
        # 应该能处理混合语言
        assert any("中文" in token or "text" in token for token in tokens)

    def test_tokenize_empty_text(self, rag_tokenizer):
        """测试空文本分词"""
        result = rag_tokenizer.tokenize("")
        assert isinstance(result, str)
        assert result == ""

        result = rag_tokenizer.tokenize("   ")
        assert isinstance(result, str)
        # 空白字符的结果取决于具体实现

    def test_tokenize_punctuation(self, rag_tokenizer):
        """测试包含标点符号的文本分词"""
        text = "你好，世界！Hello, world!"
        result = rag_tokenizer.tokenize(text)

        assert isinstance(result, str)
        assert len(result) > 0
        tokens = result.split()
        # 标点符号会被转换为空格或去除
        assert len(tokens) > 0

    def test_fine_grained_tokenize(self, rag_tokenizer):
        """测试细粒度分词"""
        # 先进行基本分词
        text = "中华人民共和国"
        basic_tokens = rag_tokenizer.tokenize(text)

        # fine_grained_tokenize需要字符串参数，不是列表
        result = rag_tokenizer.fine_grained_tokenize(basic_tokens)

        assert isinstance(result, str)
        assert len(result) > 0
        # 细粒度分词应该返回更细粒度的结果
        fine_tokens = result.split()
        assert isinstance(fine_tokens, list)

    def test_global_tokenizer_function(self):
        """测试全局分词器函数"""
        text = "测试全局分词器"
        result = tokenize(text)

        assert isinstance(result, str)
        assert len(result) > 0
        tokens = result.split()
        assert isinstance(tokens, list)

    def test_global_fine_grained_tokenize_function(self):
        """测试全局细粒度分词器函数"""
        text = "测试全局细粒度分词器"
        basic_result = tokenize(text)

        # fine_grained_tokenize需要字符串参数，不是列表
        result = fine_grained_tokenize(basic_result)

        assert isinstance(result, str)
        assert len(result) > 0
        fine_tokens = result.split()
        assert isinstance(fine_tokens, list)

    def test_tokenizer_singleton(self):
        """测试分词器单例"""
        text = "测试"
        result1 = tokenize(text)
        result2 = tokenize(text)

        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert result1 == result2  # 相同输入应该得到相同结果

    def test_tokenize_numbers(self, rag_tokenizer):
        """测试包含数字的文本"""
        text = "2023年12月25日，价格是$99.99"
        result = rag_tokenizer.tokenize(text)

        assert isinstance(result, str)
        assert len(result) > 0
        tokens = result.split()
        # 应该能处理数字和特殊字符
        assert any("2023" in token or "99" in token for token in tokens)

    def test_tokenize_long_text(self, rag_tokenizer):
        """测试长文本分词"""
        text = "这是一个很长的文本测试，包含多个句子和词汇。我们希望验证分词器在处理长文本时的性能和准确性。" * 10
        result = rag_tokenizer.tokenize(text)

        assert isinstance(result, str)
        assert len(result) > 0
        tokens = result.split()
        # 长文本应该能正常分词
        assert len(tokens) > 10

    def test_tokenize_special_characters(self, rag_tokenizer):
        """测试特殊字符处理"""
        text = "测试@#$%^&*()特殊字符"
        result = rag_tokenizer.tokenize(text)

        assert isinstance(result, str)
        assert len(result) > 0
        tokens = result.split()
        # 特殊字符会被转换为空格或去除
        assert len(tokens) > 0

    def test_unicode_content(self, rag_tokenizer):
        """测试Unicode内容"""
        unicode_text = "🚀 Unicode test 测试 🌍"
        result = rag_tokenizer.tokenize(unicode_text)

        assert isinstance(result, str)
        # Unicode字符的处理取决于具体实现
        # 至少不应该出错

    def test_case_insensitive_processing(self, rag_tokenizer):
        """测试大小写不敏感处理"""
        upper_text = "HELLO WORLD"
        lower_text = "hello world"

        result1 = rag_tokenizer.tokenize(upper_text)
        result2 = rag_tokenizer.tokenize(lower_text)

        assert isinstance(result1, str)
        assert isinstance(result2, str)
        # 由于会被转换为小写，结果应该相同
        assert result1 == result2

    def test_token_consistency(self, rag_tokenizer):
        """测试分词一致性"""
        text = "一致性测试文本"

        # 多次分词相同文本应该得到相同结果
        result1 = rag_tokenizer.tokenize(text)
        result2 = rag_tokenizer.tokenize(text)
        result3 = rag_tokenizer.tokenize(text)

        assert result1 == result2 == result3

    def test_single_character_tokens(self, rag_tokenizer):
        """测试单个字符的处理"""
        text = "a b c 一 二 三"
        result = rag_tokenizer.tokenize(text)

        assert isinstance(result, str)
        tokens = result.split()
        assert len(tokens) >= 3

    def test_mixed_language_processing(self, rag_tokenizer):
        """测试混合语言处理"""
        mixed_text = "English中文混合日本語Korean"
        result = rag_tokenizer.tokenize(mixed_text)

        assert isinstance(result, str)
        tokens = result.split()
        assert len(tokens) > 0
        # 应该能处理多种语言混合