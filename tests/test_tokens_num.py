import pytest
from pathlib import Path
import sys

# 添加项目根目录到sys.path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from nlp.tokens_num import num_tokens_from_string, num_tokens_from_messages
except ImportError:
    pytest.skip("tokens_num模块不可用", allow_module_level=True)


class TestTokensNum:
    """测试Token计算模块"""

    def test_num_tokens_from_string_empty(self):
        """测试空字符串的token计算"""
        result = num_tokens_from_string("")
        assert isinstance(result, int)
        assert result >= 0

    def test_num_tokens_from_string_simple_text(self):
        """测试简单文本的token计算"""
        text = "Hello, world!"
        result = num_tokens_from_string(text)

        assert isinstance(result, int)
        assert result > 0
        # 英文文本通常每个单词约1个token
        assert 1 <= result <= 5

    def test_num_tokens_from_string_chinese_text(self):
        """测试中文文本的token计算"""
        text = "你好，世界！"
        result = num_tokens_from_string(text)

        assert isinstance(result, int)
        assert result > 0
        # 中文通常每个字符约1-2个token
        assert 1 <= result <= 10

    def test_num_tokens_from_string_mixed_text(self):
        """测试中英文混合文本的token计算"""
        text = "Hello 你好 world 世界"
        result = num_tokens_from_string(text)

        assert isinstance(result, int)
        assert result > 0

    def test_num_tokens_from_string_long_text(self):
        """测试长文本的token计算"""
        text = "这是一个很长的文本，用来测试token计算的准确性。" * 100
        result = num_tokens_from_string(text)

        assert isinstance(result, int)
        assert result > 0
        # 长文本应该有更多的token
        assert result > 100

    def test_num_tokens_from_string_with_numbers(self):
        """测试包含数字的文本"""
        text = "The year is 2023, price is $99.99"
        result = num_tokens_from_string(text)

        assert isinstance(result, int)
        assert result > 0

    def test_num_tokens_from_string_with_punctuation(self):
        """测试包含标点符号的文本"""
        text = "Hello, world! How are you? I'm fine."
        result = num_tokens_from_string(text)

        assert isinstance(result, int)
        assert result > 0

    def test_num_tokens_from_messages_empty(self):
        """测试空消息列表的token计算"""
        messages = []
        result = num_tokens_from_messages(messages)

        assert isinstance(result, int)
        assert result >= 0

    def test_num_tokens_from_messages_single_message(self):
        """测试单条消息的token计算"""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        result = num_tokens_from_messages(messages)

        assert isinstance(result, int)
        assert result > 0

    def test_num_tokens_from_messages_multiple_messages(self):
        """测试多条消息的token计算"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        result = num_tokens_from_messages(messages)

        assert isinstance(result, int)
        assert result > 0
        # 多条消息应该比单条消息有更多的token
        single_msg_result = num_tokens_from_messages([messages[1]])
        assert result > single_msg_result

    def test_num_tokens_from_messages_different_roles(self):
        """测试不同角色的消息"""
        roles = ["system", "user", "assistant", "developer"]
        content = "Test message content"

        results = []
        for role in roles:
            messages = [{"role": role, "content": content}]
            result = num_tokens_from_messages(messages)
            results.append(result)

        # 所有结果都应该是正整数
        for result in results:
            assert isinstance(result, int)
            assert result > 0

    def test_num_tokens_from_messages_long_content(self):
        """测试长内容的消息"""
        long_content = "这是一个很长的消息内容，" * 200
        messages = [
            {"role": "user", "content": long_content}
        ]
        result = num_tokens_from_messages(messages)

        assert isinstance(result, int)
        assert result > 100  # 长内容应该有更多token

    def test_num_tokens_consistency(self):
        """测试token计算的一致性"""
        text = "Hello, world! 你好，世界！"

        # 多次计算相同文本应该得到相同结果
        result1 = num_tokens_from_string(text)
        result2 = num_tokens_from_string(text)

        assert result1 == result2

    def test_num_tokens_from_unicode_text(self):
        """测试Unicode文本的token计算"""
        unicode_text = "🚀 Hello 世界 🌍 Test 测试"
        result = num_tokens_from_string(unicode_text)

        assert isinstance(result, int)
        assert result > 0

    def test_num_tokens_comparison(self):
        """测试不同长度文本的token数量比较"""
        short_text = "Hi"
        medium_text = "Hello, how are you today?"
        long_text = "This is a much longer text that should contain more tokens than the shorter texts."

        short_result = num_tokens_from_string(short_text)
        medium_result = num_tokens_from_string(medium_text)
        long_result = num_tokens_from_string(long_text)

        assert short_result < medium_result < long_result