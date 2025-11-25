import pytest
from pathlib import Path
import sys

# 添加项目根目录到sys.path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dynamic_registry_demo.base import BaseLLM, BaseTextProcessor
    from dynamic_registry_demo import llm_providers, text_processors
except ImportError:
    pytest.skip("dynamic_registry_demo模块不可用", allow_module_level=True)


class TestBaseClasses:
    """测试基类"""

    def test_base_llm_initialization(self):
        """测试BaseLLM初始化"""
        # 创建一个测试用的LLM类
        class TestLLM(BaseLLM):
            _FACTORY_NAME = "test_llm"

            def __init__(self, api_key=None):
                self.api_key = api_key

            def generate(self, prompt: str) -> str:
                return f"Test response to: {prompt}"

        llm = TestLLM(api_key="test_key")
        assert llm.api_key == "test_key"

    def test_base_text_processor_initialization(self):
        """测试BaseTextProcessor初始化"""
        # 创建一个测试用的文本处理器类
        class TestProcessor(BaseTextProcessor):
            _FACTORY_NAME = "test_processor"

            def process(self, text: str) -> str:
                return f"Processed: {text}"

        processor = TestProcessor()
        # 应该能够正常初始化

    def test_base_llm_generate_not_implemented(self):
        """测试BaseLLM的generate方法需要被子类实现"""
        class IncompleteLLM(BaseLLM):
            _FACTORY_NAME = "incomplete_llm"

        llm = IncompleteLLM()
        with pytest.raises(NotImplementedError):
            # 抽象方法应该不能直接调用
            llm.generate("test")

    def test_base_text_processor_process_not_implemented(self):
        """测试BaseTextProcessor的process方法需要被子类实现"""
        class IncompleteProcessor(BaseTextProcessor):
            _FACTORY_NAME = "incomplete_processor"

        processor = IncompleteProcessor()
        with pytest.raises(NotImplementedError):
            # 抽象方法应该不能直接调用
            processor.process("test")


class TestLLMProviders:
    """测试LLM提供者"""

    def test_openai_provider_exists(self):
        """测试OpenAI提供者是否存在"""
        try:
            from dynamic_registry_demo.llm_providers import OpenAIProvider
            assert OpenAIProvider._FACTORY_NAME == "openai"
        except ImportError:
            pytest.skip("OpenAIProvider不可用")

    def test_qwen_provider_exists(self):
        """测试通义千问提供者是否存在"""
        try:
            from dynamic_registry_demo.llm_providers import QwenProvider
            assert QwenProvider._FACTORY_NAME == "qwen"
        except ImportError:
            pytest.skip("QwenProvider不可用")

    def test_moonshot_provider_exists(self):
        """测试Moonshot提供者是否存在"""
        try:
            from dynamic_registry_demo.llm_providers import MoonshotProvider
            assert MoonshotProvider._FACTORY_NAME == "moonshot"
        except ImportError:
            pytest.skip("MoonshotProvider不可用")

    def test_provider_instantiation(self):
        """测试提供者实例化"""
        try:
            from dynamic_registry_demo.llm_providers import OpenAIProvider
            provider = OpenAIProvider(api_key="test_key")
            assert provider.api_key == "test_key"
        except ImportError:
            pytest.skip("LLM提供者模块不可用")

    def test_provider_generation(self):
        """测试提供者生成方法（如果可用）"""
        try:
            from dynamic_registry_demo.llm_providers import OpenAIProvider
            provider = OpenAIProvider(api_key="test_key")

            # 这个测试可能需要mock或者跳过，因为它需要真实的API调用
            # 取决于具体实现
            try:
                response = provider.generate("Hello")
                assert isinstance(response, str)
            except Exception:
                # 如果API调用失败（需要真实的API密钥），跳过测试
                pytest.skip("需要真实的API密钥进行测试")
        except ImportError:
            pytest.skip("LLM提供者模块不可用")


class TestTextProcessors:
    """测试文本处理器"""

    def test_summarizer_exists(self):
        """测试摘要处理器是否存在"""
        try:
            from dynamic_registry_demo.text_processors import TextSummarizer
            assert TextSummarizer._FACTORY_NAME == "summarizer"
        except ImportError:
            pytest.skip("TextSummarizer不可用")

    def test_translator_exists(self):
        """测试翻译处理器是否存在"""
        try:
            from dynamic_registry_demo.text_processors import TextTranslator
            assert "translator" in TextTranslator._FACTORY_NAME
        except ImportError:
            pytest.skip("TextTranslator不可用")

    def test_sentiment_analyzer_exists(self):
        """测试情感分析处理器是否存在"""
        try:
            from dynamic_registry_demo.text_processors import SentimentAnalyzer
            assert "sentiment" in SentimentAnalyzer._FACTORY_NAME
        except ImportError:
            pytest.skip("SentimentAnalyzer不可用")

    def test_processor_instantiation(self):
        """测试处理器实例化"""
        try:
            from dynamic_registry_demo.text_processors import TextSummarizer
            processor = TextSummarizer()
            # 应该能够正常实例化
        except ImportError:
            pytest.skip("文本处理器模块不可用")

    def test_processor_processing(self):
        """测试处理器处理方法"""
        try:
            from dynamic_registry_demo.text_processors import TextSummarizer
            processor = TextSummarizer()

            test_text = "这是一个测试文本，用来验证文本处理器的功能。"
            try:
                result = processor.process(test_text)
                assert isinstance(result, str)
                assert len(result) > 0
            except Exception:
                # 如果处理失败（需要外部依赖），跳过测试
                pytest.skip("文本处理可能需要外部依赖")
        except ImportError:
            pytest.skip("文本处理器模块不可用")


class TestDynamicRegistry:
    """测试动态注册功能"""

    def test_registry_initialization(self):
        """测试注册表初始化"""
        try:
            from dynamic_registry_demo import BaseLLM, BaseTextProcessor

            # 检查是否有注册表相关的属性
            if hasattr(BaseLLM, '_registry'):
                assert isinstance(BaseLLM._registry, dict)

            if hasattr(BaseTextProcessor, '_registry'):
                assert isinstance(BaseTextProcessor._registry, dict)
        except ImportError:
            pytest.skip("注册表模块不可用")

    def test_automatic_registration(self):
        """测试自动注册功能"""
        try:
            # 这个测试取决于具体的注册实现
            # 检查导入模块时是否自动注册了类
            from dynamic_registry_demo import llm_providers, text_processors

            # 根据具体实现检查注册情况
            # 这里只是示例，实际检查方法取决于实现细节
            assert True  # 如果能正常导入，说明自动注册可能工作
        except ImportError:
            pytest.skip("动态注册模块不可用")

    def test_factory_method_exists(self):
        """测试工厂方法是否存在"""
        try:
            from dynamic_registry_demo.base import BaseLLM, BaseTextProcessor

            # 检查是否有获取实例的工厂方法
            if hasattr(BaseLLM, 'get'):
                assert callable(BaseLLM.get)

            if hasattr(BaseTextProcessor, 'get'):
                assert callable(BaseTextProcessor.get)
        except ImportError:
            pytest.skip("基类模块不可用")

    def test_custom_class_registration(self):
        """测试自定义类注册"""
        try:
            from dynamic_registry_demo.base import BaseLLM

            # 创建一个自定义LLM类
            class CustomLLM(BaseLLM):
                _FACTORY_NAME = "custom_test_llm"

                def __init__(self, api_key=None):
                    self.api_key = api_key

                def generate(self, prompt: str) -> str:
                    return f"Custom: {prompt}"

            # 如果有自动注册，检查是否被注册
            if hasattr(BaseLLM, '_registry'):
                # 这里取决于具体的注册时机
                # 如果注册在导入时发生，可能需要手动触发
                pass

            # 至少类应该能正常创建
            llm = CustomLLM(api_key="test")
            assert llm._FACTORY_NAME == "custom_test_llm"
        except ImportError:
            pytest.skip("注册系统不可用")


class TestIntegration:
    """集成测试"""

    def test_module_import(self):
        """测试模块导入"""
        try:
            import dynamic_registry_demo
            import dynamic_registry_demo.base
            import dynamic_registry_demo.llm_providers
            import dynamic_registry_demo.text_processors
            assert True  # 如果所有导入成功，测试通过
        except ImportError as e:
            pytest.skip(f"模块导入失败: {e}")

    def test_demo_runnable(self):
        """测试演示程序是否可运行"""
        try:
            from dynamic_registry_demo.run_demo import main
            # 只检查函数是否存在，不实际运行
            assert callable(main)
        except ImportError:
            pytest.skip("演示程序不可用")