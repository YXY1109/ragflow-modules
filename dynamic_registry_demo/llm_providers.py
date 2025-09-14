from dynamic_registry_demo.base import BaseLLM


class Base(BaseLLM):
    """Base占位符类"""
    pass


class OpenAILLM(Base):
    _FACTORY_NAME = "openai"

    def generate(self, prompt: str) -> str:
        return f"OpenAI 响应: {prompt}"


class TongyiLLM(Base):
    _FACTORY_NAME = "tongyi"

    def generate(self, prompt: str) -> str:
        return f"通义千问 响应: {prompt}"


class MoonshotLLM(Base):
    _FACTORY_NAME = "moonshot"

    def generate(self, prompt: str) -> str:
        return f"Moonshot 响应: {prompt}"
