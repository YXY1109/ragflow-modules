class BaseLLM:
    """所有LLM模型的基类"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("子类必须实现generate方法")


class BaseTextProcessor:
    """所有文本处理器的基类"""

    def process(self, text: str) -> str:
        raise NotImplementedError("子类必须实现process方法")
