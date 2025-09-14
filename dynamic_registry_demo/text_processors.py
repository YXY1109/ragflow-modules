from dynamic_registry_demo.base import BaseTextProcessor



class Base(BaseTextProcessor):
    """Base占位符类"""
    pass


class Summarizer(Base):
    _FACTORY_NAME = "summarizer"

    def process(self, text: str) -> str:
        return f"[摘要] {text[:30]}..."


class Translator(Base):
    _FACTORY_NAME = "translator"

    def process(self, text: str) -> str:
        return f"[翻译] {text} (模拟翻译)"


class SentimentAnalyzer(Base):
    _FACTORY_NAME = "sentiment"

    def process(self, text: str) -> str:
        return f"[情感分析] 正面 (模拟分析结果)"
