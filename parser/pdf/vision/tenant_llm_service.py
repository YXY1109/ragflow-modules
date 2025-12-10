from enum import StrEnum


from parser.pdf.vision.llm.chat_model import OpenAI_APIChat
from parser.pdf.vision.llm.cv_model import OpenAI_APICV


class LLMType(StrEnum):
    CHAT = 'chat'
    EMBEDDING = 'embedding'
    SPEECH2TEXT = 'speech2text'
    IMAGE2TEXT = 'image2text'
    RERANK = 'rerank'
    TTS = 'tts'
    OCR = 'ocr'


class TenantLLMService(object):

    @classmethod
    def model_instance(cls, tenant_id, llm_type, llm_name=None, lang="Chinese", **kwargs):
        if llm_type == LLMType.IMAGE2TEXT.value:
            return OpenAI_APICV("","")
        elif llm_type == LLMType.CHAT.value:
            return OpenAI_APIChat()
        return None


class LLM4Tenant:
    def __init__(self, tenant_id, llm_type, llm_name=None, lang="Chinese", **kwargs):
        self.tenant_id = tenant_id
        self.llm_type = llm_type
        self.llm_name = llm_name
        self.mdl = TenantLLMService.model_instance(tenant_id, llm_type, llm_name, lang=lang, **kwargs)
