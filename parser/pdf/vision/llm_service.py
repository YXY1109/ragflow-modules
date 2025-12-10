from parser.pdf.vision.tenant_llm_service import LLM4Tenant


class LLMBundle(LLM4Tenant):
    def __init__(self, tenant_id, llm_type, llm_name=None, lang="Chinese", **kwargs):
        super().__init__(tenant_id, llm_type, llm_name, lang, **kwargs)

    def describe_with_prompt(self, image, prompt):
        txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)
        return txt
