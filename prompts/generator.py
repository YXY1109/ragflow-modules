import jinja2

from prompts.template import load_prompt

VISION_LLM_DESCRIBE_PROMPT = load_prompt("vision_llm_describe_prompt")
PROMPT_JINJA_ENV = jinja2.Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)


def vision_llm_describe_prompt(page=None) -> str:
    template = PROMPT_JINJA_ENV.from_string(VISION_LLM_DESCRIBE_PROMPT)

    return template.render(page=page)
