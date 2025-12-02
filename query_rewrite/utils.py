# ========
# 公共工具函数 - 基于OpenRouter API
# ========
import json
import re
from typing import Dict, Any, Optional

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

# OpenRouter API配置
OPENROUTER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENROUTER_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DEFAULT_MODEL = "glm-4.6"

# 创建HTTP客户端，增加超时配置
http_client = httpx.Client(
    timeout=httpx.Timeout(
        connect=30.0,  # 连接超时30秒
        read=120.0,  # 读取超时120秒
        write=30.0,  # 写入超时30秒
        pool=30.0  # 连接池超时30秒
    ),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
)

# 创建OpenAI客户端
openai_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    http_client=http_client
)


class QueryRewriterConfig:
    """Query改写器配置类"""

    def __init__(
            self,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            model: str = DEFAULT_MODEL,
            timeout_connect: float = 30.0,
            timeout_read: float = 120.0,
            timeout_write: float = 30.0,
            timeout_pool: float = 30.0,
            max_connections: int = 10,
            max_keepalive_connections: int = 5
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.model = model
        self.timeout_config = {
            "connect": timeout_connect,
            "read": timeout_read,
            "write": timeout_write,
            "pool": timeout_pool
        }
        self.limits_config = {
            "max_connections": max_connections,
            "max_keepalive_connections": max_keepalive_connections
        }


def get_completion(
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        config: Optional[QueryRewriterConfig] = None
) -> str:
    """
    使用OpenRouter API调用LLM

    Args:
        prompt: 提示词
        model: 模型名称，如果为None则使用配置中的模型
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大token数
        config: 配置对象

    Returns:
        LLM响应文本
    """
    if config is None:
        config = QueryRewriterConfig()

    # 使用类型化的消息参数
    messages: list[ChatCompletionUserMessageParam] = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        completion = openai_client.chat.completions.create(
            model=model or config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        result = completion.choices[0].message.content
        return result if result else ""

    except Exception as e:
        print(f"OpenRouter API调用失败: {e}")
        return ""


def get_json_completion(
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,  # JSON任务使用较低温度
        max_tokens: Optional[int] = None,
        config: Optional[QueryRewriterConfig] = None
) -> Dict[str, Any]:
    """
    使用OpenRouter API获取JSON格式的响应

    Args:
        prompt: 提示词
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大token数
        config: 配置对象

    Returns:
        解析后的JSON对象
    """
    # 在提示词中明确要求JSON输出
    json_prompt = f"""{prompt}

请直接返回JSON格式的响应，不要包含任何其他文本或说明。"""

    response = get_completion(json_prompt, model, temperature, max_tokens, config)
    cleaned_response = preprocess_json_response(response)

    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        print(f"原始响应: {response}")
        return {"error": "JSON解析失败", "raw_response": response}


def preprocess_json_response(response: str) -> str:
    """
    预处理JSON响应，提取JSON部分
    """
    # 尝试提取JSON部分
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return response


class BaseQueryRewriter:
    """Query改写器基类"""

    def __init__(self, model: str = "qwen-turbo-latest"):
        self.model = model

    def rewrite(self, query: str, context: str = "") -> Any:
        """改写查询的接口，子类需要实现"""
        raise NotImplementedError("子类必须实现rewrite方法")
