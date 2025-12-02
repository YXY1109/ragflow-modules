import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

load_dotenv()

# 创建HTTP客户端，增加超时配置
http_client = httpx.Client(
    timeout=httpx.Timeout(
        connect=10.0,  # 连接超时30秒
        read=60.0,  # 读取超时60秒
        write=30.0,  # 写入超时30秒
        pool=30.0  # 连接池超时30秒
    ),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
)

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    http_client=http_client
)

# 使用类型化的消息参数
messages: list[ChatCompletionUserMessageParam] = [
    {
        "role": "user",
        "content": "你是谁"
    }
]

completion = client.chat.completions.create(
    extra_body={"enable_thinking": False},
    model="glm-4.6",
    messages=messages
)
result = completion.choices[0].message.content
# 安全地打印结果，处理特殊字符
try:
    print(result)
except UnicodeEncodeError:
    print("结果包含特殊字符，但API调用成功")
    print(f"结果长度: {len(result) if result else 0} 字符")
