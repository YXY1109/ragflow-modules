from dynamic_registry_demo import supported_llms, supported_processors


def main():
    print("===== 动态模型注册演示 =====")

    # 显示所有注册的LLM模型
    print("\n已注册的LLM模型:")
    for name, model_class in supported_llms.items():
        print(f"- {name}: {model_class.__name__}")

    # 显示所有注册的文本处理器
    print("\n已注册的文本处理器:")
    for name, processor_class in supported_processors.items():
        print(f"- {name}: {processor_class.__name__}")

    # 测试LLM模型生成
    print("\n===== LLM模型测试 =====")
    test_prompt = "你好，这是一个测试"

    for name, model_class in supported_llms.items():
        model = model_class()
        result = model.generate(test_prompt)
        print(f"\n{name} 模型结果:")
        print(result)

    # 测试文本处理器
    print("\n===== 文本处理器测试 =====")
    test_text = "这是一段用于测试文本处理器的示例文本，包含了多种处理需求。"

    for name, processor_class in supported_processors.items():
        processor = processor_class()
        result = processor.process(test_text)
        print(f"\n{name} 处理器结果:")
        print(result)


if __name__ == "__main__":
    main()
