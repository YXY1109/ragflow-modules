import importlib
import inspect

# 创建模型注册表
supported_llms = globals().get("supported_llms", {})
supported_processors = globals().get("supported_processors", {})

# 模块映射配置
MODULE_MAPPING = {
    "llm_providers": supported_llms,
    "text_processors": supported_processors
}

package_name = __name__

# 动态导入模块并注册模型
for module_name, mapping_dict in MODULE_MAPPING.items():
    full_module_name = f"{package_name}.{module_name}"
    module = importlib.import_module(full_module_name)

    base_class = None
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and name == "Base":
            base_class = obj
            break

    if base_class is not None:
        for _, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                    issubclass(obj, base_class) and
                    obj is not base_class and
                    hasattr(obj, "_FACTORY_NAME")):
                if isinstance(obj._FACTORY_NAME, list):
                    for factory_name in obj._FACTORY_NAME:
                        mapping_dict[factory_name] = obj
                else:
                    mapping_dict[obj._FACTORY_NAME] = obj

# 导出公共接口
__all__ = ["supported_llms", "supported_processors"]
