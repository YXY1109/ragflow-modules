# ragflow的模块学习

- dynamic_registry_demo: 动态注册模块
- files：测试文件
- nlp：nlp相关的代码
- parser：pdf解析

# 学习目标

- 文件解析
    - txt：只有个native和email使用，都是直接使用的，没有封装
    - pdf
    - word
    - excel
- 使用mineru将pdf转为md，同时将图片上传到minio：ragflow已经集成了mineru
- 使用ragflow切分文档，先将md文档切分
- 部署qwen3 embedding和rerank模型服务
- 部署bge-m3 embedding模型服务
- 将切分后的结果存入milvus，包括稠密向量和稀疏向量
- 将切分后的结果存入es
- 使用GraphRAG和LightRAG处理文档
- 使用raptor处理文档
- 召回逻辑参考ragflow
- mem0保存上下文
- 使用uv管理python依赖