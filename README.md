# 基于ragflow的学习记录

- 使用mineru将pdf转为md，同时将图片上传到minio
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