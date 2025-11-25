# 测试文档

这个目录包含了项目的完整测试套件，使用pytest框架进行测试。

## 测试结构

```
tests/
├── README.md                     # 本文档
├── conftest.py                   # pytest配置和共享fixture
├── pytest.ini                   # pytest配置文件
├── test_class.py                 # 基础测试类示例
├── test_fixture.py               # fixture使用示例
├── test_sample.py                # 基础测试示例
├── test_dynamic_registry.py      # 动态注册系统测试
├── test_markdown_parser_simple.py # Markdown解析器测试
├── test_rag_tokenizer_fixed.py   # RAG分词器测试
└── test_txt_parser_simple.py     # 文本解析器测试
```

## 运行测试

### 运行所有测试
```bash
# 使用uv运行测试
uv run pytest tests/

# 或者激活虚拟环境后运行
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pytest tests/
```

### 运行特定测试文件
```bash
# 运行单个测试文件
uv run pytest tests/test_rag_tokenizer_fixed.py

# 运行特定测试类
uv run pytest tests/test_rag_tokenizer_fixed.py::TestRagTokenizer

# 运行特定测试方法
uv run pytest tests/test_rag_tokenizer_fixed.py::TestRagTokenizer::test_tokenize_chinese_text
```

### 运行带详细输出的测试
```bash
# 详细输出
uv run pytest tests/ -v

# 显示测试持续时间
uv run pytest tests/ --durations=10

# 生成HTML报告
uv run pytest tests/ --html=reports/test_report.html --self-contained-html
```

### 运行带覆盖率的测试
```bash
# 生成覆盖率报告
uv run pytest tests/ --cov=parser --cov=nlp --cov=dynamic_registry_demo

# 生成HTML覆盖率报告
uv run pytest tests/ --cov=parser --cov=nlp --cov=dynamic_registry_demo --cov-report=html:reports/coverage
```

## 测试配置

### pytest.ini
主要的pytest配置文件，包含：
- 测试文件匹配模式
- 默认选项
- 标记定义
- 日志配置
- 超时设置

### conftest.py
包含共享的fixture和测试配置：
- `temp_dir`: 临时目录
- `sample_text`: 示例文本
- `sample_markdown_content`: 示例Markdown内容
- `temp_text_file`: 临时文本文件创建器
- `temp_markdown_file`: 临时Markdown文件创建器

## 测试模块说明

### 核心模块测试

#### test_txt_parser_simple.py
测试RAGFlowTxtParser类：
- 基本文本解析功能
- 不同语言和字符编码处理
- 错误处理和边界情况
- 类方法和实例方法测试

#### test_markdown_parser_simple.py
测试Markdown解析相关功能：
- RAGFlowMarkdownParser表格提取
- MarkdownElementExtractor元素提取
- 各种Markdown语法支持
- 错误处理

#### test_rag_tokenizer_fixed.py
测试RAGTokenizer分词器：
- 中英文分词功能
- 细粒度分词
- Unicode和特殊字符处理
- 全局函数测试

#### test_dynamic_registry.py
测试动态注册系统：
- 基类功能测试
- LLM提供者测试
- 文本处理器测试
- 注册机制测试

### NLP模块测试

#### test_tokens_num.py
测试token计算功能：
- 字符串token计算
- 消息token计算
- 不同语言和格式处理

#### test_merge.py
测试文本合并功能：
- 多种文本合并策略
- 空内容处理
- Unicode文本处理

## 测试标记

使用以下标记来分类测试：
- `@pytest.mark.unit`: 单元测试
- `@pytest.mark.integration`: 集成测试
- `@pytest.mark.slow`: 慢速测试
- `@pytest.mark.parser`: 解析器相关测试
- `@pytest.mark.nlp`: NLP相关测试
- `@pytest.mark.registry`: 注册系统相关测试

### 按标记运行测试
```bash
# 只运行单元测试
uv run pytest tests/ -m unit

# 只运行NLP相关测试
uv run pytest tests/ -m nlp

# 排除慢速测试
uv run pytest tests/ -m "not slow"
```

## 最佳实践

1. **编写测试时**：
   - 使用描述性的测试方法名
   - 为每个测试添加清晰的文档字符串
   - 使用适当的fixture来减少重复代码
   - 测试正常情况和边界情况

2. **运行测试时**：
   - 在提交代码前运行完整测试套件
   - 使用覆盖率报告确保代码覆盖
   - 定期运行测试以检测回归

3. **添加新功能时**：
   - 为新功能编写对应的测试
   - 更新现有的测试以适应新功能
   - 确保测试覆盖所有重要的代码路径

## 报告和输出

测试运行后会生成以下报告：
- 控制台输出：实时测试结果
- HTML报告：`reports/test_report.html`
- 覆盖率报告：`reports/coverage/index.html`
- 日志文件：`reports/pytest.log`

## 故障排除

### 常见问题

1. **导入错误**：确保项目根目录在Python路径中
2. **依赖缺失**：运行`uv sync --group dev`安装开发依赖
3. **权限问题**：确保有写入reports目录的权限
4. **编码问题**：测试文件使用UTF-8编码

### 调试测试
```bash
# 在第一个失败的测试时停止
uv run pytest tests/ -x

# 显示详细输出
uv run pytest tests/ -v -s

# 运行特定测试并进入调试器
uv run pytest tests/ --pdb
```

## 持续集成

测试配置为与CI/CD系统兼容：
- 使用标准退出码
- 支持并行执行
- 生成机器可读的报告格式
- 兼容常见的CI环境