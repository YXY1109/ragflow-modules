# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个RAGFlow模块学习项目，用于学习和实现RAG（检索增强生成）相关的各种技术模块。项目主要包含以下核心功能模块：

- **文件解析器**：支持txt、pdf、word、excel、markdown等格式的文档解析
- **NLP处理**：包含中文分词器、token计算、文本合并等自然语言处理功能
- **动态注册系统**：实现模型和处理器的动态注册机制
- **文档切分**：实现基于RAGFlow的文档切分逻辑

## 常用命令

### 环境管理
```bash
# 安装依赖
uv sync

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 激活虚拟环境 (Linux/Mac)
source .venv/bin/activate
```

### 运行演示
```bash
# 运行动态注册演示
python dynamic_registry_demo/run_demo.py

# 运行主程序
python main.py

# 测试文本解析器
python parser/txt_parser/txt_parser.py

# 测试markdown解析器
python parser/markdown_parser/markdown_test.py

# 测试分词器
python nlp/rag_tokenizer.py
```

### 开发工具
```bash
# 安装开发依赖（如果有的话）
uv sync --dev

# 运行代码检查（如果配置了）
uv run ruff check .
uv run ruff format .
```

## 高层架构

### 解析器架构
项目采用策略模式实现多格式文档解析：

- **parser/docx/** - Word文档解析，包含5种不同的解析策略：
  - `docx_parser.py` - 基础RAGFlow风格解析器
  - `laws.py` - 法律文档专用解析器
  - `manual.py` - 手册文档解析器，支持图文配对
  - `naive.py` - 简单直接的解析器
  - `qa.py` - 问答对解析器

- **parser/txt/** - 纯文本解析器，支持native和email格式

- **parser/markdown/** - Markdown解析器，支持表格提取和图片处理

- **parser/excel/** - Excel解析器，支持QA表格提取

- **parser/html/** - HTML解析器

- **parser/json/** - JSON解析器

- **parser/ppt/** - PowerPoint解析器

### 动态注册系统
实现了基于类继承的自动注册机制：

- 所有解析器都实现统一的`__call__`接口
- 支持多种文档格式和解析策略
- 便于扩展新的解析器和处理方式

### NLP处理流水线
提供完整的中文NLP处理能力：

1. **RAGTokenizer** - 基于HUQIE的中英文分词器
2. **文本合并** - 支持多种合并策略的文本处理
3. **Token计算** - 精确的token数量计算

## 核心模块结构

1. **dynamic_registry_demo/** - 动态注册系统
   - `base.py`: 定义基类 `BaseLLM` 和 `BaseTextProcessor`
   - `llm_providers.py`: LLM模型提供者实现（OpenAI、通义千问、Moonshot）
   - `text_processors.py`: 文本处理器实现（摘要、翻译、情感分析）
   - `__init__.py`: 动态注册逻辑，自动发现并注册子类
   - `run_demo.py`: 演示程序

2. **parser/** - 文档解析模块
   - `txt_parser/`: 纯文本解析器，支持普通文本和邮件格式
   - `markdown_parser/`: Markdown解析器，支持表格提取和图片处理
   - `utils.py`: 解析器通用工具函数

3. **nlp/** - 自然语言处理
   - `rag_tokenizer.py`: 基于HUQIE的中英文分词器，支持词典管理
   - `tokens_num.py`: Token计算工具
   - `merge.py`: 文本合并功能

4. **files/** - 测试文件
   - `markdown/`: Markdown测试文档

### 动态注册机制

项目实现了一个优雅的动态注册系统：

- 所有模型都继承自对应的基类（`BaseLLM` 或 `BaseTextProcessor`）
- 通过 `_FACTORY_NAME` 属性指定注册名称
- 在模块导入时自动扫描并注册符合条件的子类
- 支持单个名称或名称列表注册

### 分词器架构

`RagTokenizer` 是项目的核心NLP组件：

- 基于字典树（Trie）的分词算法
- 支持中英文混合文本处理
- 实现了最大正向和最大逆向匹配
- 包含词频统计和词性标注功能
- 支持用户自定义词典

## 依赖管理

项目使用 `uv` 作为依赖管理工具：

- 主要依赖：`nltk`, `markdown`, `beautifulsoup4`, `loguru`, `pillow`, `datrie` 等
- 支持Python 3.12+
- 所有依赖都在 `pyproject.toml` 中声明
- 使用 `uv.lock` 锁定版本以确保可重现构建

## 开发注意事项

1. **新增模型/处理器**：继承相应基类，添加 `_FACTORY_NAME` 属性，放在对应模块中即可自动注册

2. **分词器使用**：项目提供了全局分词器实例 `tokenizer`，可直接使用 `tokenize()` 和 `fine_grained_tokenize()` 函数

3. **解析器扩展**：新的解析器应实现 `__call__` 方法，遵循现有的接口规范

4. **代码风格**：遵循PEP 8规范，使用中文注释，保持代码简洁清晰

## 测试和验证

项目使用pytest作为测试框架，配置完整的测试套件：

### 测试命令
```bash
# 运行所有测试
uv run pytest tests/

# 运行特定测试文件
uv run pytest tests/test_rag_tokenizer_fixed.py

# 运行带覆盖率的测试
uv run pytest tests/ --cov=parser --cov=nlp --cov=dynamic_registry_demo

# 生成HTML测试报告
uv run pytest tests/ --html=reports/test_report.html --self-contained-html

# 按标记运行测试
uv run pytest tests/ -m unit          # 单元测试
uv run pytest tests/ -m nlp           # NLP相关测试
uv run pytest tests/ -m parser         # 解析器测试
uv run pytest tests/ -m "not slow"     # 排除慢速测试
```

### 测试结构
- `tests/conftest.py` - 共享fixture和配置
- `tests/test_dynamic_registry.py` - 动态注册系统测试
- `tests/test_markdown_parser_simple.py` - Markdown解析器测试
- `tests/test_rag_tokenizer_fixed.py` - RAG分词器测试
- `tests/test_txt_parser_simple.py` - 文本解析器测试
- `tests/test_tokens_num.py` - Token计算测试
- `tests/test_merge.py` - 文本合并测试

### 模块独立测试
每个解析器模块都有独立的main.py用于快速测试：
```bash
# 测试DOCX解析器（包含多种策略）
python parser/docx/main.py

# 测试其他解析器
python parser/txt/main.py
python parser/markdown/main.py
python parser/excel/main.py
```

### 验证文件
- `files/` 目录包含各种格式的测试文档
- 支持中文、英文、混合语言文档测试
- 包含表格、图片、复杂格式的测试用例