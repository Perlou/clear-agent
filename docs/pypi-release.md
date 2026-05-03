# PyPI 发布完整方案

> 把 `clear-agent` 发布到 PyPI（用户 `pip install clear-agent` 即可使用）的端到端 SOP。
> 本文档假定你已有 PyPI / TestPyPI 账号和 API token。

## 一、发布前体检清单

每次发版前**必须**逐项核对：

### 1.1 版本号

```bash
grep version pyproject.toml clear_agent/version.py
```

两处必须一致。**已发布的版本号不能复用**（PyPI 不允许覆盖同名同版本）。

### 1.2 元数据完整性

`pyproject.toml` 必填字段：
- `name = "clear-agent"`（PyPI 分发名，连字符）
- `version`（语义化版本：MAJOR.MINOR.PATCH，预发版用 `2.0.0a1` / `2.0.0b1` / `2.0.0rc1`）
- `description`（≤ 200 字符，PyPI 搜索结果显示）
- `readme = "README.md"`
- `license`
- `authors` / `maintainers`
- `requires-python = ">=3.10"`
- `keywords` / `classifiers`（影响 PyPI 搜索分类）
- `dependencies`（**严格 pin 上限**，避免 breaking change 影响用户）
- `[project.urls]`（Homepage / Repository / Bug Tracker / Changelog）

### 1.3 必备文件

```bash
ls README.md LICENSE pyproject.toml MANIFEST.in clear_agent/py.typed
```

- `README.md` → PyPI 长描述（注意：PyPI 不渲染 GitHub 私有 emoji / shield）
- `LICENSE` → 法律文件
- `MANIFEST.in` → 控制 sdist 包含哪些非 .py 文件
- `clear_agent/py.typed` → PEP 561 标记（让用户的 mypy 能识别本包的类型注解）

### 1.4 测试与构建

```bash
# 全量测试必须通过
pytest -q

# 类型检查（可选但建议）
mypy clear_agent

# 干净构建
rm -rf dist/ build/ *.egg-info clear_agent.egg-info
python -m build
```

输出应有两个文件：
- `dist/clear_agent-X.Y.Z-py3-none-any.whl`（wheel，pip 优先用）
- `dist/clear_agent-X.Y.Z.tar.gz`（sdist，源码包）

### 1.5 包内容审查

```bash
# 看 sdist 含哪些文件（避免误打包敏感数据）
tar -tzf dist/clear_agent-*.tar.gz | head -50

# 看 wheel 含哪些文件
unzip -l dist/clear_agent-*.whl | head -50
```

**红线**：
- ❌ 不该有 `.env` / `*.pyc` / `__pycache__/` / `.git/` / `memory/` / `tool-output/`
- ❌ 不该有测试数据 / 大文件 / 内部 spec
- ✅ 应该有 `clear_agent/**/*.py` + `LICENSE` + `README.md` + `py.typed`

### 1.6 元数据校验

```bash
twine check dist/*
```

必须输出 `PASSED`。常见失败原因：
- `description` 含未转义的 RST 标记
- README 含 PyPI 不支持的 markdown 扩展
- license 字段格式错误

### 1.7 干净环境安装验证

最容易遗漏的一步：

```bash
# 在临时虚拟环境装 wheel，import 应正常
mktmp_dir=$(mktemp -d)
python -m venv "$mktmp_dir/venv"
source "$mktmp_dir/venv/bin/activate"
pip install dist/clear_agent-*.whl
python -c "import clear_agent; print(clear_agent.__version__)"
python -c "from clear_agent import ClearAgentLLM, ReActAgent, build_supervisor_graph"
deactivate
rm -rf "$mktmp_dir"
```

**关键**：用 wheel 安装而不是 `pip install -e .`，模拟终端用户体验。

## 二、首次发布

### 2.1 注册 PyPI 账号

1. **TestPyPI**（测试发布用）：https://test.pypi.org/account/register/
2. **PyPI**（正式）：https://pypi.org/account/register/

两个**独立账号**（共用邮箱即可，但要分别注册）。

### 2.2 启用 2FA + 生成 API Token

PyPI 现在**强制要求 2FA**：

1. 登录 → Account settings → Two factor authentication → 启用（推荐 Authenticator app，备份恢复码）
2. Account settings → API tokens → "Add API token"
   - Token name: `clear-agent-publish`
   - Scope: 第一次发版选 "Entire account"（之后可改为 "Project: clear-agent" 限定）
   - **生成的 token 只显示一次** —— 立即复制保存到密码管理器

Token 形如：`pypi-AgEIcHlwaS5vcmcCJG...`（前缀 `pypi-` 是固定标识）

### 2.3 配置本地凭证

**选项 A：环境变量**（推荐 CI / 临时使用）
```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-AgEIcHlwaS5vcmcCJG..."
```

**选项 B：`~/.pypirc` 文件**（推荐本地长期）
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcCJG...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEIdGVzdC5weXBpLm9yZ...
```

**安全**：
- `~/.pypirc` 权限设 `600`：`chmod 600 ~/.pypirc`
- **绝不**把 token 提交到 git
- token 泄露 → 立即去 PyPI 撤销

### 2.4 先发 TestPyPI 验证

```bash
# 上传到 TestPyPI（不影响正式索引）
python -m twine upload --repository testpypi dist/*
```

成功后访问 `https://test.pypi.org/project/clear-agent/` 检查页面渲染。

**TestPyPI 验证安装**：
```bash
# TestPyPI 需要从 PyPI 拉依赖（因为 TestPyPI 上没有 openai 等包）
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            clear-agent

python -c "import clear_agent; print(clear_agent.__version__)"
```

确认无误后再发正式 PyPI。

### 2.5 发布到正式 PyPI

```bash
python -m twine upload dist/*
```

完成后：
- 访问 `https://pypi.org/project/clear-agent/` 查看页面
- 等 1-2 分钟 CDN 同步后任意人 `pip install clear-agent` 可装
- 在 PyPI 后台查看 Project 页面：可控制版本下架、设置主页等

### 2.6 打 Git 标签

```bash
git tag -a v2.0.0 -m "Release 2.0.0"
git push origin v2.0.0
```

GitHub 上可基于 tag 创建 Release（贴 changelog + 附件 dist/）。

## 三、后续版本发布流程

每次发新版重复以下：

```bash
# 1. 改版本号
sed -i '' 's/version = "2.0.0"/version = "2.0.1"/' pyproject.toml
sed -i '' 's/__version__ = "2.0.0"/__version__ = "2.0.1"/' clear_agent/version.py

# 2. 更新 CHANGELOG.md（记录本次改了什么）

# 3. 测试 + 构建
pytest -q
rm -rf dist/ build/ *.egg-info
python -m build
twine check dist/*

# 4. 干净环境验证（见 1.7）

# 5. 发版
python -m twine upload dist/*

# 6. 打 tag
git commit -am "release: 2.0.1"
git tag -a v2.0.1 -m "Release 2.0.1"
git push && git push --tags
```

## 四、版本号规范（语义化版本）

| 改动类型 | 版本号变化 | 例子 |
|---|---|---|
| 不兼容的 API 修改 | MAJOR +1 | `2.0.0` → `3.0.0` |
| 向后兼容的功能添加 | MINOR +1 | `2.0.0` → `2.1.0` |
| 向后兼容的 bug 修复 | PATCH +1 | `2.0.0` → `2.0.1` |
| 预发版（不稳定） | 加后缀 | `2.1.0a1` / `2.1.0b1` / `2.1.0rc1` |

`pip install clear-agent` 默认装稳定版；预发版需 `pip install --pre clear-agent`。

## 五、失败排错

### 5.1 `twine check` 失败：long_description 渲染错误

```
ERROR: `long_description` has syntax errors in markup
```

→ README.md 含 PyPI 不支持的 markdown。常见：
- 复杂表格嵌套
- HTML 标签
- 相对路径图片（PyPI 不渲染本地图片，要用绝对 URL）

**修复**：删除问题段落或转纯文本。

### 5.2 上传时 `403 Forbidden`

- token 错误 → 重新生成
- 项目名被占用 → 改 `name = "..."`
- 已发版本不能覆盖 → bump version

### 5.3 用户 `pip install` 报缺依赖

```
ERROR: Could not find a version that satisfies the requirement xxx
```

→ 你的 `dependencies` 中某个包不存在 / 版本号写错。
**修复**：本地 `pip install dist/*.whl` 重现 + 修 `pyproject.toml`。

### 5.4 发版后才发现严重 bug

PyPI **不允许**重新上传同版本号文件。**正确做法**：
1. 修 bug → bump 到下一个 patch（如 `2.0.0` → `2.0.1`）
2. 在 PyPI 后台**Yank** 掉问题版本（不是删除，是隐藏 + 警告）：
   - 进 https://pypi.org/manage/project/clear-agent/release/2.0.0/
   - 点 "Yank release"
3. 已经 `pip install==2.0.0` 的用户能继续用，新装的人会装 `2.0.1`

## 六、CI/CD 自动发版（推荐）

避免每次手动操作的最佳实践 —— GitHub Actions 自动发版：

`.github/workflows/release.yml`：

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write   # PyPI Trusted Publishing 用
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install build
        run: pip install build

      - name: Build distributions
        run: python -m build

      - name: Verify with twine
        run: |
          pip install twine
          twine check dist/*

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # 用 Trusted Publishing 无需 token
        # 在 PyPI 后台配置：Account → Publishing → Add a new pending publisher
```

**Trusted Publishing**（强烈推荐）：
- 不用在 GitHub Secrets 里存 PyPI token
- PyPI 通过 OIDC 验证 GitHub Actions workflow 身份
- 配置：https://docs.pypi.org/trusted-publishers/

触发：在 GitHub 上创建 Release（基于 git tag），自动跑 workflow → PyPI。

## 七、本项目的当前状态（2.0.0 首发清单）

```bash
✅ pyproject.toml 元数据完整（name / version / description / classifiers / keywords / urls）
✅ LICENSE 存在（CC BY-NC-SA 4.0）
✅ README.md 完整（按功能宣传）
✅ MANIFEST.in 完整
✅ clear_agent/py.typed 存在（PEP 561）
✅ 全量测试 744 passed + 2 skipped
✅ python -m build 通过
✅ twine check dist/* PASSED
✅ 干净环境 wheel 安装 + import 验证
```

**首发命令**（执行前请确认 PyPI 账号 + token 已就位）：

```bash
# 1. 体检
rm -rf dist/ build/ *.egg-info clear_agent.egg-info
pytest -q
python -m build
twine check dist/*

# 2. 先发 TestPyPI
python -m twine upload --repository testpypi dist/*

# 3. TestPyPI 装机验证
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            clear-agent
python -c "import clear_agent; print(clear_agent.__version__)"

# 4. 发正式 PyPI
python -m twine upload dist/*

# 5. 打 tag
git tag -a v2.0.0 -m "Release 2.0.0"
git push origin v2.0.0
```

## 八、附：CHANGELOG.md 建议模板

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-05-03

### Added
- StateGraph + Checkpoint + HITL
- Multi-agent (supervisor / swarm / handoff)
- MCP protocol integration
- Complete RAG pipeline + Memory layers
- Structured output + Eval-harness
- LCEL-lite Runnable + `|`
- Resilience (Retry / Fallback) + Pydantic tool auto-schema
- Multimodal (vision / audio) + Prompt caching
- Anthropic / Gemini async paths

### Changed
- ...

### Removed
- ...

[2.0.0]: https://github.com/Perlou/clear-agent/releases/tag/v2.0.0
```

---

## 九、相关链接

- PyPI 官方指南：https://packaging.python.org/en/latest/tutorials/packaging-projects/
- Trusted Publishing：https://docs.pypi.org/trusted-publishers/
- 语义化版本：https://semver.org/
- Keep a Changelog：https://keepachangelog.com/
- twine 文档：https://twine.readthedocs.io/
- build 文档：https://build.pypa.io/
