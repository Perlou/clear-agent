#!/usr/bin/env bash
# =============================================================================
# scripts/release.sh — clear-agent 一键发布脚本
# =============================================================================
# 用法：
#   ./scripts/release.sh                  # 交互式，发到 PyPI（推荐）
#   ./scripts/release.sh --test           # 发到 TestPyPI
#   ./scripts/release.sh --dry-run        # 只构建 + 校验，不上传
#   ./scripts/release.sh --skip-tests     # 跳过 pytest（紧急 bugfix 用）
#   ./scripts/release.sh --skip-tag       # 不打 git tag
#   ./scripts/release.sh --skip-clean-install  # 不做干净环境装机验证
#   ./scripts/release.sh --yes            # 不交互，全部 yes（CI 用）
#   ./scripts/release.sh --version 2.0.1  # 自动 bump 版本号到 2.0.1
#   ./scripts/release.sh --bump patch     # 自动 bump（patch / minor / major）
#
# 凭证：
#   1. 环境变量 TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-XXX
#   2. ~/.pypirc 配置（推荐本地）
#   3. 都没配 → 脚本会引导你配置
#
# 退出码：
#   0  成功
#   1  通用失败
#   2  参数错误
#   3  环境/工具缺失
#   4  版本检查失败
#   5  测试失败
#   6  构建失败
#   7  上传失败
# =============================================================================

set -euo pipefail

# ----------- 颜色 -----------
if [[ -t 1 ]]; then
    R='\033[0;31m'  # red
    G='\033[0;32m'  # green
    Y='\033[0;33m'  # yellow
    B='\033[0;34m'  # blue
    M='\033[0;35m'  # magenta
    C='\033[0;36m'  # cyan
    W='\033[1;37m'  # white bold
    N='\033[0m'     # reset
else
    R= G= Y= B= M= C= W= N=
fi

step()    { echo -e "\n${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"; echo -e "${B}▶ $*${N}"; echo -e "${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"; }
ok()      { echo -e "${G}✅ $*${N}"; }
warn()    { echo -e "${Y}⚠️  $*${N}"; }
err()     { echo -e "${R}❌ $*${N}" >&2; }
info()    { echo -e "${C}ℹ️  $*${N}"; }
header()  { echo -e "${M}$*${N}"; }

# ----------- 默认参数 -----------
TARGET="pypi"               # pypi / testpypi
DRY_RUN=0
SKIP_TESTS=0
SKIP_TAG=0
SKIP_CLEAN_INSTALL=0
YES=0
NEW_VERSION=""
BUMP=""

# ----------- 项目根目录 -----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)"
cd "$ROOT_DIR"

# ----------- 解析参数 -----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)               TARGET="testpypi"; shift ;;
        --dry-run)            DRY_RUN=1; shift ;;
        --skip-tests)         SKIP_TESTS=1; shift ;;
        --skip-tag)           SKIP_TAG=1; shift ;;
        --skip-clean-install) SKIP_CLEAN_INSTALL=1; shift ;;
        --yes|-y)             YES=1; shift ;;
        --version)            NEW_VERSION="$2"; shift 2 ;;
        --bump)               BUMP="$2"; shift 2 ;;
        -h|--help)
            sed -n '4,30p' "$0" | sed 's/^# //;s/^#//'
            exit 0
            ;;
        *)
            err "未知参数: $1"
            echo "运行 $0 --help 查看用法"
            exit 2
            ;;
    esac
done

# ----------- 交互式确认 helper -----------
confirm() {
    local prompt="$1"
    if [[ $YES -eq 1 ]]; then
        info "$prompt → 自动 yes（--yes 模式）"
        return 0
    fi
    echo -en "${Y}$prompt [y/N]:${N} "
    read -r ans
    [[ "$ans" =~ ^[yY] ]]
}

# =============================================================================
# Phase 0: 工具检查
# =============================================================================
step "Phase 0  工具与环境检查"

# 找 python：优先 .venv → python3 → python
PY=""
if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
else
    err "找不到 python。请装 Python 3.10+"
    exit 3
fi
ok "Python: $PY ($($PY --version))"

# 检查 build / twine
for mod in build twine; do
    if ! $PY -c "import $mod" 2>/dev/null; then
        warn "缺少 $mod。尝试安装..."
        if command -v uv &>/dev/null; then
            uv pip install -q "$mod" || { err "uv 安装 $mod 失败"; exit 3; }
        else
            $PY -m pip install -q "$mod" || { err "pip 安装 $mod 失败"; exit 3; }
        fi
    fi
    ok "$mod 已就绪"
done

# Git
if ! command -v git &>/dev/null; then
    warn "git 未安装；--skip-tag 自动开启"
    SKIP_TAG=1
else
    ok "git: $(git --version | awk '{print $3}')"
fi

# =============================================================================
# Phase 1: 工作区状态检查
# =============================================================================
step "Phase 1  Git 工作区检查"

if [[ $SKIP_TAG -eq 0 ]]; then
    if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
        warn "工作区有未提交改动："
        git status --short | head -20
        if ! confirm "继续发版？（建议先 commit）"; then
            err "已取消"
            exit 1
        fi
    else
        ok "工作区干净"
    fi

    BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
    info "当前分支: $BRANCH"

    if [[ "$BRANCH" != "main" && "$BRANCH" != "master" ]]; then
        warn "当前不在 main/master 分支"
        if ! confirm "继续从 '$BRANCH' 发版？"; then
            exit 1
        fi
    fi
fi

# =============================================================================
# Phase 2: 版本号处理
# =============================================================================
step "Phase 2  版本号"

PYPROJECT="pyproject.toml"
VERSION_PY="clear_agent/version.py"

current_version() {
    grep -m1 '^version = ' "$PYPROJECT" | sed -E 's/version = "(.+)"/\1/'
}

bump_version() {
    local cur="$1" kind="$2"
    IFS='.' read -ra parts <<< "$cur"
    local major="${parts[0]}" minor="${parts[1]}" patch="${parts[2]}"
    case "$kind" in
        major) echo "$((major + 1)).0.0" ;;
        minor) echo "${major}.$((minor + 1)).0" ;;
        patch) echo "${major}.${minor}.$((patch + 1))" ;;
        *) err "--bump 只支持 major / minor / patch"; exit 2 ;;
    esac
}

write_version() {
    local v="$1"
    # macOS 和 Linux 兼容的 sed -i
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/^version = \".*\"/version = \"$v\"/" "$PYPROJECT"
        sed -i '' "s/^__version__ = \".*\"/__version__ = \"$v\"/" "$VERSION_PY"
    else
        sed -i "s/^version = \".*\"/version = \"$v\"/" "$PYPROJECT"
        sed -i "s/^__version__ = \".*\"/__version__ = \"$v\"/" "$VERSION_PY"
    fi
}

CUR=$(current_version)
PY_VER=$(grep -m1 '__version__ = ' "$VERSION_PY" | sed -E "s/__version__ = \"(.+)\"/\1/")

if [[ "$CUR" != "$PY_VER" ]]; then
    err "版本号不一致："
    err "  pyproject.toml:        $CUR"
    err "  clear_agent/version.py: $PY_VER"
    exit 4
fi
info "当前版本: $CUR"

# 自动 bump
if [[ -n "$NEW_VERSION" ]]; then
    info "目标版本: $NEW_VERSION（--version 指定）"
    if confirm "把版本号从 $CUR 改成 $NEW_VERSION？"; then
        write_version "$NEW_VERSION"
        CUR="$NEW_VERSION"
        ok "已更新版本号"
    fi
elif [[ -n "$BUMP" ]]; then
    NV=$(bump_version "$CUR" "$BUMP")
    info "目标版本: $NV（--bump $BUMP）"
    if confirm "把版本号从 $CUR 改成 $NV？"; then
        write_version "$NV"
        CUR="$NV"
        ok "已更新版本号"
    fi
fi

VERSION="$CUR"
TAG="v$VERSION"

# 检查 PyPI 是否已存在
if [[ "$TARGET" == "pypi" ]]; then
    INDEX_URL="https://pypi.org/pypi/clear-agent/$VERSION/json"
elif [[ "$TARGET" == "testpypi" ]]; then
    INDEX_URL="https://test.pypi.org/pypi/clear-agent/$VERSION/json"
fi

if command -v curl &>/dev/null; then
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" "$INDEX_URL" 2>/dev/null || echo "000")
    if [[ "$HTTP" == "200" ]]; then
        err "${TARGET} 上 clear-agent==${VERSION} 已存在（PyPI 不允许覆盖）"
        err "请先 bump 版本号：./scripts/release.sh --bump patch"
        exit 4
    elif [[ "$HTTP" == "404" ]]; then
        ok "${TARGET} 上 clear-agent==${VERSION} 未占用"
    else
        warn "无法验证 ${TARGET} 上的版本占用情况（HTTP $HTTP），继续..."
    fi
fi

# 检查是否已有同 tag
if [[ $SKIP_TAG -eq 0 ]] && git rev-parse "$TAG" &>/dev/null; then
    warn "git tag '$TAG' 已存在"
    if ! confirm "继续？（已有 tag 时通常应该 bump 版本）"; then
        exit 1
    fi
fi

# =============================================================================
# Phase 3: 必备文件检查
# =============================================================================
step "Phase 3  必备文件"

REQUIRED=("README.md" "LICENSE" "pyproject.toml" "MANIFEST.in" "clear_agent/py.typed")
for f in "${REQUIRED[@]}"; do
    if [[ -f "$f" ]]; then
        ok "$f"
    else
        err "缺失: $f"
        exit 4
    fi
done

# =============================================================================
# Phase 4: 测试
# =============================================================================
if [[ $SKIP_TESTS -eq 0 ]]; then
    step "Phase 4  全量 pytest"

    if ! $PY -c "import pytest" 2>/dev/null; then
        warn "pytest 未安装，跳过测试"
    else
        if $PY -m pytest -q --tb=line 2>&1 | tee /tmp/release-pytest.log | tail -3; then
            ok "全量测试通过"
        else
            # pytest 自身可能因部分集成测试缺 API key 而失败；但核心套件不应失败
            err "测试失败"
            tail -30 /tmp/release-pytest.log
            if ! confirm "强行继续？（不推荐！）"; then
                exit 5
            fi
        fi
    fi
else
    warn "跳过测试（--skip-tests）"
fi

# =============================================================================
# Phase 5: 清理 + 构建
# =============================================================================
step "Phase 5  清理 + 构建"

rm -rf dist/ build/ ./*.egg-info ./clear_agent.egg-info 2>/dev/null || true
ok "已清理 dist/ build/ *.egg-info"

if ! $PY -m build 2>&1 | tail -5; then
    err "构建失败"
    exit 6
fi
ok "构建完成"

ls -lh dist/

# =============================================================================
# Phase 6: twine check
# =============================================================================
step "Phase 6  twine check"

if ! $PY -m twine check dist/* 2>&1; then
    err "twine check 失败"
    exit 6
fi
ok "twine check PASSED"

# =============================================================================
# Phase 7: 包内容审查
# =============================================================================
step "Phase 7  包内容审查"

WHEEL="dist/clear_agent-${VERSION}-py3-none-any.whl"
SDIST="dist/clear_agent-${VERSION}.tar.gz"

if [[ ! -f "$WHEEL" || ! -f "$SDIST" ]]; then
    err "构建产物缺失：$WHEEL 或 $SDIST"
    exit 6
fi

# 红线检查
SUSPECT=$(unzip -l "$WHEEL" 2>/dev/null | awk '{print $4}' | grep -E '\.env$|^\.git/|__pycache__|\.pyc$|\.DS_Store$|^memory/|^tool-output/' || true)
if [[ -n "$SUSPECT" ]]; then
    err "wheel 中发现敏感/不该入包的文件："
    echo "$SUSPECT"
    exit 6
fi
ok "包内容审查通过（无敏感文件）"

NFILES=$(unzip -l "$WHEEL" 2>/dev/null | tail -1 | awk '{print $2}')
WSIZE=$(ls -lh "$WHEEL" | awk '{print $5}')
SSIZE=$(ls -lh "$SDIST" | awk '{print $5}')
info "Wheel: $WSIZE / $NFILES 文件"
info "Sdist: $SSIZE"

# =============================================================================
# Phase 8: 干净环境装机验证
# =============================================================================
if [[ $SKIP_CLEAN_INSTALL -eq 0 ]]; then
    step "Phase 8  干净环境装机验证"

    TMP=$(mktemp -d)
    trap "rm -rf '$TMP'" EXIT

    if command -v uv &>/dev/null; then
        uv venv -p 3.10 "$TMP/venv" 2>&1 | tail -1
        VPY="$TMP/venv/bin/python"
        VENV_PIP="uv pip install --python $VPY"
    else
        $PY -m venv "$TMP/venv"
        VPY="$TMP/venv/bin/python"
        VENV_PIP="$VPY -m pip install -q"
    fi

    info "在 $TMP/venv 安装 wheel..."
    if ! $VENV_PIP "$WHEEL" 2>&1 | tail -3; then
        err "wheel 在干净环境安装失败"
        exit 6
    fi

    if ! "$VPY" -c "
import clear_agent
assert clear_agent.__version__ == '$VERSION', f'version mismatch: {clear_agent.__version__} != $VERSION'
from clear_agent import (
    ClearAgentLLM, ReActAgent, SimpleAgent, ReflectionAgent, PlanSolveAgent,
    ToolRegistry, CalculatorTool, build_react_graph,
    StateGraph, CompiledGraph, START, END, RunConfig,
    Handoff, build_supervisor_graph, build_swarm_graph,
)
print('✅ all imports OK / version=$VERSION')
" 2>&1; then
        err "干净环境 import 失败"
        exit 6
    fi
    ok "干净环境装机验证通过"
fi

# =============================================================================
# Phase 9: 上传
# =============================================================================
step "Phase 9  上传"

if [[ $DRY_RUN -eq 1 ]]; then
    warn "DRY-RUN 模式 —— 跳过上传"
    info "构建产物保留在 ./dist/"
    info "实际上传命令: $PY -m twine upload --repository ${TARGET} dist/*"
    exit 0
fi

# 凭证检查
HAS_CRED=0
if [[ -n "${TWINE_USERNAME:-}" && -n "${TWINE_PASSWORD:-}" ]]; then
    info "使用环境变量凭证 (TWINE_USERNAME=$TWINE_USERNAME)"
    HAS_CRED=1
elif [[ -f "$HOME/.pypirc" ]]; then
    if grep -q "$TARGET" "$HOME/.pypirc"; then
        info "使用 ~/.pypirc 凭证"
        HAS_CRED=1
    fi
fi

if [[ $HAS_CRED -eq 0 ]]; then
    warn "未检测到 PyPI 凭证。可选方式："
    echo "  1. 环境变量:"
    echo "       export TWINE_USERNAME=__token__"
    echo "       export TWINE_PASSWORD=pypi-xxxxx"
    echo "  2. ~/.pypirc 文件（详见 docs/pypi-release.md §2.3）"
    echo ""
    if ! confirm "继续？（twine 会交互式询问 username/password）"; then
        exit 1
    fi
fi

if [[ "$TARGET" == "testpypi" ]]; then
    UPLOAD_URL="https://test.pypi.org/project/clear-agent/$VERSION/"
    INSTALL_CMD="pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ clear-agent==$VERSION"
else
    UPLOAD_URL="https://pypi.org/project/clear-agent/$VERSION/"
    INSTALL_CMD="pip install clear-agent==$VERSION"
fi

header "即将上传到 ${TARGET}: clear-agent==${VERSION}"
if ! confirm "确认上传？"; then
    err "已取消"
    exit 1
fi

if ! $PY -m twine upload --repository "$TARGET" --non-interactive dist/* 2>&1; then
    err "上传失败"
    err "如果是 'File already exists' → 已发版本不能覆盖，请 bump 版本"
    err "如果是 '403 Forbidden' → token 错误或权限不足"
    err "其他错误 → 检查网络 + 凭证"
    exit 7
fi

ok "上传成功"
header "🎉 ${UPLOAD_URL}"
info "用户安装命令: ${INSTALL_CMD}"

# =============================================================================
# Phase 10: Git tag
# =============================================================================
if [[ $SKIP_TAG -eq 0 && "$TARGET" == "pypi" ]]; then
    step "Phase 10  Git tag"

    if git rev-parse "$TAG" &>/dev/null; then
        warn "tag $TAG 已存在，跳过"
    else
        if confirm "创建并推送 git tag '$TAG'？"; then
            git tag -a "$TAG" -m "Release $VERSION"
            if confirm "推送到 origin？"; then
                git push origin "$TAG"
                ok "tag $TAG 已推送"
                info "GitHub Release 页面: https://github.com/Perlou/clear-agent/releases/new?tag=${TAG}"
            else
                info "tag 已创建本地，未推送（之后用：git push origin $TAG）"
            fi
        fi
    fi
fi

# =============================================================================
# 收尾
# =============================================================================
echo ""
echo -e "${G}╔══════════════════════════════════════════════╗${N}"
echo -e "${G}║  🎉 clear-agent ${VERSION} 发布完成！${N}"
echo -e "${G}╚══════════════════════════════════════════════╝${N}"
echo ""
info "PyPI 页面: $UPLOAD_URL"
info "用户安装:  $INSTALL_CMD"
echo ""
echo "如果发现严重 bug 想下架（不会真删，仅警告）："
echo "  访问 https://pypi.org/manage/project/clear-agent/release/${VERSION}/ → Yank"
echo "  然后修 bug → bump 版本 → 重发"


