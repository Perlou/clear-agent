"""
ClearAgent 安装配置

这个文件主要用于向后兼容，现代Python项目推荐使用pyproject.toml
"""

from setuptools import setup, find_packages

# 从pyproject.toml读取配置，这里提供一个简化版本
setup(
    name="clear-agent",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(include=["clear_agent*"]),
    python_requires=">=3.10",
)
