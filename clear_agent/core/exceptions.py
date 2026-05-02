"""异常体系"""


class ClearAgentException(Exception):
    """ClearAgent基础异常类"""

    pass


class LLMException(ClearAgentException):
    """LLM相关异常"""

    pass


class AgentException(ClearAgentException):
    """Agent相关异常"""

    pass


class ConfigException(ClearAgentException):
    """配置相关异常"""

    pass


class ToolException(ClearAgentException):
    """工具相关异常"""

    pass
