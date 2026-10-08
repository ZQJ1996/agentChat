"""LangGraph shared state."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _merge_lists(a: list[str], b: list[str]) -> list[str]:
    return (a or []) + (b or [])


class AgentState(TypedDict, total=False):
    # 多轮对话历史，每项形如 {"role": "user|assistant", "content": "..."}
    messages: list[dict[str, str]]
    # 当前用户 ID，用于订单查询、偏好记忆等
    user_id: str
    # 角色：user / agent / admin，决定工具权限
    role: str
    # 会话 ID，LangGraph Checkpoint 按它区分多轮记忆
    thread_id: str
    # 本次请求追踪 ID，用于日志串联
    trace_id: str
    # 当前用户问题（可能已脱敏）
    query: str
    # 意图标签：product_query / ticket / order_query / data_query 等
    intent: str
    # 意图置信度 0~1，过低会转人工
    confidence: float
    # 实际走过的节点路径；用 _merge_lists 做列表累加而不是覆盖
    route_path: Annotated[list[str], _merge_lists]
    # RAG 检索到的证据片段（含 source、score、citation）
    evidence: list[dict[str, Any]]
    # 子 Agent 的中间回答
    agent_answer: str
    # Supervisor 汇总后给用户的最终回答
    final_answer: str
    # 是否需要转人工
    needs_human: bool
    # 本次创建或查询到的工单号
    ticket_id: str | None
    # 本次解析/查询到的订单号
    order_id: str | None
    # 工单工具已重试次数
    tool_retries: int
    # 累计 token 用量
    token_usage: int
    # 最近一次错误信息（工具失败等）
    error: str | None
    # 数据分析 Agent 的 SQL 查询结果行
    data_result: list[dict[str, Any]]
    # 额外元数据：路由原因、SQL、是否继续开单等
    meta: dict[str, Any]
