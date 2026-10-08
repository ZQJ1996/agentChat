"""Deterministic mock model for offline demos."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from app.models.base import ChatMessage, ChatResult, ModelProvider


class MockModelProvider(ModelProvider):
    name = "mock"
    dim = 64

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        user_text = _last_user(messages)
        system = _last_system(messages)
        content = _mock_reply(user_text, system, messages)
        prompt_tokens = max(1, len(user_text) // 2)
        completion_tokens = max(1, len(content) // 2)
        return ChatResult(
            content=content,
            model="mock-v1",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(t, self.dim) for t in texts]


def _last_user(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return messages[-1].content if messages else ""


def _last_system(messages: list[ChatMessage]) -> str:
    for m in messages:
        if m.role == "system":
            return m.content
    return ""


def _hash_embed(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    if not tokens:
        tokens = ["empty"]
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        for i in range(dim):
            byte = digest[i % len(digest)]
            vec[i] += ((byte / 255.0) * 2 - 1) / math.sqrt(len(tokens))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _mock_reply(user_text: str, system: str, messages: list[ChatMessage]) -> str:
    lower = user_text.lower()
    joined = (system + "\n" + user_text).lower()

    if "意图路由" in system or "intent" in joined and "confidence" in joined:
        intent, confidence, reason = _route_intent(user_text)
        return json.dumps(
            {"intent": intent, "confidence": confidence, "reason": reason},
            ensure_ascii=False,
        )

    if "检查回答" in system or '"ok"' in system.lower():
        return json.dumps({"ok": True, "reason": "mock verifier pass"}, ensure_ascii=False)

    if "只读数据分析" in system or "nl2sql" in joined:
        if any(k in user_text for k in ("订单", "销量", "统计", "多少", "数量")):
            return "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status;"
        return "SELECT priority, COUNT(*) AS cnt FROM tickets GROUP BY priority;"

    if "工单" in system:
        if any(k in user_text for k in ("创建", "开", "投诉", "升级")):
            return (
                "将为您创建工单。"
                ' TOOL_CALL create_ticket title="用户投诉" desc="用户反馈问题" '
                'priority="high" category="complaint"'
            )
        if any(k in user_text for k in ("查询", "状态", "进度")):
            m = re.search(r"(T?\d{3,})", user_text)
            tid = m.group(1) if m else "T1001"
            return f'查询工单。 TOOL_CALL query_ticket ticket_id="{tid}"'
        return "请提供工单号或需要办理的操作。"

    if "客服问答" in system or "证据" in system:
        evidence = ""
        question = user_text
        for m in messages:
            if m.role == "user":
                content = m.content
                if "用户问题" in content:
                    qm = re.search(r"用户问题：(.+?)(?:\n\n证据：|$)", content, re.DOTALL)
                    if qm:
                        question = qm.group(1).strip()
                if "证据" in content:
                    em = re.search(r"证据：\s*(.*)$", content, re.DOTALL)
                    evidence = (em.group(1).strip() if em else content)
        if "证据不足" in evidence or "无相关证据" in evidence or not evidence.strip():
            return "根据现有知识库无法确定答案，建议转人工处理。"
        snippet = _extract_relevant_snippet(question, evidence)
        if not snippet.strip() or snippet.strip().startswith("用户问题"):
            return "根据现有知识库无法确定答案，建议转人工处理。"
        return f"根据知识库：{snippet} [1]"

    if "主管" in system or "汇总" in system:
        return user_text if user_text else "已为您处理完毕。"

    # generic fallback used by chitchat / supervisor inputs
    if any(k in user_text for k in ("你好", "hello", "hi")):
        return "您好，我是企业知识助手，可协助产品咨询、订单查询、工单处理与数据统计。"
    if "退" in user_text or "换货" in user_text:
        return "退货需在签收 7 日内申请；质量问题换货时限为 15 日。[1]"
    if "工单" in user_text:
        return "我可以帮您创建或查询工单，请说明问题与优先级。"
    if "订单" in user_text:
        return "请提供订单号，我可查询物流与是否超时。"
    return f"已收到您的问题：{user_text[:120]}。我将基于企业知识与工具尽力协助。"


def _extract_relevant_snippet(question: str, evidence: str) -> str:
    blocks = re.split(r"\n\n+", evidence)
    q_chars = set(re.findall(r"[\u4e00-\u9fff]", question.lower()))
    q_words = set(re.findall(r"[a-z0-9_]+", question.lower()))
    scored_blocks: list[tuple[float, str]] = []
    for block in blocks:
        lines = block.strip().splitlines()
        body = "\n".join(lines[1:]).strip() if lines and lines[0].startswith("[") else block.strip()
        if not body:
            continue
        body_chars = set(re.findall(r"[\u4e00-\u9fff]", body.lower()))
        body_words = set(re.findall(r"[a-z0-9_]+", body.lower()))
        score = len(q_chars & body_chars) + 3 * len(q_words & body_words)
        # boost section header keywords present in question
        header = lines[0] if lines else ""
        if any(k in header for k in re.findall(r"[\u4e00-\u9fff]{2,}", question)):
            score += 5
        scored_blocks.append((score, body))
    scored_blocks.sort(key=lambda x: x[0], reverse=True)
    if not scored_blocks:
        return evidence[:240]
    # merge top-2 related chunks for better coverage
    top_bodies = [b for _, b in scored_blocks[:2]]
    best = "\n".join(top_bodies)
    sentences = re.split(r"(?<=[。！？\n])", best)
    picked = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s_chars = set(re.findall(r"[\u4e00-\u9fff]", s.lower()))
        s_words = set(re.findall(r"[a-z0-9_]+", s.lower()))
        if (q_chars and len(q_chars & s_chars) >= max(1, min(3, len(q_chars) // 5))) or (
            q_words & s_words
        ):
            picked.append(s)
    if picked:
        return "".join(picked[:6])[:500]
    return best[:500]


def _route_intent(text: str) -> tuple[str, float, str]:
    if any(k in text for k in ("忽略以上", "忽略之前", "jailbreak", "system prompt")):
        return "unclear", 0.2, "possible prompt injection"
    if any(k in text for k in ("统计", "报表", "多少单", "销量", "分析", "SQL", "有多少", "汇总")):
        return "data_query", 0.9, "analytics keywords"
    if any(k in text for k in ("订单", "物流", "快递", "发货", "签收")) and re.search(
        r"ORD\d+", text, re.IGNORECASE
    ):
        return "order_query", 0.93, "explicit order id"
    if any(k in text for k in ("创建", "开", "投诉", "升级", "派单")) and any(
        k in text for k in ("工单", "投诉")
    ) and not any(k in text for k in ("几天", "多久", "什么", "哪些", "如何", "怎么", "政策")):
        return "ticket", 0.92, "ticket action keywords"
    if any(k in text for k in ("查询工单", "工单状态", "工单进度")) or re.search(r"T\d{3,}", text):
        return "ticket", 0.9, "ticket query"
    if any(k in text for k in ("订单", "物流", "快递", "发货", "签收")):
        if any(k in text for k in ("几天", "多久", "政策", "可以投诉")):
            return "product_query", 0.88, "policy about logistics"
        if any(k in text for k in ("工单", "投诉", "开")):
            return "order_query", 0.9, "order with possible ticket follow-up"
        return "order_query", 0.88, "order keywords"
    if any(k in text for k in ("故障", "登录失败", "报错", "排查", "SSE", "检索", "账号锁定", "锁定")):
        return "technical", 0.84, "technical keywords"
    if any(
        k in text
        for k in (
            "退货",
            "换货",
            "退款",
            "政策",
            "功能",
            "权限",
            "产品",
            "FAQ",
            "知识",
            "优先级",
            "生命周期",
            "SLA",
            "渠道",
            "热线",
            "模型",
            "API Key",
            "mock",
        )
    ):
        return "product_query", 0.9, "product/policy keywords"
    if any(k in text for k in ("工单",)) and any(k in text for k in ("哪些", "什么", "如何", "怎么", "规范")):
        return "product_query", 0.86, "ticket knowledge question"
    if any(k in text for k in ("你好", "谢谢", "在吗", "hello", "hi")):
        return "chitchat", 0.8, "greeting"
    if len(text.strip()) < 2:
        return "unclear", 0.3, "empty"
    return "product_query", 0.55, "default product"
