# 工单处理流程

## 工单生命周期
`open` → `in_progress` → `resolved` → `closed`  
也可从任意未关闭状态转 `escalated`（升级人工）。

## 创建规范
- **标题**：简洁描述问题，例如「订单超时未送达」。
- **描述**：包含订单号、发生时间、已尝试操作。
- **分类**：product / logistics / billing / technical / complaint。
- **优先级**：
  - low：咨询类
  - medium：一般故障
  - high：物流超时 >7 天、大额资损风险
  - urgent：生产事故、支付故障

## 自动处理规则
1. 用户投诉物流超时且订单超 7 天未签收 → 自动创建投诉工单，优先级 high。
2. 知识库无法回答且置信度低 → 不创建工单，转为人工会话。
3. 工具调用失败 → 重试 1 次 → 语义化报错 → 转人工。

## 查询与更新
- `query_ticket(ticket_id)`：查询状态、优先级、负责人。
- `update_ticket_status(ticket_id, status)`：更新状态。
- 禁止普通用户修改他人工单。

## SLA（示例）
- high：2 小时内响应
- medium：8 小时内响应
- low：24 小时内响应
