# AtlasCare — KPI Framework

## 1. Business KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| Autonomous resolution rate | > 70% of Tier-1 interactions handled without human escalation | escalated_cases / total_requests |
| Escalation accuracy | > 95% of escalations are valid (not false positives) | human_reviewed_valid / total_escalations |
| Customer satisfaction (CSAT) | > 4.2 / 5.0 | Post-interaction survey |
| First-contact resolution rate | > 80% | Resolved in single session / total sessions |
| Average handling time | < 30 seconds for J1; < 90 seconds for J2 | latency_ms from trace |

---

## 2. Quality KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| Response grounding rate | 100% — no fabricated data | Audit: response fields traceable to tool output |
| Hallucination incidents | 0 per week | Manual audit + automated regex checks |
| J1 latency SLA | < 3,000 ms p99 | trace.latency_ms |
| J2 step completion rate | > 95% all steps succeed | successful_steps / planned_steps |
| Planner parse success rate | > 99% valid JSON plans | (total - PlannerError) / total |
| Refund amount accuracy | 100% — Decimal arithmetic, no float errors | Audit refund records vs order totals |

---

## 3. Safety KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| Threshold enforcement rate | 100% — zero autonomous refunds above Rs.25,000 | payment_tool.process_refund calls with amount > 25000 |
| GR-001 false negative rate | 0% — no high-value refund reaches payment tool | Audit: trace tool_calls for process_refund on escalation cases |
| GR-004 trigger rate | 0 per week (should never fire in production) | guardrail_events with rule_id=GR-004 |
| Cross-customer access attempts | Tracked and alerted | ownership_denied status in trace tool_calls |
| Escalation case trace linkage | 100% — every case has trace_id | CRM audit: cases without trace_id |

---

## 4. Operational KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| API availability | > 99.9% uptime | Health check success rate |
| p50 latency | < 800 ms | trace.latency_ms distribution |
| p99 latency | < 3,000 ms | trace.latency_ms distribution |
| Payment gateway retry rate | < 5% | retried_calls / total_payment_calls |
| Payment gateway failure rate | < 1% after retries | failed_after_retries / total_payment_calls |
| LLM error rate | < 0.5% | PlannerError / total_requests |
| Error rate (5xx) | < 0.1% | HTTP 500 responses / total_requests |
| Trace completeness | 100% — every request has trace_id | Missing trace_id in response |

---

## 5. Audit & Compliance KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| Escalation case creation latency | < 5 seconds from detection | created_at - request_start |
| Case handoff summary completeness | 100% contain: customer_id, order_id, trace_id, amount, reason | CRM case description audit |
| Refund record immutability | 0 mutations post-creation | Append-only store audit |
| Sensitive field leak rate | 0 — API keys, tokens never in logs or responses | Log audit for REDACTED markers |