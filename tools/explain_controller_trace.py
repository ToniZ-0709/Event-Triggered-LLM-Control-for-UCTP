"""Render an evidence-linked, offline account of controller review events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def display(value: Any) -> str:
    """Encode trace values as data, including missing values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def seq(value: Any) -> list:
    return value if isinstance(value, list) else []


def record(lines: list[str], label: str, value: Any, source: str) -> None:
    lines.append(f"- **{label}:** `{display(value)}` (trace: `{source}`)")


def render_event(event: dict, line_number: int) -> str:
    state = obj(event.get("state"))
    action = obj(event.get("action"))
    outcome = obj(event.get("outcome"))
    metadata = obj(event.get("metadata"))
    roles = seq(event.get("roles"))
    generation = event.get("generation")
    lines = [f"## Review tại dòng {line_number}", ""]
    if any(obj(r).get("role") == "inspector" for r in roles):
        lines.append("StatePacket dưới đây là packet của Inspector. Input Planner phụ thuộc phiên bản controller: bản v2 gửi toàn bộ packet, các phiên bản khác có thể lọc region. Trace này không tự xác nhận payload Planner; đối chiếu source của run.")
        lines.append("")
    record(lines, "Generation", generation, "generation")
    record(lines, "Run", {k: metadata.get(k) for k in ("instance", "seed", "run_id", "packet_sha256")}, "metadata")
    record(lines, "Trigger", event.get("trigger"), "trigger")
    record(lines, "Trạng thái quyết định", event.get("decision_status"), "decision_status")
    record(lines, "Kiểm tra action", event.get("validation"), "validation")
    for key in ("global_search_state", "recent_quality", "recent_outcomes", "action_limits"):
        if key in state:
            record(lines, f"StatePacket: {key}", state[key], f"state.{key}")

    for i, raw_role in enumerate(roles):
        role = obj(raw_role)
        record(lines, f"Đầu ra vai trò {i + 1}",
               {k: role.get(k) for k in ("role", "status", "raw_decision")}, f"roles[{i}]")
    if not roles:
        lines.append("- **Đầu ra vai trò:** không có trong trace.")

    record(lines, "Action sau validation", event.get("action"), "action")
    if action.get("operator_id") == "no_op":
        proposed = any(obj(r).get("status") == "ok" and
                       obj(obj(r).get("raw_decision")).get("operator_id") == "no_op"
                       for r in roles)
        if proposed and event.get("validation") == "valid_no_op" and event.get("decision_status") == "decision_received":
            lines.append("- **Nguồn no_op:** được đề xuất trong đầu ra vai trò và hợp lệ.")
        else:
            lines.append("- **Nguồn no_op:** không xác nhận được đây là lựa chọn trực tiếp của LLM; kiểm tra decision_status và validation ở trên.")

    regions = seq(state.get("candidate_regions"))
    if "candidate_regions" in state and isinstance(state["candidate_regions"], list):
        alternatives = [{"target": obj(r).get("target"),
                         "operators": obj(obj(r).get("available_actions")).get("operators")}
                        for r in regions]
        record(lines, "Các region và operator được công bố", alternatives, "state.candidate_regions")
    else:
        lines.append("- **Các lựa chọn:** không có dữ liệu StatePacket trong trace.")

    region_id = action.get("region_id")
    if region_id is not None:
        matches = [(i, obj(r)) for i, r in enumerate(regions)
                   if obj(obj(r).get("target")).get("region_id") == region_id]
        if len(matches) == 1:
            i, region = matches[0]
            record(lines, "Region được chọn", region, f"state.candidate_regions[{i}]")
        else:
            lines.append("- **Region được chọn:** không tìm được duy nhất trong StatePacket đã lưu.")
        boundaries = seq(state.get("boundary_context"))
        matches = [(i, obj(b)) for i, b in enumerate(boundaries) if obj(b).get("region_id") == region_id]
        if len(matches) == 1:
            i, boundary = matches[0]
            record(lines, "Boundary của region", boundary, f"state.boundary_context[{i}]")

    if event.get("outcome") is None:
        lines.append("- **Kết quả operator:** không có trong trace; không suy ra thành công hoặc thất bại.")
    else:
        record(lines, "Kết quả operator", outcome, "outcome")
    record(lines, "Incumbent trước review", event.get("incumbent_before"), "incumbent_before")
    record(lines, "Incumbent sau review", event.get("incumbent_after"), "incumbent_after")
    lines.extend(["", "Chỉ mô tả dữ liệu đã ghi. Điểm trước/sau và việc action được chọn không chứng minh nguyên nhân; các lựa chọn khác chưa được thử. Không suy ra lý do nội bộ của LLM."])
    return "\n".join(lines)


def render_trace(path: Path) -> str:
    reviews = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON không hợp lệ tại dòng {line_number}: {exc.msg}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"Dòng {line_number} phải là một JSON object")
            reviews.append(render_event(event, line_number))
    if not reviews:
        raise ValueError("Trace không có review nào")
    return "\n\n".join(["# Giải thích các controller review", f"Nguồn: `{path}`. Mỗi mục trỏ đến dòng JSONL và trường tương ứng.", *reviews]) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Đường dẫn controller_trace.jsonl")
    args = parser.parse_args()
    try:
        result = render_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"Không đọc được trace: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
