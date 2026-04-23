"""
将预抽取的 JSON 知识库（如 haiku_knowledge_base.json）转换为 Markdown，
便于直接通过 Federal_KG 的 /api/graph/ontology/generate 上传。

用法：
    python scripts/convert_kb_to_md.py data/haiku_knowledge_base.json \
        --output data/haiku_knowledge_base.md

约定：输入 JSON 顶层为 {paper_id: {field: value, ...}} 字典；每个 value
可以是 str / list[str] / dict。脚本不做字段重命名，按原字段名作为 Markdown
小节标题输出，方便 schema 抽取看到结构化提示。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return "\n".join(f"- {v}" for v in value if v)
        return "\n".join(f"- {json.dumps(v, ensure_ascii=False)}" for v in value)
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            rendered = render_value(v)
            if not rendered:
                continue
            if "\n" in rendered:
                lines.append(f"- **{k}**:\n{rendered}")
            else:
                lines.append(f"- **{k}**: {rendered}")
        return "\n".join(lines)
    return str(value)


def render_paper(paper_id: str, paper: dict) -> str:
    title = paper.get("paper_title") or paper.get("title") or paper_id
    if isinstance(title, list):
        title = title[0] if title else paper_id

    lines = [f"## {paper_id}  {title}", ""]
    for field, value in paper.items():
        if field in ("paper_title", "paper_id"):
            continue
        rendered = render_value(value)
        if not rendered:
            continue
        lines.append(f"### {field}")
        lines.append("")
        lines.append(rendered)
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="输入 JSON 文件路径")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="输出 MD 路径，默认与输入同名但扩展名为 .md"
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    output = args.output or args.input.with_suffix(".md")

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("Expected top-level dict keyed by paper_id", file=sys.stderr)
        return 1

    paper_ids = sorted(data.keys())
    blocks = [
        f"# Knowledge Base ({args.input.name})",
        "",
        f"Source: `{args.input.name}` — {len(paper_ids)} papers",
        "",
    ]
    for pid in paper_ids:
        paper = data[pid]
        if not isinstance(paper, dict):
            continue
        blocks.append(render_paper(pid, paper))
        blocks.append("")

    output.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size / 1024:.1f} KB, {len(paper_ids)} papers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
