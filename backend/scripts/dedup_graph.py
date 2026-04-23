"""
去重 CLI：合并同一图谱内"名字相似 + 类型相同"的节点。

用法：
    python -m backend.scripts.dedup_graph <graph_id> [--threshold 0.88] [--dry-run]

示例：
    # 预览（不动数据）
    python -m backend.scripts.dedup_graph fkg_abc123 --dry-run

    # 实际执行
    python -m backend.scripts.dedup_graph fkg_abc123 --threshold 0.9
"""

import argparse
import json
import sys
from pathlib import Path

# 兼容直接 `python backend/scripts/dedup_graph.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.services.graph_dedup import GraphDedupService  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="合并图谱内的重复节点")
    parser.add_argument("graph_id", help="目标图谱 ID，如 fkg_xxxxxxxxxxxx")
    parser.add_argument("--threshold", type=float, default=0.88,
                        help="名字相似度阈值，0-1，越高越严格（默认 0.88）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印合并计划，不修改数据库")
    args = parser.parse_args()

    svc = GraphDedupService()
    report = svc.run(args.graph_id, threshold=args.threshold, dry_run=args.dry_run)

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
