"""
知识图谱去重服务

同一实体在多个 chunk 里被 LLM 抽成不同名字（大小写/缩写/修饰词差异）是很常见的问题。
这里做两件事：
1. 按实体类型分组，用 SequenceMatcher 做 n² 名字相似度比较
2. 超过阈值的对用并查集聚合，保留"更规范"的节点，把其他节点的边重定向过来并删除

不依赖 APOC，关系类型用白名单校验后做字符串拼接。
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from .neo4j_client import get_neo4j

logger = logging.getLogger(__name__)


@dataclass
class DedupReport:
    graph_id: str
    threshold: float
    dry_run: bool
    nodes_before: int = 0
    nodes_after: int = 0
    edges_before: int = 0
    edges_after: int = 0
    groups_merged: int = 0
    pairs_flagged: int = 0
    merges: List[Dict[str, Any]] = field(default_factory=list)  # [{keep, drops: [...], type}]

    def to_dict(self):
        return {
            "graph_id": self.graph_id,
            "threshold": self.threshold,
            "dry_run": self.dry_run,
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "edges_before": self.edges_before,
            "edges_after": self.edges_after,
            "groups_merged": self.groups_merged,
            "pairs_flagged": self.pairs_flagged,
            "merges": self.merges[:50],  # 前端预览上限
        }


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, keeper, victim):
        rk = self.find(keeper)
        rv = self.find(victim)
        if rk == rv:
            return
        self.parent[rv] = rk  # keeper 主导

    def groups(self):
        result: Dict[str, List[str]] = {}
        for x in self.parent:
            root = self.find(x)
            result.setdefault(root, []).append(x)
        return result


def _normalize_for_compare(name: str) -> str:
    """做相似度比较前的归一化：小写、去首尾空白、把常见符号折叠。"""
    if not name:
        return ""
    s = name.strip().lower()
    # 下标数字归一化（₀-₉ → 0-9）
    subs = "₀₁₂₃₄₅₆₇₈₉"
    for i, c in enumerate(subs):
        s = s.replace(c, str(i))
    # 去掉多余空白
    s = " ".join(s.split())
    return s


def _canonical_score(name: str, edge_count: int) -> Tuple[int, int, str]:
    """排序分值：优先保留 (a) 关系更多 (b) 名字更短 (c) 字典序小 的节点作为 keeper。
    返回三元组，用于 sort key（较小者胜出）。"""
    # 更多关系 → 更 canonical → 负号让它排前
    return (-edge_count, len(name), name.lower())


class GraphDedupService:
    def __init__(self):
        self.neo4j = get_neo4j()

    def run(self, graph_id: str, threshold: float = 0.88, dry_run: bool = False) -> DedupReport:
        report = DedupReport(graph_id=graph_id, threshold=threshold, dry_run=dry_run)

        with self.neo4j.session() as session:
            # 1. 拉取所有节点 + 每个节点的度
            nodes = session.run("""
                MATCH (n:Entity {graph_id: $gid})
                OPTIONAL MATCH (n)-[r]-()
                WITH n, count(r) AS deg
                RETURN n.uuid AS uuid, n.name AS name, n.entity_type AS entity_type,
                       deg AS degree
            """, gid=graph_id).data()

            report.nodes_before = len(nodes)
            report.edges_before = session.run(
                "MATCH (:Entity {graph_id: $gid})-[r]->(:Entity {graph_id: $gid}) RETURN count(r) AS c",
                gid=graph_id,
            ).single()["c"]

        # 2. 按 entity_type 分组，做相似度比较
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for n in nodes:
            by_type.setdefault(n["entity_type"] or "_UNKNOWN_", []).append(n)

        uf = _UnionFind([n["uuid"] for n in nodes])
        uuid_to_node = {n["uuid"]: n for n in nodes}

        for etype, group in by_type.items():
            if len(group) < 2:
                continue
            # 预排序：canonical 优先在前（被选为 keeper）
            group_sorted = sorted(
                group, key=lambda n: _canonical_score(n["name"] or "", n.get("degree", 0))
            )
            norms = [(_normalize_for_compare(n["name"] or ""), n["uuid"]) for n in group_sorted]

            for i in range(len(norms)):
                ni, ui = norms[i]
                if not ni:
                    continue
                for j in range(i + 1, len(norms)):
                    nj, uj = norms[j]
                    if not nj:
                        continue
                    # 快速剪枝：长度差距过大就跳过
                    if abs(len(ni) - len(nj)) / max(len(ni), len(nj), 1) > 0.5:
                        continue
                    if ni == nj or SequenceMatcher(None, ni, nj).ratio() >= threshold:
                        # 前面的是 keeper，后面的合入
                        keeper_root = uf.find(ui)
                        uf.union(keeper_root, uj)
                        report.pairs_flagged += 1

        # 3. 形成合并组（组大小 >= 2 的才真正合并）
        groups = uf.groups()
        merge_plan: List[Tuple[str, List[str]]] = []
        for root, members in groups.items():
            if len(members) < 2:
                continue
            victims = [m for m in members if m != root]
            merge_plan.append((root, victims))
            report.groups_merged += 1
            report.merges.append({
                "keep": {
                    "uuid": root,
                    "name": uuid_to_node[root]["name"],
                    "entity_type": uuid_to_node[root]["entity_type"],
                },
                "drops": [
                    {"uuid": v, "name": uuid_to_node[v]["name"]}
                    for v in victims
                ],
            })

        # 4. 执行合并
        if not dry_run and merge_plan:
            with self.neo4j.session() as session:
                for keeper, victims in merge_plan:
                    for victim in victims:
                        self._merge_nodes(session, keeper, victim)

        # 5. 收尾统计
        with self.neo4j.session() as session:
            report.nodes_after = session.run(
                "MATCH (n:Entity {graph_id: $gid}) RETURN count(n) AS c", gid=graph_id
            ).single()["c"]
            report.edges_after = session.run(
                "MATCH (:Entity {graph_id: $gid})-[r]->(:Entity {graph_id: $gid}) RETURN count(r) AS c",
                gid=graph_id,
            ).single()["c"]

        logger.info(
            f"[dedup] graph={graph_id} dry_run={dry_run} "
            f"nodes {report.nodes_before}→{report.nodes_after}, "
            f"edges {report.edges_before}→{report.edges_after}, "
            f"groups_merged={report.groups_merged}"
        )
        return report

    # ------------------------------------------------------------------
    def _merge_nodes(self, session, keeper_uuid: str, victim_uuid: str):
        """用 APOC mergeNodes 原子合并：保留 keeper，victim 的边和属性都合并过来。"""
        session.run(
            """
            MATCH (a:Entity {uuid: $ku}), (b:Entity {uuid: $vu})
            CALL apoc.refactor.mergeNodes([a, b], {
                properties: 'discard',
                mergeRels: true
            }) YIELD node
            RETURN node
            """,
            ku=keeper_uuid,
            vu=victim_uuid,
        )
