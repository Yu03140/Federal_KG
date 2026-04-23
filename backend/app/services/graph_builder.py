"""
图谱构建服务（Neo4j 后端）

接口语义保持与原 Zep 版本一致：
- create_graph(name) -> graph_id
- set_ontology(graph_id, ontology)  # Neo4j 下为空操作，ontology 由 project.json 持有
- add_text_batches(graph_id, chunks, batch_size, progress_callback) -> List[str]
  在新实现中：每个 chunk 调用 LLM 抽取器，按 ontology 白名单过滤，再 MERGE 到 Neo4j
- _wait_for_episodes(...)  # Neo4j 写入是同步的，保留为近似 no-op
- get_graph_data(graph_id) -> {graph_id, nodes, edges, node_count, edge_count}
- delete_graph(graph_id)

多领域联邦准备：所有节点带 `graph_id` 和 `domain` 属性，方便后续跨域 Cypher 查询。
"""

import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..utils.locale import get_locale, set_locale, t
from .extractors import TripleExtractor
from .neo4j_client import get_neo4j
from .text_processor import TextProcessor

logger = logging.getLogger(__name__)


@dataclass
class GraphInfo:
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _safe_rel_type(name: str, whitelist: set) -> Optional[str]:
    """只允许 ontology 白名单里的关系类型，且必须是 UPPER_SNAKE_CASE。"""
    if not name or name not in whitelist:
        return None
    if not _REL_TYPE_RE.match(name):
        return None
    return name


class GraphBuilderService:
    """知识图谱构建服务（Neo4j 实现）"""

    def __init__(self, api_key: Optional[str] = None):
        # api_key 参数保留以兼容旧调用方，Neo4j 场景下不使用
        self.neo4j = get_neo4j()
        self.task_manager = TaskManager()
        self.extractor = TripleExtractor()

    # ------------------------------------------------------------------
    # 异步入口（保留，历史上被直接调用）
    # ------------------------------------------------------------------
    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "Federal KG Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
        domain: str = "general",
    ) -> str:
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
                "domain": domain,
            },
        )
        current_locale = get_locale()
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size, domain, current_locale),
        )
        thread.daemon = True
        thread.start()
        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        domain: str,
        locale: str = "zh",
    ):
        set_locale(locale)
        try:
            self.task_manager.update_task(
                task_id, status=TaskStatus.PROCESSING, progress=5, message=t("progress.startBuildingGraph")
            )
            graph_id = self.create_graph(graph_name, domain=domain)
            self.task_manager.update_task(task_id, progress=10, message=t("progress.graphCreated", graphId=graph_id))

            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(task_id, progress=15, message=t("progress.ontologySet"))

            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(task_id, progress=20, message=t("progress.textSplit", count=total_chunks))

            def cb(msg, prog):
                self.task_manager.update_task(task_id, progress=20 + int(prog * 70), message=msg)

            self.add_text_batches(graph_id, chunks, batch_size, cb, ontology=ontology, domain=domain)

            self.task_manager.update_task(task_id, progress=95, message=t("progress.fetchingGraphInfo"))
            info = self._get_graph_info(graph_id)

            self.task_manager.complete_task(
                task_id,
                {"graph_id": graph_id, "graph_info": info.to_dict(), "chunks_processed": total_chunks},
            )
        except Exception as e:
            import traceback
            self.task_manager.fail_task(task_id, f"{e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # 公开接口（被 api/graph.py 直接调用）
    # ------------------------------------------------------------------
    def create_graph(self, name: str, domain: str = "general") -> str:
        graph_id = f"fkg_{uuid.uuid4().hex[:16]}"
        with self.neo4j.session() as session:
            session.run(
                """
                MERGE (g:GraphMeta {graph_id: $gid})
                SET g.name = $name,
                    g.domain = $domain,
                    g.created_at = coalesce(g.created_at, timestamp())
                """,
                gid=graph_id, name=name, domain=domain,
            )
        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """把 ontology 镜像到 GraphMeta 节点，便于 Neo4j 端溯源。权威拷贝仍在 project.json。"""
        import json as _json
        with self.neo4j.session() as session:
            session.run(
                "MATCH (g:GraphMeta {graph_id: $gid}) SET g.ontology = $ontology",
                gid=graph_id, ontology=_json.dumps(ontology, ensure_ascii=False),
            )

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
        ontology: Optional[Dict[str, Any]] = None,
        domain: str = "general",
        max_workers: int = 5,
    ) -> List[str]:
        """并发抽取三元组并写入 Neo4j。返回每个 chunk 的占位 id（兼容旧签名）。"""
        if ontology is None:
            ontology = self._load_ontology(graph_id)

        edge_whitelist = {e.get("name") for e in ontology.get("edge_types", []) if e.get("name")}
        total = len(chunks)
        completed = 0
        lock = threading.Lock()

        def extract_one(idx: int, chunk: str):
            try:
                return idx, self.extractor.extract(chunk, ontology)
            except Exception as e:
                logger.warning(f"抽取 chunk {idx} 失败：{e}")
                return idx, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_one, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                idx, result = future.result()
                with lock:
                    completed += 1
                    if progress_callback:
                        progress_callback(
                            t("progress.sendingBatch", current=completed, total=total, chunks=1),
                            completed / max(total, 1),
                        )
                if result and not result.is_empty():
                    self._write_to_neo4j(graph_id, domain, result, edge_whitelist)

        return [f"chunk_{i}" for i in range(total)]

    def _wait_for_episodes(self, episode_uuids, progress_callback=None, timeout: int = 600):
        """Neo4j 写入同步完成；此方法仅为兼容旧调用保留。"""
        if progress_callback:
            progress_callback(t("progress.processingComplete", completed=len(episode_uuids or []), total=len(episode_uuids or [])), 1.0)

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        with self.neo4j.session() as session:
            node_records = session.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN n.uuid AS uuid, n.name AS name, n.entity_type AS entity_type,
                       n.domain AS domain, n.created_at AS created_at,
                       properties(n) AS props
                """,
                gid=graph_id,
            ).data()

            edge_records = session.run(
                """
                MATCH (a:Entity {graph_id: $gid})-[r]->(b:Entity {graph_id: $gid})
                RETURN r.uuid AS uuid, type(r) AS rel_type, r.evidence AS evidence,
                       r.created_at AS created_at,
                       a.uuid AS source_uuid, a.name AS source_name,
                       b.uuid AS target_uuid, b.name AS target_name,
                       properties(r) AS props
                """,
                gid=graph_id,
            ).data()

        reserved = {"uuid", "name", "entity_type", "graph_id", "domain", "created_at"}
        nodes_data = []
        for r in node_records:
            props = r.get("props") or {}
            attributes = {k: v for k, v in props.items() if k not in reserved}
            nodes_data.append({
                "uuid": r["uuid"],
                "name": r["name"] or "",
                "labels": [r.get("entity_type")] if r.get("entity_type") else [],
                "summary": "",
                "attributes": attributes,
                "created_at": str(r["created_at"]) if r.get("created_at") else None,
            })

        rel_reserved = {"uuid", "evidence", "created_at"}
        edges_data = []
        for r in edge_records:
            props = r.get("props") or {}
            attributes = {k: v for k, v in props.items() if k not in rel_reserved}
            edges_data.append({
                "uuid": r.get("uuid"),
                "name": r.get("rel_type") or "",
                "fact": r.get("evidence") or "",
                "fact_type": r.get("rel_type") or "",
                "source_node_uuid": r.get("source_uuid"),
                "target_node_uuid": r.get("target_uuid"),
                "source_node_name": r.get("source_name") or "",
                "target_node_name": r.get("target_name") or "",
                "attributes": attributes,
                "created_at": str(r["created_at"]) if r.get("created_at") else None,
                "valid_at": None,
                "invalid_at": None,
                "expired_at": None,
                "episodes": [],
            })

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        with self.neo4j.session() as session:
            session.run("MATCH (n:Entity {graph_id: $gid}) DETACH DELETE n", gid=graph_id)
            session.run("MATCH (g:GraphMeta {graph_id: $gid}) DETACH DELETE g", gid=graph_id)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _load_ontology(self, graph_id: str) -> Dict[str, Any]:
        import json as _json
        with self.neo4j.session() as session:
            rec = session.run(
                "MATCH (g:GraphMeta {graph_id: $gid}) RETURN g.ontology AS ontology",
                gid=graph_id,
            ).single()
        if rec and rec["ontology"]:
            try:
                return _json.loads(rec["ontology"])
            except Exception:
                pass
        return {"entity_types": [], "edge_types": []}

    def _write_to_neo4j(
        self,
        graph_id: str,
        domain: str,
        result,
        edge_whitelist: set,
        source_evidence_chunk: str = "",
    ):
        entities_payload: List[Dict[str, Any]] = []
        for e in result.entities:
            entities_payload.append({
                "name": e.name,
                "entity_type": e.entity_type,
                "uuid": uuid.uuid4().hex,
                "attributes": e.attributes or {},
            })

        triples_by_relation: Dict[str, List[Dict[str, Any]]] = {}
        for tri in result.triples:
            rel = _safe_rel_type(tri.relation, edge_whitelist)
            if not rel:
                continue
            triples_by_relation.setdefault(rel, []).append({
                "src_name": tri.source_name,
                "src_type": tri.source_type,
                "tgt_name": tri.target_name,
                "tgt_type": tri.target_type,
                "evidence": tri.evidence or "",
                "attributes": tri.attributes or {},
                "uuid": uuid.uuid4().hex,
            })

        with self.neo4j.session() as session:
            if entities_payload:
                session.run(
                    """
                    UNWIND $batch AS e
                    MERGE (n:Entity {graph_id: $gid, name: e.name, entity_type: e.entity_type})
                    ON CREATE SET
                        n.uuid = e.uuid,
                        n.domain = $domain,
                        n.created_at = timestamp()
                    SET n += e.attributes
                    """,
                    batch=entities_payload, gid=graph_id, domain=domain,
                )

            for rel, triples in triples_by_relation.items():
                session.run(
                    f"""
                    UNWIND $batch AS t
                    MATCH (a:Entity {{graph_id: $gid, name: t.src_name, entity_type: t.src_type}})
                    MATCH (b:Entity {{graph_id: $gid, name: t.tgt_name, entity_type: t.tgt_type}})
                    MERGE (a)-[r:`{rel}`]->(b)
                    ON CREATE SET
                        r.uuid = t.uuid,
                        r.created_at = timestamp(),
                        r.evidence = t.evidence
                    SET r += t.attributes
                    """,
                    batch=triples, gid=graph_id,
                )

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        with self.neo4j.session() as session:
            node_stats = session.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN count(n) AS cnt, collect(DISTINCT n.entity_type) AS types
                """,
                gid=graph_id,
            ).single()
            edge_count = session.run(
                """
                MATCH (:Entity {graph_id: $gid})-[r]->(:Entity {graph_id: $gid})
                RETURN count(r) AS cnt
                """,
                gid=graph_id,
            ).single()["cnt"]

        return GraphInfo(
            graph_id=graph_id,
            node_count=node_stats["cnt"] if node_stats else 0,
            edge_count=edge_count or 0,
            entity_types=[t for t in (node_stats["types"] if node_stats else []) if t],
        )
