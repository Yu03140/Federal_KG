"""
Neo4j 客户端封装

提供全局单例 driver、session 上下文、启动时约束/索引初始化。
替换原 Zep Cloud 作为知识图谱主存储。
"""

import logging
import threading
from contextlib import contextmanager
from typing import Optional

from neo4j import GraphDatabase, Driver, Session

from ..config import Config

logger = logging.getLogger(__name__)


_CONSTRAINTS = [
    "CREATE CONSTRAINT entity_uuid IF NOT EXISTS FOR (n:Entity) REQUIRE n.uuid IS UNIQUE",
    "CREATE CONSTRAINT graph_meta_id IF NOT EXISTS FOR (g:GraphMeta) REQUIRE g.graph_id IS UNIQUE",
]

_INDEXES = [
    "CREATE INDEX entity_graph_id IF NOT EXISTS FOR (n:Entity) ON (n.graph_id)",
    "CREATE INDEX entity_domain IF NOT EXISTS FOR (n:Entity) ON (n.domain)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.entity_type)",
]


class Neo4jClient:
    """进程级单例 Neo4j driver 持有者"""

    _instance: Optional["Neo4jClient"] = None
    _lock = threading.Lock()

    def __init__(self):
        uri = Config.NEO4J_URI
        user = Config.NEO4J_USER
        password = Config.NEO4J_PASSWORD
        if not password:
            raise ValueError("NEO4J_PASSWORD 未配置")

        self._database = Config.NEO4J_DATABASE or "neo4j"
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Neo4j driver connected: {uri} (db={self._database})")

    @classmethod
    def instance(cls) -> "Neo4jClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def driver(self) -> Driver:
        return self._driver

    @property
    def database(self) -> str:
        return self._database

    @contextmanager
    def session(self) -> Session:
        with self._driver.session(database=self._database) as session:
            yield session

    def verify_connectivity(self):
        self._driver.verify_connectivity()

    def ensure_schema(self):
        """幂等创建约束和索引，应用启动时调用一次。"""
        with self.session() as session:
            for stmt in _CONSTRAINTS + _INDEXES:
                session.run(stmt)
        logger.info("Neo4j schema (constraints + indexes) ensured")

    def close(self):
        if self._driver:
            self._driver.close()


def get_neo4j() -> Neo4jClient:
    return Neo4jClient.instance()
