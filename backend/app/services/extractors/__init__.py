"""
Federal_KG LLM 抽取器模块

把文本 chunk + 用户确认过的 ontology 输入 LLM，
输出结构化三元组，用于 Neo4j 建图。
"""

from .triple_extractor import TripleExtractor, ExtractionResult

__all__ = ["TripleExtractor", "ExtractionResult"]
