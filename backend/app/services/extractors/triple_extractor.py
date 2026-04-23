"""
LLM 三元组抽取器

输入：文本 chunk + 已确认的 ontology（entity_types / edge_types）
输出：结构化实体和三元组列表，严格遵循 ontology 类型白名单。

失败时返回空结果而不是抛错，让上层建图流程能够继续处理剩余 chunk。
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedTriple:
    source_name: str
    source_type: str
    relation: str
    target_name: str
    target_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence: str = ""


@dataclass
class ExtractionResult:
    entities: List[ExtractedEntity] = field(default_factory=list)
    triples: List[ExtractedTriple] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.entities and not self.triples


_SYSTEM_PROMPT = """你是一个科学文献知识图谱抽取器，专注物质科学领域（材料、化学、物理、航天等）。

你的任务：从给定文本片段中，严格按照给定的 ontology（实体类型白名单 + 关系类型白名单）抽取实体和三元组。

**硬约束 — 违反即视为错误输出：**
1. 实体类型必须是 ontology.entity_types 中的 name 之一，不允许任何自由发挥
2. 关系类型必须是 ontology.edge_types 中的 name 之一
3. 三元组的 (source_type, relation, target_type) 必须匹配该 relation 的 source_targets 之一
4. 实体属性只能使用该类型在 ontology 中声明过的属性名；未出现在文本中的属性直接省略，不要编造
5. 每条三元组必须给出 `evidence`（文本中最简短的原始证据片段，1-2 句）
6. 如果文本与 ontology 领域无关或没有可靠信息，返回空列表 `{"entities": [], "triples": []}`
7. 输出必须是有效 JSON，不要任何其他文字

**实体 name 规范化（硬性要求 — 跨 chunk 必须保持一致，否则会产生重复节点）：**

1. **化学物质优先使用分子式**：
   - ✓ `Li₂O₂`、`TiO₂`、`CH₃NH₃PbI₃`、`Al₂O₃`、`MoS₂`
   - ✗ `lithium peroxide`、`titanium dioxide`、`methylammonium lead iodide`

2. **无分子式时使用 IUPAC 标准名或最规范的单一英文名**（除非原文是中文领域术语）

3. **缩写必须展开为全称**：
   - `LIB` → `Lithium-ion battery`
   - `PV` → `Photovoltaic`
   - `SEI` → `Solid electrolyte interphase`
   - `OER` → `Oxygen evolution reaction`
   - `DFT` → `Density functional theory`

4. **去掉形容词、修饰语、限定词**（修饰信息应进 attributes，而非 name）：
   - ✓ `Perovskite`（type=Material）
   - ✗ `halide perovskite`、`lead halide perovskite`、`mixed-cation perovskite`

5. **不在 name 中嵌入数值/浓度/单位**：
   - ✓ `TiO₂` + attribute `concentration="5 wt%"`
   - ✗ `5 wt% TiO₂`、`0.1 M H₂SO₄`

6. **同一实体在该文本片段中多次出现时，必须始终使用完全相同的 name 字符串**（大小写、空格、下标符号都要一致）

7. **保持原文语言**：原文是中文术语则保留中文；不强制翻译

**输出 JSON schema：**
```json
{
  "entities": [
    {"name": "string", "type": "<EntityType>", "attributes": {"<attr>": "<value>"}}
  ],
  "triples": [
    {
      "source_name": "string", "source_type": "<EntityType>",
      "relation": "<EdgeType>",
      "target_name": "string", "target_type": "<EntityType>",
      "attributes": {},
      "evidence": "原文片段"
    }
  ]
}
```
"""


class TripleExtractor:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def extract(self, chunk: str, ontology: Dict[str, Any]) -> ExtractionResult:
        if not chunk or not chunk.strip():
            return ExtractionResult()

        user_prompt = self._build_user_prompt(chunk, ontology)
        try:
            raw = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
        except Exception as e:
            logger.warning(f"TripleExtractor LLM 调用失败，跳过该 chunk：{e}")
            return ExtractionResult()

        return self._validate_and_parse(raw, ontology)

    @staticmethod
    def _build_user_prompt(chunk: str, ontology: Dict[str, Any]) -> str:
        compact_ontology = {
            "entity_types": [
                {
                    "name": et.get("name"),
                    "description": et.get("description", ""),
                    "attributes": [a.get("name") for a in et.get("attributes", [])],
                }
                for et in ontology.get("entity_types", [])
            ],
            "edge_types": [
                {
                    "name": ed.get("name"),
                    "description": ed.get("description", ""),
                    "source_targets": ed.get("source_targets", []),
                }
                for ed in ontology.get("edge_types", [])
            ],
        }
        return (
            "**Ontology（严格遵守）：**\n"
            f"```json\n{json.dumps(compact_ontology, ensure_ascii=False, indent=2)}\n```\n\n"
            "**文本片段：**\n"
            f"```\n{chunk}\n```\n\n"
            "现在输出 JSON。"
        )

    @staticmethod
    def _validate_and_parse(raw: Dict[str, Any], ontology: Dict[str, Any]) -> ExtractionResult:
        entity_type_set = {et["name"] for et in ontology.get("entity_types", []) if et.get("name")}
        edge_index: Dict[str, List[Dict[str, str]]] = {}
        for ed in ontology.get("edge_types", []):
            name = ed.get("name")
            if name:
                edge_index[name] = ed.get("source_targets", []) or []

        allowed_attrs: Dict[str, set] = {
            et["name"]: {a.get("name") for a in et.get("attributes", []) if a.get("name")}
            for et in ontology.get("entity_types", [])
            if et.get("name")
        }

        result = ExtractionResult()

        for e in raw.get("entities", []) or []:
            name = (e.get("name") or "").strip()
            etype = (e.get("type") or "").strip()
            if not name or etype not in entity_type_set:
                continue
            attrs = e.get("attributes") or {}
            clean_attrs = {
                k: v for k, v in attrs.items()
                if isinstance(k, str) and k in allowed_attrs.get(etype, set()) and v not in (None, "")
            }
            result.entities.append(ExtractedEntity(name=name, entity_type=etype, attributes=clean_attrs))

        for t in raw.get("triples", []) or []:
            relation = (t.get("relation") or "").strip()
            if relation not in edge_index:
                continue
            src_name = (t.get("source_name") or "").strip()
            src_type = (t.get("source_type") or "").strip()
            tgt_name = (t.get("target_name") or "").strip()
            tgt_type = (t.get("target_type") or "").strip()
            if not (src_name and tgt_name and src_type in entity_type_set and tgt_type in entity_type_set):
                continue

            pairs = edge_index[relation]
            if pairs and not any(
                p.get("source") == src_type and p.get("target") == tgt_type for p in pairs
            ):
                continue

            result.triples.append(ExtractedTriple(
                source_name=src_name,
                source_type=src_type,
                relation=relation,
                target_name=tgt_name,
                target_type=tgt_type,
                attributes=t.get("attributes") or {},
                evidence=(t.get("evidence") or "")[:500],
            ))

        return result
