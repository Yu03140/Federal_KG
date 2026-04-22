"""
本体生成服务
接口1：分析科研文献内容，生成适合物质科学垂直知识图谱的实体和关系类型定义
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction

logger = logging.getLogger(__name__)


def _to_pascal_case(name: str) -> str:
    """将任意格式的名称转换为 PascalCase（如 'works_for' -> 'WorksFor', 'person' -> 'Person'）"""
    # 按非字母数字字符分割
    parts = re.split(r'[^a-zA-Z0-9]+', name)
    # 再按 camelCase 边界分割（如 'camelCase' -> ['camel', 'Case']）
    words = []
    for part in parts:
        words.extend(re.sub(r'([a-z])([A-Z])', r'\1_\2', part).split('_'))
    # 每个词首字母大写，过滤空串
    result = ''.join(word.capitalize() for word in words if word)
    return result if result else 'Unknown'


# 本体生成的系统提示词
ONTOLOGY_SYSTEM_PROMPT = """你是一个专业的知识图谱本体设计专家，专注于**物质科学领域**（包含材料、化学、物理、航天等学科）。你的任务是分析给定的科研文献内容和研究目标，设计适合**构建物质科学垂直领域知识图谱**的实体类型和关系类型。

**重要：你必须输出有效的JSON格式数据，不要输出任何其他内容。**

## 核心任务背景

我们正在构建一个面向**物质科学的垂直领域知识图谱**，目标是支持科研人员进行：
- 跨文献、跨学科的知识整合
- 功能性路径（如"某属性 → 结构 → 合成方法"）的推理式检索
- 隐性关联的发现（例如"某元素掺杂对某属性的影响"）

因此，**实体应当是科研语境中具有明确语义的对象、方法、过程、性质或概念**：

**应当包含**：
- 具体物质（化合物、晶体、聚合物、合金、半导体、溶剂、试剂等）
- 结构与组分（晶体结构、相、微结构、官能团、掺杂剂等）
- 性质与参数（物理/化学/力学/光电性质，及其量化指标）
- 过程与方法（合成路线、制备工艺、表征手段、计算/仿真方法）
- 装置与系统（反应装置、器件、样品、实验平台）
- 应用领域（能源、催化、电子、生物医学、航天等）
- 抽象科学概念（理论模型、现象、机制、原理）

**应当避免**：
- 把文献作者、机构、期刊作为核心实体（Phase 1 不建模引用/社会关系）
- 过于宽泛的无界概念（如"科学"、"研究"）
- 把数值本身作为实体（数值应作为属性）

## 输出格式

请输出JSON格式，包含以下结构：

```json
{
    "entity_types": [
        {
            "name": "实体类型名称（英文，PascalCase）",
            "description": "简短描述（英文，不超过100字符）",
            "attributes": [
                {
                    "name": "属性名（英文，snake_case）",
                    "type": "text",
                    "description": "属性描述"
                }
            ],
            "examples": ["示例实体1", "示例实体2"]
        }
    ],
    "edge_types": [
        {
            "name": "关系类型名称（英文，UPPER_SNAKE_CASE）",
            "description": "简短描述（英文，不超过100字符）",
            "source_targets": [
                {"source": "源实体类型", "target": "目标实体类型"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "对文献内容的简要分析说明（中文或英文，按语言指令）"
}
```

## 设计指南（极其重要！）

### 1. 实体类型设计 - 必须严格遵守

**数量要求：必须正好10个实体类型**

**层次结构要求（必须同时包含具体类型和兜底类型）**：

你的10个实体类型必须包含以下层次：

A. **兜底类型（必须包含，放在列表最后2个）**：
   - `Material`: 任意未归入更具体物质子类的材料/物质实体兜底类型（如化合物、晶体、薄膜、粉末、液相等通用材料对象）。
   - `Concept`: 抽象科学概念的兜底类型（理论、机制、现象、原理、方法学等不属于具体物质或装置的对象）。

B. **具体类型（8个，根据文献内容设计）**：
   - 从文献中识别本领域核心对象，设计领域相关且边界清晰的类型
   - 例：钙钛矿太阳能电池相关文献可能有 `Perovskite`, `Dopant`, `PhotovoltaicProperty`, `DepositionMethod`, `SolarCellDevice`, `CharacterizationMethod`
   - 例：催化相关文献可能有 `Catalyst`, `Reactant`, `CatalyticReaction`, `ActiveSite`, `Selectivity`
   - 例：航天材料相关文献可能有 `Alloy`, `ThermalProperty`, `MechanicalTest`, `SpacecraftComponent`

**为什么需要兜底类型**：
- 文献中会出现大量未被前 8 个具体类型完全覆盖的对象（例如"某无名中间体"、"某通用衬底"），归入 `Material`
- 类似的现象、定律、机制（例如"Schottky 势垒"、"自由基聚合机制"）归入 `Concept`

**具体类型的设计原则**：
- 从文本中识别出高频出现或关键的科研对象
- 每个具体类型应有明确的边界，避免与其它类型重叠
- description 应清晰说明该类型与兜底类型 (`Material` / `Concept`) 的区别

### 2. 关系类型设计

- 数量：6-10个
- 关系应反映科研语义（组分构成、表征、合成、性质表达、应用等），**不应**是社交互动类关系（如评论、回应）
- 确保关系的 source_targets 涵盖你定义的实体类型；优先构造能形成"功能性路径"的关系链（如 Material → EXHIBITS_PROPERTY → Property、Material → SYNTHESIZED_BY → Method）

### 3. 属性设计

- 每个实体类型1-3个关键属性
- **注意**：属性名不能使用 `name`、`uuid`、`group_id`、`created_at`、`summary`（这些是 Zep 系统保留字）
- 推荐示例：`formula`, `material_type`, `crystal_system`, `band_gap`, `synthesis_route`, `measurement_unit`, `value_range`, `concept_type`, `definition` 等

## 实体类型参考（候选池，按文献实际内容取舍）

**物质/结构类（具体）**：
- Compound: 化学化合物
- CrystalStructure: 晶体结构 / 相
- Polymer: 聚合物
- Alloy: 合金
- ThinFilm: 薄膜
- Nanomaterial: 纳米材料
- Dopant: 掺杂剂
- Reagent: 试剂 / 前驱体

**物质类（兜底）**：
- Material: 任意材料/物质（不属于上述具体类型时使用）

**性质类（具体）**：
- PhysicalProperty: 物理性质（含力学、电学、光学、热学等可量化属性）
- ChemicalProperty: 化学性质（反应性、稳定性等）

**过程 / 方法 / 装置类（具体）**：
- SynthesisMethod: 合成方法 / 制备工艺
- CharacterizationMethod: 表征手段
- SimulationMethod: 理论/计算/仿真方法
- Device: 器件 / 装置
- Experiment: 实验流程

**抽象概念类（兜底）**：
- Concept: 理论、机制、现象、原理、方法学（不属于上述具体类型时使用）

**应用类（具体）**：
- ApplicationDomain: 应用领域

## 关系类型参考

- COMPOSED_OF: 组成 / 含有（组分关系）
- EXHIBITS_PROPERTY: 表现出某种性质
- SYNTHESIZED_BY: 通过某方法合成/制备
- CHARACTERIZED_BY: 通过某手段表征
- SIMULATED_BY: 通过某方法计算/仿真
- DOPED_WITH: 掺杂有
- DERIVED_FROM: 衍生于 / 源于
- APPLIED_IN: 应用于
- MEASURED_IN: 在某装置/实验中测量
- GOVERNED_BY: 受某原理/机制支配
- INTERACTS_WITH: 与...相互作用
- PART_OF: 属于 / 构成
"""


class OntologyGenerator:
    """
    本体生成器
    分析文本内容，生成实体和关系类型定义
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成本体定义

        Args:
            document_texts: 文档文本列表
            simulation_requirement: 研究目标 / 领域说明（字段名沿用自旧版社媒模拟，
                当前语义为"用户提供的研究目标、领域范围或建模重点说明"）
            additional_context: 额外上下文

        Returns:
            本体定义（entity_types, edge_types等）
        """
        # 构建用户消息
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        lang_instruction = get_language_instruction()
        system_prompt = f"{ONTOLOGY_SYSTEM_PROMPT}\n\n{lang_instruction}\nIMPORTANT: Entity type names MUST be in English PascalCase (e.g., 'CrystalStructure', 'CharacterizationMethod'). Relationship type names MUST be in English UPPER_SNAKE_CASE (e.g., 'EXHIBITS_PROPERTY', 'SYNTHESIZED_BY'). Attribute names MUST be in English snake_case. Only description fields and analysis_summary should use the specified language above."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 调用LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # 验证和后处理
        result = self.validate_and_process(result)

        return result
    
    # 传给 LLM 的文本最大长度（5万字）
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """构建用户消息"""
        
        # 合并文本
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # 如果文本超过5万字，截断（仅影响传给LLM的内容，不影响图谱构建）
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(原文共{original_length}字，已截取前{self.MAX_TEXT_LENGTH_FOR_LLM}字用于本体分析)..."
        
        message = f"""## 研究目标 / 领域说明

{simulation_requirement}

## 文献内容

{combined_text}
"""

        if additional_context:
            message += f"""
## 额外说明

{additional_context}
"""

        message += """
请根据以上内容，设计适合**物质科学垂直知识图谱**的实体类型和关系类型。

**必须遵守的规则**：
1. 必须正好输出10个实体类型
2. 最后2个必须是兜底类型：Material（物质兜底）和 Concept（抽象概念/方法/现象兜底）
3. 前8个是根据文献内容设计的具体领域类型
4. 实体应聚焦于科研语义对象（物质、结构、性质、过程、方法、装置、应用、概念），避免把作者/机构/期刊作为核心实体
5. 属性名不能使用 name、uuid、group_id、created_at、summary 等 Zep 保留字；推荐用 formula、material_type、value_range、concept_type、definition 等
"""

        return message
    
    def validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证和后处理本体结果（对 LLM 输出或用户编辑后的 ontology 均适用）"""
        
        # 确保必要字段存在
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # 验证实体类型
        # 记录原始名称到 PascalCase 的映射，用于后续修正 edge 的 source_targets 引用
        entity_name_map = {}
        for entity in result["entity_types"]:
            # 强制将 entity name 转为 PascalCase（Zep API 要求）
            if "name" in entity:
                original_name = entity["name"]
                entity["name"] = _to_pascal_case(original_name)
                if entity["name"] != original_name:
                    logger.warning(f"Entity type name '{original_name}' auto-converted to '{entity['name']}'")
                entity_name_map[original_name] = entity["name"]
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # 确保description不超过100字符
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # 验证关系类型
        for edge in result["edge_types"]:
            # 强制将 edge name 转为 SCREAMING_SNAKE_CASE（Zep API 要求）
            if "name" in edge:
                original_name = edge["name"]
                edge["name"] = original_name.upper()
                if edge["name"] != original_name:
                    logger.warning(f"Edge type name '{original_name}' auto-converted to '{edge['name']}'")
            # 修正 source_targets 中的实体名称引用，与转换后的 PascalCase 保持一致
            for st in edge.get("source_targets", []):
                if st.get("source") in entity_name_map:
                    st["source"] = entity_name_map[st["source"]]
                if st.get("target") in entity_name_map:
                    st["target"] = entity_name_map[st["target"]]
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Zep API 限制：最多 10 个自定义实体类型，最多 10 个自定义边类型
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # 去重：按 name 去重，保留首次出现的
        seen_names = set()
        deduped = []
        for entity in result["entity_types"]:
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(entity)
            elif name in seen_names:
                logger.warning(f"Duplicate entity type '{name}' removed during validation")
        result["entity_types"] = deduped

        # 兜底类型定义（物质科学领域）
        material_fallback = {
            "name": "Material",
            "description": "Generic material entity for substances not fitting more specific material subtypes.",
            "attributes": [
                {"name": "formula", "type": "text", "description": "Chemical formula or composition of the material"},
                {"name": "material_type", "type": "text", "description": "Coarse category, e.g. compound/alloy/polymer/film"}
            ],
            "examples": ["unnamed intermediate", "generic substrate"]
        }

        concept_fallback = {
            "name": "Concept",
            "description": "Abstract scientific concept: theory, mechanism, phenomenon, principle or methodology.",
            "attributes": [
                {"name": "concept_type", "type": "text", "description": "Kind of concept, e.g. theory/mechanism/phenomenon"},
                {"name": "definition", "type": "text", "description": "Short definition of the concept"}
            ],
            "examples": ["Schottky barrier", "free-radical polymerization mechanism"]
        }

        # 检查是否已有兜底类型
        entity_names = {e["name"] for e in result["entity_types"]}
        has_material = "Material" in entity_names
        has_concept = "Concept" in entity_names

        # 需要添加的兜底类型
        fallbacks_to_add = []
        if not has_material:
            fallbacks_to_add.append(material_fallback)
        if not has_concept:
            fallbacks_to_add.append(concept_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # 如果添加后会超过 10 个，需要移除一些现有类型
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # 计算需要移除多少个
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # 从末尾移除（保留前面更重要的具体类型）
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # 添加兜底类型
            result["entity_types"].extend(fallbacks_to_add)
        
        # 最终确保不超过限制（防御性编程）
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        将本体定义转换为Python代码（类似ontology.py）
        
        Args:
            ontology: 本体定义
            
        Returns:
            Python代码字符串
        """
        code_lines = [
            '"""',
            '自定义实体类型定义',
            '由 Federal_KG 自动生成，用于物质科学垂直知识图谱',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== 实体类型定义 ==============',
            '',
        ]
        
        # 生成实体类型
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== 关系类型定义 ==============')
        code_lines.append('')
        
        # 生成关系类型
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # 转换为PascalCase类名
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # 生成类型字典
        code_lines.append('# ============== 类型配置 ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # 生成边的source_targets映射
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

