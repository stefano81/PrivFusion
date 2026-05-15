from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class DatasetInformation:
    name: str
    semantic: SemanticInformation
    structure: StructuralInformation
    relationships: list[SemanticRelationship]


@dataclass
class SemanticInformation:
    description: str
    column_types: list[str | None]
    column_semantic: list[str | None]
    dataset_topics: list[str]


@dataclass
class StructuralInformation:
    names: list[str]
    types: list[str]
    constraints: list[StructuralConstraint]
    description: str


class StructuralConstraint(Enum):
    unique = auto()
    primary = auto()
    foreign = auto()


@dataclass
class SemanticRelationship:
    column_from: str
    column_to: str
    reason: str
