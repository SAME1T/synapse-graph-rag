"""
TODO: Graf istatistiklerini ve komşuluk sorgularının veri şekillerini tanımlar.
"""

from typing import List

from pydantic import BaseModel


class GraphStatsResponse(BaseModel):
    entity_count: int
    relationship_count: int


class NeighborEntity(BaseModel):
    entity: str
    type: str
    hops_away: int


class EntityRelationship(BaseModel):
    subject: str
    predicate: str
    object: str


class EntityNeighborsResponse(BaseModel):
    entity: str
    neighbors: List[NeighborEntity]
    direct_relationships: List[EntityRelationship]


class GraphNode(BaseModel):
    id: str
    display_name: str
    type: str


class GraphEdge(BaseModel):
    subject: str
    predicate: str
    object: str


class GraphDumpResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
