"""
TODO: Graf istatistiklerini ve varlık komşuluklarını sorgulamak için endpoint'ler.
"""

import logging

from fastapi import APIRouter

from app.api.schemas.graph_schemas import (
    EntityNeighborsResponse,
    GraphDumpResponse,
    GraphStatsResponse,
)
from app.repositories.graph_repository import graph_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["Graph"])


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """Graftaki toplam varlık ve ilişki sayısını döner - ingestion sonrası doğrulama için kullanışlı."""
    return GraphStatsResponse(
        entity_count=graph_repository.count_entities(),
        relationship_count=graph_repository.count_relationships(),
    )


@router.get("/entities/{entity_name}/neighbors", response_model=EntityNeighborsResponse)
async def get_entity_neighbors(entity_name: str, depth: int = 1):
    """Bir varlığın 'depth' adım uzağa kadar bağlı olduğu varlıkları ve doğrudan ilişkilerini döner."""
    neighbors = graph_repository.get_neighbors(entity_name, depth=depth)
    relationships = graph_repository.get_relationships_for_entity(entity_name)

    return EntityNeighborsResponse(
        entity=entity_name, neighbors=neighbors, direct_relationships=relationships
    )


@router.get("/dump", response_model=GraphDumpResponse)
async def dump_graph():
    """Debug: graftaki tüm düğüm ve kenarları döner - node kopukluğu/normalizasyon sorunlarını görmek için."""
    data = graph_repository.dump_all()
    return GraphDumpResponse(**data)
