"""
TODO: graph_repository.py'nin (özellikle find_path_between) LLM extraction'dan
BAĞIMSIZ olarak doğru çalıştığını kanıtlayan test. Elle, temiz veri ekleyerek,
algoritmanın kendisinin doğru olduğunu izole şekilde gösteriyoruz - bu, LLM'in
ürettiği gürültülü veriden tamamen ayrı bir doğrulama katmanı.
"""

import tempfile
from pathlib import Path

import pytest

from app.repositories.graph_repository import GraphRepository


@pytest.fixture
def repo():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = Path(tmp_dir) / "test_graph.pkl"
        yield GraphRepository(store_path=store_path)


def test_find_path_between_two_hops(repo):
    # Elle, TEMİZ ve doğru veri ekliyoruz - LLM çıkarım hatalarından tamamen bağımsız
    repo.add_entity("Mehmet Demir", "kişi", document_id="doc1")
    repo.add_entity("Rüzgar Projesi", "kavram", document_id="doc1")
    repo.add_entity("Ayşe Er", "kişi", document_id="doc2")

    repo.add_relationship("Mehmet Demir", "yönetir", "Rüzgar Projesi", document_id="doc1", chunk_id="c1")
    repo.add_relationship("Ayşe Er", "sorumludur", "Rüzgar Projesi", document_id="doc2", chunk_id="c2")

    path = repo.find_path_between("Mehmet Demir", "Ayşe Er")

    assert len(path) == 2


def test_no_path_when_disconnected(repo):
    repo.add_entity("A", "kavram", document_id="doc1")
    repo.add_entity("B", "kavram", document_id="doc2")
    # hiç ilişki eklenmedi - bağlantı olmamalı
    path = repo.find_path_between("A", "B")
    assert path == []


def test_no_path_when_entities_missing(repo):
    path = repo.find_path_between("olmayan varlık 1", "olmayan varlık 2")
    assert path == []
