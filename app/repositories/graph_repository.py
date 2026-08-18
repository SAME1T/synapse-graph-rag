"""
TODO: Varlık-ilişki grafiğini NetworkX ile bellekte tutan ve diske kaydeden repository.
Node isimleri normalize edilerek (küçük harf) saklanır - "Python" ve "python" gibi
yazım farklarını birleştirir, ama "ABD" ile "Amerika Birleşik Devletleri" gibi
anlamca aynı ama yazım olarak farklı varlıkları BİRLEŞTİRMEZ. Bu bilinçli bir
basitleştirme - ileride embedding tabanlı entity resolution eklenebilir.
NOT: networkx 3.x kendi gpickle fonksiyonlarını kaldırdığı için standart pickle kullanılıyor.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import networkx as nx

from app.core.config import settings

logger = logging.getLogger(__name__)


class GraphRepository:
    """
    Varlıkları düğüm (node), ilişkileri kenar (edge) olarak tutan yönlendirilmiş
    çoklu graf (iki varlık arasında birden fazla farklı ilişki olabilir).
    """

    def __init__(self, store_path: Path = settings.GRAPH_STORE_PATH):
        self._store_path = store_path
        self._graph: Optional[nx.MultiDiGraph] = None

    def _ensure_loaded(self) -> None:
        if self._graph is not None:
            return

        if self._store_path.exists():
            try:
                with self._store_path.open("rb") as f:
                    self._graph = pickle.load(f)
                logger.info(
                    f"Graf diskten yüklendi: {self._graph.number_of_nodes()} varlık, "
                    f"{self._graph.number_of_edges()} ilişki."
                )
            except Exception as exc:
                logger.error(f"Graf dosyası okunamadı, yeni graf oluşturuluyor: {exc}")
                self._graph = nx.MultiDiGraph()
        else:
            self._graph = nx.MultiDiGraph()
            logger.info("Yeni boş graf oluşturuldu (henüz kayıtlı dosya yok).")

    def _save(self) -> None:
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            with self._store_path.open("wb") as f:
                pickle.dump(self._graph, f)
        except Exception as exc:
            logger.error(f"Graf diske kaydedilemedi: {exc}")
            raise RuntimeError("Graf kaydedilemedi.") from exc

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()

    def add_entity(self, name: str, entity_type: str, document_id: str) -> None:
        self._ensure_loaded()
        node_id = self._normalize(name)
        if not node_id:
            return

        if self._graph.has_node(node_id):
            self._graph.nodes[node_id]["document_ids"].add(document_id)
        else:
            self._graph.add_node(
                node_id,
                display_name=name.strip(),
                type=entity_type,
                document_ids={document_id},
            )

    def add_relationship(
        self, subject: str, predicate: str, obj: str, document_id: str, chunk_id: str
    ) -> None:
        self._ensure_loaded()
        subject_id = self._normalize(subject)
        object_id = self._normalize(obj)
        if not subject_id or not object_id:
            return

        # İlişkinin uçlarındaki varlıklar entities listesinde kaçmış olabilir - güvenlik için ekliyoruz.
        if not self._graph.has_node(subject_id):
            self._graph.add_node(
                subject_id, display_name=subject.strip(), type="unknown", document_ids={document_id}
            )
        if not self._graph.has_node(object_id):
            self._graph.add_node(
                object_id, display_name=obj.strip(), type="unknown", document_ids={document_id}
            )

        self._graph.add_edge(
            subject_id, object_id, predicate=predicate, document_id=document_id, chunk_id=chunk_id
        )

    def commit(self) -> None:
        """Bellekteki değişiklikleri diske yazar - bir doküman işlendikten sonra bir kez çağrılır."""
        self._ensure_loaded()
        self._save()

    def get_neighbors(self, entity_name: str, depth: int = 1) -> List[dict]:
        """
        Bir varlıktan başlayarak 'depth' adım uzağa kadar bağlı tüm varlıkları döner.
        GraphRAG'ın çekirdek fonksiyonu - vektör aramanın bulamadığı ilişkisel
        soruları bu cevaplar.
        """
        self._ensure_loaded()
        node_id = self._normalize(entity_name)
        if not self._graph.has_node(node_id):
            return []

        undirected_view = self._graph.to_undirected(as_view=True)
        nearby_nodes = nx.single_source_shortest_path_length(undirected_view, node_id, cutoff=depth)

        results = []
        for neighbor_id, hops in nearby_nodes.items():
            if neighbor_id == node_id:
                continue
            node_data = self._graph.nodes[neighbor_id]
            results.append(
                {
                    "entity": node_data.get("display_name", neighbor_id),
                    "type": node_data.get("type", "unknown"),
                    "hops_away": hops,
                }
            )
        return results

    def get_relationships_for_entity(self, entity_name: str) -> List[dict]:
        """Bir varlığın doğrudan (1 hop) tüm ilişkilerini (giden ve gelen) döner."""
        self._ensure_loaded()
        node_id = self._normalize(entity_name)
        if not self._graph.has_node(node_id):
            return []

        relationships = []
        for _, target, data in self._graph.out_edges(node_id, data=True):
            relationships.append(
                {
                    "subject": self._graph.nodes[node_id]["display_name"],
                    "predicate": data.get("predicate", ""),
                    "object": self._graph.nodes[target]["display_name"],
                }
            )
        for source, _, data in self._graph.in_edges(node_id, data=True):
            relationships.append(
                {
                    "subject": self._graph.nodes[source]["display_name"],
                    "predicate": data.get("predicate", ""),
                    "object": self._graph.nodes[node_id]["display_name"],
                }
            )
        return relationships

    def find_path_between(self, entity_a: str, entity_b: str, max_depth: int = 3) -> List[dict]:
        """
        İki varlık arasındaki EN KISA bağlantı zincirini bulur (yön önemsemeden).
        "X ile Y arasındaki bağlantı nedir" sorularının doğru cevabı budur - her iki
        varlığın kendi ilişkilerini ayrı ayrı listelemek yerine, aralarındaki gerçek
        zinciri (varsa bir ara varlık üzerinden) tek, bağlantılı bir yapı olarak döner.
        """
        self._ensure_loaded()
        node_a = self._normalize(entity_a)
        node_b = self._normalize(entity_b)

        if not self._graph.has_node(node_a) or not self._graph.has_node(node_b):
            return []

        undirected_view = self._graph.to_undirected(as_view=True)
        try:
            path_nodes = nx.shortest_path(undirected_view, node_a, node_b)
        except nx.NetworkXNoPath:
            return []

        if len(path_nodes) - 1 > max_depth:
            return []

        path_facts = []
        for i in range(len(path_nodes) - 1):
            source, target = path_nodes[i], path_nodes[i + 1]
            if self._graph.has_edge(source, target):
                edge_data = list(self._graph.get_edge_data(source, target).values())[0]
                subject_display = self._graph.nodes[source]["display_name"]
                object_display = self._graph.nodes[target]["display_name"]
            else:
                edge_data = list(self._graph.get_edge_data(target, source).values())[0]
                subject_display = self._graph.nodes[target]["display_name"]
                object_display = self._graph.nodes[source]["display_name"]

            path_facts.append(
                {
                    "subject": subject_display,
                    "predicate": edge_data.get("predicate", ""),
                    "object": object_display,
                }
            )
        return path_facts

    def find_entities_by_name(self, query: str) -> List[dict]:
        """
        Kullanıcı sorusundaki kelimelerin graftaki hangi varlıklara denk geldiğini
        basit bir alt-metin araması ile bulur. retrieval_service, soruda hangi
        varlıkların geçtiğini tespit etmek için bunu kullanacak (Adım 20).
        """
        self._ensure_loaded()
        query_lower = query.lower()
        matches = []
        for node_id, data in self._graph.nodes(data=True):
            if node_id and node_id in query_lower:
                matches.append(
                    {"entity": data.get("display_name", node_id), "type": data.get("type", "unknown")}
                )
        return matches

    def count_entities(self) -> int:
        self._ensure_loaded()
        return self._graph.number_of_nodes()

    def count_relationships(self) -> int:
        self._ensure_loaded()
        return self._graph.number_of_edges()

    def dump_all(self) -> dict:
        """
        Debug amaçlı: graftaki TÜM düğümleri ve TÜM kenarları döner.
        Coreference/normalizasyon sorunlarını teşhis etmek için kullanılır.
        """
        self._ensure_loaded()
        nodes = [
            {
                "id": node_id,
                "display_name": data.get("display_name", node_id),
                "type": data.get("type", "unknown"),
            }
            for node_id, data in self._graph.nodes(data=True)
        ]
        edges = [
            {
                "subject": self._graph.nodes[u]["display_name"],
                "predicate": data.get("predicate", ""),
                "object": self._graph.nodes[v]["display_name"],
            }
            for u, v, data in self._graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}


graph_repository = GraphRepository()
