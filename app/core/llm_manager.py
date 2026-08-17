"""
TODO: Foundry Local SDK ile yerel LLM'e bağlanan tek merkezi nokta.
Projedeki başka hiçbir dosya foundry_local_sdk'yi doğrudan import etmez,
hepsi bu sınıf üzerinden konuşur.
"""

import logging
from foundry_local_sdk import Configuration, FoundryLocalManager

from app.core.config import settings

logger = logging.getLogger(__name__)


class LocalLLMManager:
    """
    Foundry Local üzerinde çalışan yerel LLM ile iletişimi yöneten sınıf.
    Model indirme, belleğe yükleme ve cevap üretme burada toplanır.
    """

    def __init__(self, model_alias: str = settings.LLM_MODEL_ALIAS):
        self._model_alias = model_alias
        self._manager = None
        self._model = None
        self._chat_client = None
        self._is_ready = False

    def initialize(self) -> None:
        """
        Foundry Local servisini başlatır, modeli indirir (yoksa) ve belleğe yükler.
        Uygulama açılışında BİR KEZ çağrılmalı (main.py içinde bağlayacağız).
        """
        try:
            config = Configuration(app_name="synapse_graph_rag")
            FoundryLocalManager.initialize(config)
            self._manager = FoundryLocalManager.instance

            logger.info("Execution provider'lar kaydediliyor...")
            self._manager.download_and_register_eps()

            logger.info(f"Model aranıyor: {self._model_alias}")
            self._model = self._manager.catalog.get_model(self._model_alias)

            logger.info("Model indiriliyor (önbellekte varsa atlanır)...")
            self._model.download()

            logger.info("Model belleğe yükleniyor...")
            self._model.load()

            self._chat_client = self._model.get_chat_client()
            self._is_ready = True
            logger.info(f"Model hazır: {self._model_alias}")

        except Exception as exc:
            logger.error(f"Foundry Local başlatılamadı: {exc}")
            raise RuntimeError(
                f"Yerel LLM başlatılamadı. Foundry Local servisinin çalıştığından "
                f"ve '{self._model_alias}' modelinin kurulu olduğundan emin olun."
            ) from exc

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Verilen system prompt + kullanıcı mesajına göre tek seferlik bir cevap üretir.
        RAG cevaplarını üretirken bunu kullanacağız.
        """
        if not self._is_ready:
            raise RuntimeError("LLM henüz başlatılmadı. Önce initialize() çağrılmalı.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._chat_client.complete_chat(messages)
            return response.choices[0].message.content
        except Exception as exc:
            logger.error(f"Cevap üretilirken hata oluştu: {exc}")
            raise RuntimeError("Yerel model cevap üretemedi.") from exc

    def shutdown(self) -> None:
        """
        Uygulama kapanırken modeli bellekten kaldırır.
        """
        if self._model and self._is_ready:
            self._model.unload()
            self._is_ready = False
            logger.info("Model bellekten kaldırıldı.")


# Tüm uygulamanın kullanacağı tek nesne (basit bir singleton mantığı)
llm_manager = LocalLLMManager()