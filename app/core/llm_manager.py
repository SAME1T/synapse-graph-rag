"""
TODO: Foundry Local SDK ile yerel LLM'lere bağlanan tek merkezi nokta.
Farklı görevler farklı modeller kullanabilir (örn. hızlı bir model generation için,
daha güçlü bir model yapılandırılmış çıkarım için) - bu sınıf birden fazla modeli
aynı anda belleğe alıp alias'a göre yönetir, aynı model iki görevde de kullanılıyorsa
gereksiz yeniden yükleme yapmaz.
"""

import logging
from typing import Dict, Optional, Tuple

from foundry_local_sdk import Configuration, FoundryLocalManager

from app.core.config import settings

logger = logging.getLogger(__name__)


class LocalLLMManager:
    def __init__(self):
        self._manager = None
        self._eps_registered = False
        self._loaded_models: Dict[str, Tuple[object, object]] = {}  # alias -> (model, chat_client)

    def _ensure_foundry_initialized(self) -> None:
        if self._manager is not None:
            return
        try:
            config = Configuration(app_name="synapse_graph_rag")
            FoundryLocalManager.initialize(config)
            self._manager = FoundryLocalManager.instance
        except Exception as exc:
            logger.error(f"Foundry Local başlatılamadı: {exc}")
            raise RuntimeError("Foundry Local servisi başlatılamadı.") from exc

    def _ensure_eps_registered(self) -> None:
        if self._eps_registered:
            return
        self._manager.download_and_register_eps()
        self._eps_registered = True

    def load_model(
        self,
        model_alias: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        frequency_penalty: float | None = None,
    ) -> None:
        """
        Belirtilen alias'a sahip modeli indirir (yoksa) ve belleğe yükler.
        temperature/max_tokens/frequency_penalty verilirse, üretim davranışını
        sınırlamak için client.settings üzerine uygulanır - ÖZELLİKLE max_tokens
        önemli, çünkü onsuz model bazen tekrar döngüsüne girip sınırsız üretime
        devam edebiliyor (gerçek bir dokümanla test ederken gözlemlendi).
        """
        if model_alias in self._loaded_models:
            return

        self._ensure_foundry_initialized()
        self._ensure_eps_registered()

        try:
            logger.info(f"Model aranıyor: {model_alias}")
            model = self._manager.catalog.get_model(model_alias)

            logger.info(f"Model indiriliyor (önbellekte varsa atlanır): {model_alias}")
            model.download()

            logger.info(f"Model belleğe yükleniyor: {model_alias}")
            model.load()

            chat_client = model.get_chat_client()

            if temperature is not None and hasattr(chat_client, "settings"):
                chat_client.settings.temperature = temperature
            if max_tokens is not None and hasattr(chat_client, "settings"):
                chat_client.settings.max_tokens = max_tokens
            if frequency_penalty is not None and hasattr(chat_client.settings, "frequency_penalty"):
                chat_client.settings.frequency_penalty = frequency_penalty

            if hasattr(chat_client.settings, "stop"):
                chat_client.settings.stop = ["\nuser", "user\n", "KAYNAK METİNLER"]

            self._loaded_models[model_alias] = (model, chat_client)
            logger.info(
                f"Model hazır: {model_alias} "
                f"(temperature={temperature}, max_tokens={max_tokens}, frequency_penalty={frequency_penalty})"
            )

        except Exception as exc:
            logger.error(f"Model yüklenemedi ({model_alias}): {exc}")
            raise RuntimeError(
                f"'{model_alias}' modeli yüklenemedi. Foundry Local servisinin "
                f"çalıştığından ve modelin kurulu olduğundan emin olun."
            ) from exc

    def generate(
        self, system_prompt: str, user_prompt: str, model_alias: Optional[str] = None
    ) -> str:
        """
        Belirtilen (veya varsayılan generation) modelle tek seferlik bir cevap üretir.
        Model henüz yüklü değilse otomatik olarak yükler (lazy loading) -
        böylece extraction modeli sadece ilk kullanıldığında indirilir.
        """
        alias = model_alias or settings.LLM_MODEL_ALIAS

        if alias not in self._loaded_models:
            self.load_model(alias)

        _, chat_client = self._loaded_models[alias]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = chat_client.complete_chat(messages)
            return response.choices[0].message.content
        except Exception as exc:
            logger.error(f"Cevap üretilirken hata oluştu ({alias}): {exc}")
            raise RuntimeError("Yerel model cevap üretemedi.") from exc

    def shutdown(self) -> None:
        """Yüklü tüm modelleri bellekten kaldırır."""
        for alias, (model, _) in list(self._loaded_models.items()):
            try:
                model.unload()
                logger.info(f"Model bellekten kaldırıldı: {alias}")
            except Exception as exc:
                logger.warning(f"Model kaldırılırken hata oluştu ({alias}): {exc}")
        self._loaded_models.clear()


llm_manager = LocalLLMManager()