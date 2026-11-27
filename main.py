"""
Главный файл для запуска RAG-ассистента.

Это точка входа в приложение. Здесь происходит:
1. Загрузка конфигурации из файла .env
2. Инициализация компонентов системы (кеш, векторная база, RAG, логгер)
3. Загрузка документов в векторную базу знаний при первом запуске
4. Создание и запуск Telegram-бота

Обработка пользовательских запросов осуществляется в модуле telegram_bot.py.
Интерактивный консольный режим работы не предусмотрен.
"""

import os
import traceback
from dotenv import load_dotenv

from embeddings import EmbeddingStore, get_sample_documents
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger
from telegram_bot import TelegramRAGBot


def initialize_system():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not api_key:
        raise ValueError("Не задан OPENAI_API_KEY в файле .env")

    if not telegram_token:
        raise ValueError("Не задан TELEGRAM_BOT_TOKEN в файле .env")

    print("\n🚀 ИНИЦИАЛИЗАЦИЯ RAG-АССИСТЕНТА")

    print("\n[1/4] Инициализация кеша...")
    cache = ResponseCache(cache_file="cache.json")

    print("\n[2/4] Инициализация векторного хранилища...")
    embedding_store = EmbeddingStore(
        collection_name="rag_documents",
        persist_directory="./chroma_db",
        embedding_model="text-embedding-3-small",
        api_key=api_key,
        base_url=base_url
    )

    if embedding_store.collection.count() == 0:
        print("\n📝 База данных пуста. Загружаем документы...")
        documents = get_sample_documents()
        embedding_store.add_documents(documents)
    else:
        print(
            f"✓ В базе уже есть "
            f"{embedding_store.collection.count()} документов"
        )

    print("\n[3/4] Инициализация RAG-ассистента...")
    rag_assistant = RAGAssistant(
        embedding_store=embedding_store,
        model="gpt-4o-mini",
        temperature=0.7
    )

    print("\n[4/4] Инициализация логгера...")
    logger = DatabaseLogger(db_path="logs.db")

    print("\n✅ СИСТЕМА ГОТОВА К РАБОТЕ")

    return telegram_token, rag_assistant, cache, logger


def main():
    try:
        token, rag_assistant, cache, logger = initialize_system()

        print("\n🤖 ЗАПУСК TELEGRAM БОТА")

        bot = TelegramRAGBot(
            token=token,
            rag_assistant=rag_assistant,
            cache=cache,
            logger=logger
        )

        bot.run()

    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
