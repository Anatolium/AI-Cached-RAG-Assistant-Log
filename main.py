"""
Главный файл для запуска RAG-ассистента.

Это точка входа в приложение. Здесь происходит:
1. Загрузка конфигурации
2. Инициализация всех компонентов (кеш, векторная база, RAG)
3. Добавление примеров документов (при первом запуске)
4. Запуск Telegram-бота
5. Исключён интерактивный цикл общения с пользователем
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv
from embeddings import EmbeddingStore, get_sample_documents
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger
from telegram_bot import TelegramRAGBot


def initialize_system():
    """
    Инициализирует все компоненты RAG-системы.

    Returns:
        Кортеж (embedding_store, rag_assistant, cache, logger)
    """
    print("=" * 70)
    print("🚀 ИНИЦИАЛИЗАЦИЯ RAG-АССИСТЕНТА")
    print("=" * 70)

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError(
            "Не задан OPENAI_API_KEY в файле .env"
        )

    # 1. Инициализируем кеш для хранения ответов
    print("\n[1/4] Инициализация кеша...")
    cache = ResponseCache(cache_file="cache.json")

    # 2. Инициализируем векторное хранилище ChromaDB
    print("\n[2/4] Инициализация векторного хранилища...")
    embedding_store = EmbeddingStore(
        collection_name="rag_documents",
        persist_directory="./chroma_db",
        embedding_model="text-embedding-3-small",  # Модель OpenAI для эмбеддингов
        api_key=api_key,
        base_url=base_url
    )

    # Проверяем, нужно ли добавить примеры документов
    if embedding_store.collection.count() == 0:
        print("\n📝 База данных пуста. Добавляем примеры документов...")
        sample_docs = get_sample_documents()
        embedding_store.add_documents(sample_docs)
    else:
        print(f"✓ В базе уже есть {embedding_store.collection.count()} документов")

    # 3. Инициализируем RAG-ассистента
    print("\n[3/4] Инициализация RAG-ассистента...")
    rag_assistant = RAGAssistant(
        embedding_store=embedding_store,
        model="gpt-4o-mini",
        temperature=0.7
    )

    # 4. Инициализируем логгер базы данных
    print("\n[4/4] Инициализация логгера базы данных...")
    logger = DatabaseLogger(db_path="logs.db")
    print("✓ Логгер инициализирован")

    print("\n" + "=" * 70)
    print("✅ СИСТЕМА ГОТОВА К РАБОТЕ")
    print("=" * 70)

    return embedding_store, rag_assistant, cache, logger


def answer_question(
        query: str,
        rag_assistant: RAGAssistant,
        cache: ResponseCache,
        logger: DatabaseLogger,
        source: str = "telegram",
        user_id: Optional[str] = None,
        username: Optional[str] = None
) -> str:
    """
    Отвечает на вопрос пользователя с использованием кеша и RAG.

    Логика работы:
    1. Проверяем кеш - если ответ есть, возвращаем его
    2. Если ответа нет, выполняем RAG (поиск + генерация)
    3. Сохраняем новый ответ в кеш
    4. Логируем взаимодействие в базу данных
    5. Возвращаем ответ

    Args:
        query: Вопрос пользователя
        rag_assistant: Экземпляр RAG-ассистента
        cache: Экземпляр кеша
        logger: Экземпляр логгера базы данных
        source: Источник запроса (telegram и т.д.)
        user_id: ID пользователя (для Telegram)
        username: Имя пользователя

    Returns:
        Ответ на вопрос
    """
    print("\n" + "=" * 70)
    print(f"❓ ВОПРОС: {query}")
    print("=" * 70)

    start_time = time.time()

    # Шаг 1: Проверяем кеш
    print("\n[Шаг 1] Проверка кеша...")
    cached_answer = cache.get(query)
    from_cache = cached_answer is not None

    if cached_answer:
        # Ответ найден в кеше - возвращаем его
        print("\n💾 Ответ из кеша:")
        print("-" * 70)
        print(cached_answer)
        print("-" * 70)
        answer = cached_answer
    else:
        # Шаг 2: Ответа нет в кеше - выполняем RAG
        print("\n[Шаг 2] Выполнение RAG (поиск + генерация)...")

        try:
            answer, search_results = rag_assistant.generate_response(
                query=query,
                top_k=3,
                verbose=True
            )

            # Шаг 3: Сохраняем ответ в кеш
            print("\n[Шаг 3] Сохранение ответа в кеш...")
            cache.set(query, answer)

            # Выводим финальный ответ
            print("\n💡 ОТВЕТ:")
            print("-" * 70)
            print(answer)
            print("-" * 70)

        except Exception as e:
            error_msg = f"Ошибка при обработке запроса: {str(e)}"
            print(f"\n❌ {error_msg}")
            answer = error_msg

    # Шаг 4: Логируем взаимодействие
    response_time_ms = int((time.time() - start_time) * 1000)
    logger.log_interaction(
        query=query,
        response=answer,
        source=source,
        user_id=user_id,
        username=username,
        from_cache=from_cache,
        response_time_ms=response_time_ms
    )

    return answer


def main():
    """
    Главная функция приложения — запуск Telegram-бота.
    """
    try:
        # Инициализируем систему
        embedding_store, rag_assistant, cache, logger = initialize_system()

        load_dotenv()

        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")

        # Запускаем Telegram бота
        print("\n" + "=" * 70)
        print("🤖 ЗАПУСК TELEGRAM БОТА")
        print("=" * 70)
        bot = TelegramRAGBot(
            token=telegram_token,
            rag_assistant=rag_assistant,
            cache=cache,
            logger=logger
        )
        bot.run()

    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
