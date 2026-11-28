"""
Модуль для Telegram бота, интегрированного с RAG-ассистентом.

Бот позволяет пользователям задавать вопросы ассистенту через Telegram
и получать ответы на основе векторного поиска и LLM.
"""

import os
import time
from typing import Optional
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from telegram.error import TimedOut, NetworkError, RetryAfter
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger


class TelegramRAGBot:
    """
    Telegram бот для RAG-ассистента.
    
    Обрабатывает команды и сообщения от пользователей,
    логирует все взаимодействия в базу данных.
    """

    def __init__(
            self,
            token: str,
            rag_assistant: RAGAssistant,
            cache: ResponseCache,
            logger: DatabaseLogger
    ):
        """
        Инициализация Telegram бота.

        Args:
            token: Токен Telegram бота от @BotFather
            rag_assistant: Экземпляр RAG-ассистента
            cache: Экземпляр кеша ответов
            logger: Экземпляр логгера базы данных
        """
        self.rag_assistant = rag_assistant
        self.cache = cache
        self.logger = logger

        # Увеличиваем таймауты, чтобы реже ловить ReadTimeout
        from telegram.request import HTTPXRequest

        request = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=30.0,
            read_timeout=30.0,
            write_timeout=30.0,
            pool_timeout=30.0,
        )

        # Создаём приложение с увеличенными таймаутами
        self.application = (
            Application.builder()
            .token(token)
            .request(request)
            .build()
        )

        # Регистрируем обработчики команд
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("logs", self.logs_command))

        # Регистрируем обработчик текстовых сообщений
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_message = """
🤖 Добро пожаловать в RAG-ассистента!

Я могу отвечать на ваши вопросы, используя базу знаний.

Доступные команды:
/help - показать справку
/stats - статистика системы
/logs - получить логи в CSV формате

Просто напишите мне вопрос, и я постараюсь на него ответить!
        """
        await update.message.reply_text(welcome_message.strip())

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
📚 Справка по использованию бота:

• Просто напишите вопрос - я отвечу на основе базы знаний
• Использую RAG (Retrieval-Augmented Generation) для точных ответов
• Ответы кешируются для быстрой работы

Команды:
/start - начать работу с ботом
/help - показать эту справку
/stats - статистика системы (документы, кеш)
/logs - получить логи взаимодействий в CSV формате

Примеры вопросов:
• "Что такое Python?"
• "Расскажи про RAG"
• "Что такое векторные базы данных?"
        """
        await update.message.reply_text(help_text.strip())

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        try:
            # Получаем статистику системы
            doc_count = self.rag_assistant.embedding_store.collection.count()
            cache_size = self.cache.size()
            model = self.rag_assistant.model

            # Получаем статистику из логов
            log_stats = self.logger.get_stats()

            stats_message = f"""
📊 СТАТИСТИКА СИСТЕМЫ:

📚 База знаний:
  • Документов в ChromaDB: {doc_count}
  • Модель LLM: {model}

💾 Кеш:
  • Записей в кеше: {cache_size}

📝 Логи:
  • Всего запросов: {log_stats['total_requests']}
  • Из кеша: {log_stats['cached_requests']}
  • Уникальных пользователей: {log_stats['unique_users']}
  • Среднее время ответа: {log_stats['avg_response_time_ms']:.0f} мс
            """

            await update.message.reply_text(stats_message.strip())

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при получении статистики: {str(e)}")

    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /logs - экспорт логов в CSV"""
        try:
            user_id = str(update.effective_user.id)

            # Экспортируем логи текущего пользователя
            csv_content = self.logger.export_to_csv(user_id=user_id)

            if not csv_content:
                await update.message.reply_text(
                    "📝 Логов для вашего пользователя не найдено."
                )
                return

            # Сохраняем во временный файл
            filename = f"logs_{user_id}_{int(time.time())}.csv"
            with open(filename, 'w', encoding='utf-8-sig') as f:
                f.write(csv_content)

            # Отправляем файл пользователю
            with open(filename, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=filename,
                    caption="📊 Ваши логи взаимодействий с ботом"
                )

            # Удаляем временный файл
            os.remove(filename)

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при экспорте логов: {str(e)}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений от пользователей"""
        user_message = update.message.text
        user = update.effective_user
        user_id = str(user.id)
        username = user.username or user.first_name or "Unknown"

        start_time = time.time()

        try:
            print(f"\n❓ Новый запрос от {username} (ID: {user_id})")

            await update.message.chat.send_action(action="typing")

            # Шаг 1. Проверка кеша
            print("[Шаг 1] Проверка кеша...")
            cached_answer = self.cache.get(user_message)
            from_cache = cached_answer is not None

            if from_cache:
                print("✓ Найден ответ в кеше")
                answer = cached_answer
            else:
                # Шаг 2. Выполнение RAG
                print("[Шаг 2] Выполнение RAG (поиск + генерация)...")

                answer, _ = self.rag_assistant.generate_response(
                    query=user_message,
                    top_k=3,
                    verbose=False
                )

                # Шаг 3. Сохранение в кеш
                print("[Шаг 3] Сохранение ответа в кеш...")
                self.cache.set(user_message, answer)

            # Шаг 4. Вывод вопроса и ответа в лог
            answer_preview = " ".join(answer.split())[:70]

            print(f"\n📝 ВОПРОС: {user_message}")
            print(f"📌 ИСТОЧНИК: {'КЕШ' if from_cache else 'RAG + LLM'}")
            print(f"💬 ОТВЕТ (первые 70 символов): {answer_preview}")
            print(f"📏 Полная длина ответа: {len(answer)} символов")

            # Шаг 5. Запись в базу данных
            response_time_ms = int((time.time() - start_time) * 1000)

            print("[Шаг 4] Запись взаимодействия в базу данных...")

            self.logger.log_interaction(
                query=user_message,
                response=answer,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=from_cache,
                response_time_ms=response_time_ms
            )
            print("✓ Взаимодействие записано")

            # Шаг 6. Отправка ответа
            print(f"[Шаг 5] Отправка ответа ({len(answer)} символов)...")

            max_length = 4000
            for i in range(0, len(answer), max_length):
                await update.message.reply_text(answer[i:i + max_length])

            if from_cache:
                await update.message.reply_text("💾 (ответ из кеша)")

            print(f"✓ Обработка завершена за {response_time_ms} мс")

        except Exception as e:
            error_message = f"❌ Произошла ошибка при обработке запроса: {str(e)}"
            print(error_message)

            await update.message.reply_text(error_message)

            self.logger.log_interaction(
                query=user_message,
                response=error_message,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=False,
                response_time_ms=int((time.time() - start_time) * 1000)
            )

    def run(self):
        print("🤖 Запуск Telegram бота...")
        print("Бот готов к работе! Нажмите Ctrl+C для остановки.")

        while True:
            try:
                self.application.run_polling(
                    drop_pending_updates=True,
                    allowed_updates=Update.ALL_TYPES,
                    close_loop=False,
                )
                break
            except RetryAfter as e:
                wait = int(e.retry_after) + 2
                print(f"⏳ Flood control. Жду {wait} сек...")
                time.sleep(wait)
            except (TimedOut, NetworkError) as e:
                print(f"🌐 Таймаут/сеть: {e}. Повтор через 10 сек...")
                time.sleep(10)
            except Exception as e:
                print(f"❌ Неожиданная ошибка: {e}")
                time.sleep(15)
