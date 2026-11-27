# Запрос к Telegram Bot API для получения информации о боте: имя пользователя (username), имя (first_name), статус…

import asyncio
from telegram import Bot


async def test():
    bot = Bot(token="9174419496:AAEJcszOWizoRuLjNRwbcm88EY0lVfYusv4")

    me = await bot.get_me()
    # print(me)
    print("Информация о боте:")
    print(f"  ID:          {me.id}")
    print(f"  Имя:         {me.first_name}")
    print(f"  Username:    @{me.username}")
    print(f"  Бот:         {me.is_bot}")


asyncio.run(test())
