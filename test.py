import asyncio
from telegram import Bot

async def test():
    bot = Bot(token="9174419496:AAEJcszOWizoRuLjNRwbcm88EY0lVfYusv4")
    me = await bot.get_me()
    print(me)

asyncio.run(test())
