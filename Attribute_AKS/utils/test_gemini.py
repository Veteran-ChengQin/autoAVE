import asyncio
from gemini_webapi import GeminiClient
from gemini_webapi.constants import Model
# Replace "COOKIE VALUE HERE" with your actual cookie values.
# Leave Secure_1PSIDTS empty if it's not available for your account.
Secure_1PSID = "g.a0004gh93Aq-5hKB6eQvMWASnpIclogBUCxsIZw9Fx_otH_aKwWlQCbtMupuGCyQbnw44QvHFAACgYKAVkSARASFQHGX2MiuJeUN18kF3wOtPQAtkH9CRoVAUF8yKrk8JL2V0PDWeZN09jMUMhM0076"
Secure_1PSIDTS = "sidts-CjIBflaCdS_0J1GgfCR8qeqzx0w2nlzshOwGCCBH1IlQNiQKcKQwu-absE3VBhdtkiTQIRAA"

async def main():
    client = GeminiClient(Secure_1PSID, Secure_1PSIDTS, proxy=None)
    await client.init(timeout=30, auto_close=False, close_delay=300, auto_refresh=True)

    response = await client.generate_content(
            "请你介绍一下秦始皇",
            # "请你从视频中提取商品的这几个属性的值，以json格式输出，属性列表为:['Color' , 'Hair Type', 'Brand']",
            model=Model.G_3_0_PRO,
            # files=["ShortForm-Generic-480p-16-9-1409173089793-rpcbe5.mp4"],
        )
    print(response.text)

asyncio.run(main())