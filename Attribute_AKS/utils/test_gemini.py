import asyncio
from gemini_webapi import GeminiClient
from gemini_webapi.constants import Model
# Replace "COOKIE VALUE HERE" with your actual cookie values.
# Leave Secure_1PSIDTS empty if it's not available for your account.
Secure_1PSID = "g.a0004gh93Ev7sY1mvu8LGEAzjwbyExVZNYTfpsO6QDcxobNTBjeOfnvI9AjGSHp3JxNnO48jGQACgYKAfESARASFQHGX2MiW4v-thqjBiWM9h9GMSyO9BoVAUF8yKpRTqr2KcZb3oMjXc9GA9IN0076"
Secure_1PSIDTS = "sidts-CjIBflaCdYu1H2CpG3X-4knwWASaKQFwPq5VQXErmm0b2ZZOcH5DZ7y8Qq_VjiYVt5eA9hAA"

async def main():
    client = GeminiClient()
    # await client.init(timeout=60, auto_close=False, close_delay=300, auto_refresh=False)

    response = await client.generate_content(
            # "请你介绍一下秦始皇",
            "该视频主要用于介绍假发，请你从视频中提取商品的这几个属性的值，以json格式输出，属性列表为:['Color', 'Hair Type', 'Material'] ",
            model=Model.G_3_0_PRO,
            files=["ShortForm-Generic-480p-16-9-1409173089793-rpcbe5.mp4"],
        )
    print(response.text)

asyncio.run(main())