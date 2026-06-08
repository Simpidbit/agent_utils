import asyncio

from openai import AsyncOpenAI

from openai.types.responses import ResponseTextDeltaEvent

BASEURL="http://127.0.0.1:13579/v1"
APIKEY="sk-70f2ac68c6dda8a1ee487bf5972b1aec9645d574f8f02509c675205c6b6ab0ae"
MODEL="gpt-5.5"

client = AsyncOpenAI(api_key = APIKEY, base_url = BASEURL)

async def main1():
    stream = await client.responses.create(
        model = "gpt-5.5",
        input = "用 C 语言写一个完整的 JSON 解析器",
        stream = True
    )
    print(type(stream))

    index = 0
    async for event in stream:
        if isinstance(event, ResponseTextDeltaEvent):
            '''
            class ResponseTextDeltaEvent(BaseModel):
                content_index: int          # 文本 delta 添加到哪个 content part
                delta: str                  # 本次新增的文本片段，最常用
                item_id: str                # 这个 delta 属于哪个输出 item
                logprobs: List[Logprob]     # 本次 delta 中 token 的 log probability 信息，
                                            # logprobs 里的元素是 Logprob，包含 token、logprob 和可选的 top_logprobs；
                                            # top_logprobs 表示最多 20 个最可能 token 的概率信息。
                output_index: int           # 输出 item 在 response output 列表中的索引
                sequence_number: int        # 事件序号，可用于保持事件顺序
                type: Literal["response.output_text.delta"]
            '''
            print(f'{index}: delta: [{event.delta}], seq_num: [{event.sequence_number}], item_id: [{event.item_id}] ')
        else:
            print(f'{index}: {type(event)}')
        index += 1

async def main2():
    async with client.responses.stream(
        model="gpt-5.5",
        input="用三句话介绍 Python asyncio。",
    ) as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)

        final_response = stream.get_final_response()
        # final_response 是累积后的完整 Response 对象
        # 例如可用于日志、拿 response id、usage、结构化输出等

if __name__ == '__main__':
    asyncio.run(main2())
