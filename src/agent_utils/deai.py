from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai import OpenAIError
from openai import AsyncStream

from .llmapi import OnlineLLM

import simpidlog

load_dotenv(Path(__file__).with_name(".env"))

_ERROR_PREFIX = '@Simpidbit/agent_utils/llmapi.py\n'

async def deai(text: str) -> str:
    llm = OnlineLLM()

    llm_response = await llm.call_responses(
        system_prompt = 
'''
'''
    )
