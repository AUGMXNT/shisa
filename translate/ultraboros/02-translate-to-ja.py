from functools import lru_cache
from google.auth.transport import requests as google_requests
from google.oauth2 import service_account
from loguru import logger
from typing import Dict, Any, List
import aiohttp
import asyncio
import backoff
import copy
import json
import hashlib
import os
import sqlite3
import sys
import time

# Max concurrency in calls to bison API.
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = 'https://api.openai.com/v1/chat/completions'

# Prompt template.
TEMPLATE = """
Translate the following to Japanese:

{text}

The text translated to Japanese is:
"""

# GCP variables.
GCP_PROJECT = os.getenv("GCP_PROJECT", "replaceme")
BISON_URL = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT}/locations/us-central1/publishers/google/models/text-bison-32k:predict"

# Retry exceptions, which we can retry.
class RetriableError(RuntimeError):
    ...

@lru_cache()
def _get_vertexai_token(_):
    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/cloud-platform.read-only",
    ]
    path = os.environ["VERTEXAI_CREDENTIALS_PATH"]
    credentials = service_account.Credentials.from_service_account_file(
        path, scopes=scopes
    )
    credentials.refresh(google_requests.Request())
    return credentials.token

def get_vertexai_token():
    return _get_vertexai_token(round(time.time() / 300))

async def handle_bison_error(result):
    text = await result.text()
    logger.error(f"Error querying bison: {result.status}: {text}")
    code, status = None, None
    try:
        body = await result.json()
        code = body["error"].get("code")
        status = body["error"].get("status")
    except Exception:
        await asyncio.sleep(1)
        ...
    if code == 429:
        await asyncio.sleep(3)
        raise RetriableError(text)
    raise Exception(f"Error querying bison [{code}]: {text}")

@backoff.on_exception(backoff.expo, (RetriableError,))
async def post_bison(body: Dict[str, Any]):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            BISON_URL, json=body, headers={"Authorization": f"Bearer {get_vertexai_token()}"}
        ) as result:
            if result.status != 200:
                await handle_bison_error(result)
            data = await result.json()
            if data['predictions'][0].get("safetyAttributes", {}).get("blocked"):
                raise Exception("Response blocked by vertex.")
            return data

async def handle_openai_error(result):
    text = await result.text()
    logger.error(f"Error querying OpenAI: {result.status}: {text}")
    if "too many requests" in text.lower():
        raise RetriableError(text)
    if "rate limit reached" in text.lower():
        raise RetriableError(text)
    elif "context_length_exceeded" in text.lower():
        raise Exception(text)
    elif "server_error" in text and "overloaded" in text.lower():
        raise RetriableError(text)
    elif "bad gateway" in text.lower() or "server_error" in text.lower():
        raise RetriableError(text)
    else:
        raise Exception(text)

@backoff.on_exception(backoff.expo, (RetriableError,))
async def post_openai(body: Dict[str, Any]):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with aiohttp.ClientSession() as session:
        async with session.post(OPENAI_URL, json=body, headers=headers) as result:
            if result.status != 200:
                await handle_openai_error(result)
            return await result.json()

# Translate a single input text, which handles posting to bison and extracting response.
async def translate_bison(text: str) -> str:
    params = {
        "temperature": 0.05,
        "maxDecodeSteps": 8192,
        "topK": 40,
        "topP": 0.8,
    }
    body = {"instances": [{"content": TEMPLATE.format(text=text)}], "parameters": params}
    try:
        result = await post_bison(body)
    except Exception as exc:
        logger.warning(f"Translation fail: {exc}")
        raise
    return 'bison', result["predictions"][0]["content"].strip()

async def translate_openai(text: str) -> str:
    body = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": TEMPLATE.format(text=text)},
        ],
        "temperature": 0.1,
    }
    result = await post_openai(body)
    return 'gpt-4', result["choices"][0]["message"]["content"]

async def translate(text: str) -> str:
    try:
        return await translate_bison(text)
    except Exception as exc:
        logger.warning(f'Bison translation fail: {exc}')
        try:
            return await translate_openai(text)
        except Exception as exc2:
            logger.error(f'Bison and OpenAI fail: {exc2}')
    return None, None

async def translate_paragraph(conn: Any, paragraph: str) -> str:
    cursor = conn.cursor()
    id_ = hashlib.md5(paragraph.encode()).hexdigest()
    logger.debug(f"Translating {id_}...")

    # Check the cache.
    cursor.execute("SELECT translator, prompt_ja FROM prompts WHERE prompt_en = ?", (paragraph,))
    row = cursor.fetchone()
    if row is not None:
        translator = row[0]
        value_ja = row[1]
    else:
        translator, value_ja = await translate(paragraph)
        if value_ja:
            cursor.execute("INSERT INTO prompts (prompt_en, prompt_ja, translator) VALUES (?, ?, ?) ON CONFLICT(prompt_en) DO NOTHING", (paragraph, value_ja, translator))
            conn.commit()
    if not value_ja:
        return None
    logger.info(f"  [{translator}]: {paragraph}")
    logger.success(f"  {value_ja}")
    return value_ja

# Translate a single conversation, which will have multiple text blocks to translate.
async def translate_turns(conn: Any, id_: str, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    cursor = conn.cursor()
    logger.debug(f"Translating {id_}...")
    translated = []
    for turn in turns:
        translated.append(copy.deepcopy(turn))

        # Split into paragraphs so bison is less likely to repeat.
        value = turn.get('content', turn.get('value'))
        if not value:
            return
        paragraphs = [value]
        if len(value) >= 2048:
            paragraphs = value.split('\n\n')

        # Translate.
        paragraphs_ja = []
        for paragraph in paragraphs:
            paragraph_ja = await translate_paragraph(conn, paragraph)
            if not paragraph_ja:
                return
            paragraphs_ja.append(paragraph_ja)
        value_ja = "\n\n".join(paragraphs_ja)
        #logger.info(f"  {value}")
        #logger.success(f"  {value_ja}")
        if 'role' in translated[-1]:
            translated[-1]['from'] = translated[-1].pop('role')
        translated[-1]["value"] = value_ja
        translated[-1].pop('content', None)
    translated_as_json = json.dumps(translated)
    cursor.execute("UPDATE ultraboros SET conversation_ja = ? WHERE id = ?", (translated_as_json, id_))
    conn.commit()

async def main():
    # Connect to the SQLite database
    conn = sqlite3.connect("ultraboros.db")

    # Create a cursor object
    c = conn.cursor()
    c.execute("SELECT id, conversation FROM ultraboros WHERE conversation_ja IS NULL")
    rows = c.fetchall()

    # Iterate through the dataset, using asyncio tasks for max throughput.
    tasks = []
    async def _concurrency_check():
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=0.0)
        while len(pending) >= MAX_CONCURRENCY:
            await asyncio.sleep(0.1)
            _, pending = await asyncio.wait(tasks, timeout=0.0)
    for row in rows:
        turns = json.loads(row[1])
        await _concurrency_check()
        tasks.append(asyncio.create_task(translate_turns(conn, row[0], turns)))
    await asyncio.wait(tasks)
    conn.close()

asyncio.run(main())
