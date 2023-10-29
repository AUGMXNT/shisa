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
import os
import sqlite3
import sys
import time

# Max concurrency in calls to bison API.
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))

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
class RateLimitError(RuntimeError):
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
        raise RateLimitError(text)
    raise Exception(f"Error querying bison [{code}]: {text}")

@backoff.on_exception(backoff.expo, (RateLimitError,))
async def post_bison(body: Dict[str, Any]):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            BISON_URL, json=body, headers={"Authorization": f"Bearer {get_vertexai_token()}"}
        ) as result:
            if result.status != 200:
                await handle_bison_error(result)
            data = await result.json()
            if data.get("safetyAttributes", {}).get("blocked"):
                raise Exception("Response blocked by vertex.")
            return data

# Translate a single input text, which handles posting to bison and extracting response.
async def translate(text: str) -> str:
    params = {
        "temperature": 0.1,
        "maxDecodeSteps": 8192,
        "topK": 40,
        "topP": 0.8,
    }
    body = {"instances": [{"content": TEMPLATE.format(text=text)}], "parameters": params}
    try:
        result = await post_bison(body)
    except Exception as exc:
        logger.warning(f"Translation fail: {exc}")
        return None
    return result["predictions"][0]["content"].strip()

# Translate a single conversation, which will have multiple text blocks to translate.
async def translate_turns(conn: Any, id_: str, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    cursor = conn.cursor()
    logger.debug(f"Translating {id_}...")
    translated = []
    for turn in turns:
        translated.append(copy.deepcopy(turn))

        # Check the cache.
        cursor.execute("SELECT prompt_ja FROM prompts WHERE prompt_en = ?", (turn["value"],))
        row = cursor.fetchone()
        if row is not None:
            value_ja = row[0]
        else:
            value_ja = await translate(turn["value"])
            if value_ja:
                cursor.execute("INSERT INTO prompts (prompt_en, prompt_ja) VALUES (?, ?) ON CONFLICT(prompt_en) DO NOTHING", (turn["value"], value_ja))
                conn.commit()
        if not value_ja:
            return
        logger.info(f"  {turn['value']}")
        logger.success(f"  {value_ja}")
        translated[-1]["value"] = value_ja
    translated_as_json = json.dumps(translated)
    cursor.execute("UPDATE airoboros_31 SET conversation_ja = ?, translator = ? WHERE id = ?", (translated_as_json, "text-bison-32k", id_))
    conn.commit()

async def main():
    # Connect to the SQLite database
    conn = sqlite3.connect("airoboros_bison.db")

    # Create a cursor object
    c = conn.cursor()
    c.execute("SELECT id, category, conversation FROM airoboros_31 WHERE category != 'mathjson' AND conversation_ja IS NULL")
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
    count = 0
    for row in rows:
        turns = json.loads(row[2])
        await _concurrency_check()
        tasks.append(asyncio.create_task(translate_turns(conn, row[0], turns)))
        count += 1
        if count > 8:
            break
    await asyncio.wait(tasks)
    conn.close()

asyncio.run(main())
