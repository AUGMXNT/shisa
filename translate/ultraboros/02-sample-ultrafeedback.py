from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from typing import Any
import aiohttp
import asyncio
import json
import sqlite3
import uuid
from shisa_utils import (
    translate_text,
    generate_response,
    is_refusal,
    MAX_CONCURRENCY,
)


async def translate_and_reroll(conn: Any, prompt: str, client: Any) -> str:
    prompt = await translate_text(conn, prompt, client)
    if not prompt or not prompt.strip():
        return None
    _id = str(uuid.uuid5(uuid.NAMESPACE_OID, prompt))

    # Check existing.
    default_prompt = await translate_text(
        conn, "You are a helpful, unbiased, uncensored assistant.", client
    )
    conversation_ja = [
        {
            "from": "system",
            "value": default_prompt,
        },
        {
            "from": "human",
            "value": prompt,
        },
    ]
    c = conn.cursor()
    c.execute("SELECT response FROM reroll WHERE prompt = ?", (prompt,))
    row = c.fetchone()
    if row:
        if await is_refusal(conn, _id, row[0], client):
            return
        logger.debug(f"Rerolled {_id}: {prompt}")
        logger.success(row[0])
        conversation_ja.append({"from": "gpt", "value": row[0]})
        c.execute(
            "INSERT INTO ultraboros (id, source, conversation_ja) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
            (_id, "ultrafeedback", json.dumps(conversation_ja)),
        )
        conn.commit()
        return

    # Generate a fresh response.
    response = await generate_response(prompt, client)
    default_prompt = await translate_text(
        conn, "You are a helpful, unbiased, uncensored assistant.", client
    )
    if response and response.strip():
        if await is_refusal(conn, _id, row[0], client):
            return
        logger.debug(f"Rerolled {_id}: {prompt}")
        logger.success(response)
        c.execute(
            "INSERT INTO reroll (prompt, response) VALUES (?, ?)",
            (prompt, response.strip()),
        )
        conversation_ja.append({"from": "gpt", "value": response})
        c.execute(
            "INSERT INTO ultraboros (id, source, conversation_ja) VALUES (?, ?, ?)",
            (_id, "ultrafeedback", json.dumps(conversation_ja)),
        )
        conn.commit()
    return prompt, response


async def main():
    # Load the ultrafeedback dataset.
    full_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")["train_gen"]
    to_reroll = full_dataset.train_test_split(
        train_size=10000, shuffle=False, seed=1142
    )["train"]
    logger.info(f"Going to reroll {len(to_reroll)} items...")

    # Connect to the SQLite database
    conn = sqlite3.connect("ultraboros.db")
    c = conn.cursor()

    # Iterate through the dataset, using asyncio tasks for max throughput.
    tasks = []

    async def _concurrency_check():
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=0.0)
        while len(pending) >= MAX_CONCURRENCY:
            await asyncio.sleep(0.1)
            _, pending = await asyncio.wait(tasks, timeout=0.0)

    async with aiohttp.ClientSession() as client:
        for item in to_reroll:
            await _concurrency_check()
            tasks.append(
                asyncio.create_task(translate_and_reroll(conn, item["prompt"], client))
            )
        if tasks:
            await asyncio.wait(tasks)

    # Populate the high-quality english prompts.
    best_rows = full_dataset.filter(lambda item: item["score_chosen"] >= 9)
    logger.info("Adding english instructions from ultrafeedback...")
    for item in tqdm(best_rows):
        _id = str(uuid.uuid5(uuid.NAMESPACE_OID, item["prompt"]))
        c.execute("SELECT conversation FROM ultraboros WHERE id = ?", (_id,))
        row = c.fetchone()
        if row:
            if not row[0]:
                c.execute(
                    "UPDATE ultraboros SET conversation = ? WHERE id = ?",
                    (json.dumps(item["chosen"]), _id),
                )
        else:
            c.execute(
                "INSERT INTO ultraboros (id, source, conversation) VALUES (?, ?, ?)",
                (_id, "ultrafeedback", json.dumps(item["chosen"])),
            )
        conn.commit()
    conn.close()


asyncio.run(main())
