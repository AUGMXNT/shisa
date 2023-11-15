from data_selection import HashedNgramDSIR
from loguru import logger
from tqdm import tqdm
import aiohttp
import asyncio
import datasets
import glob
import json
import os
import sqlite3
import uuid
from shisa_utils import (
    translate_turns,
    MAX_CONCURRENCY,
)


def sample_slimorca(conn):
    if os.path.isdir("orca-sampled"):
        logger.info("Already sampled, to restart delete the `orca-sampled` directory.")
        return
    logger.info("Loading slim orca...")
    ds = datasets.load_dataset("Open-Orca/SlimOrca")
    with open("slimorca/slim.jsonl", "w") as outfile:
        for item in ds["train"]:
            val = json.dumps(item["conversations"])
            first_response = list(
                filter(
                    lambda x: x.get("from", x.get("role")) in ("gpt", "assistant"),
                    item["conversations"],
                )
            )[0]
            outfile.write(
                json.dumps(
                    {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_OID, val)),
                        "conversations": item["conversations"],
                        "text": first_response.get(
                            "value", first_response.get("content")
                        ),
                    }
                )
                + "\n"
            )

    # Create smaller sample with DSIR
    logger.info("Creating sample...")
    with open("sample.jsonl", "a+") as outfile:
        ...
    orca_datasets = glob.glob("slimorca/*.jsonl")
    dsir = HashedNgramDSIR(orca_datasets, ["sample.jsonl"], cache_dir=".cache/dsir")
    dsir.fit_importance_estimator(num_tokens_to_fit="auto")
    dsir.compute_importance_weights()
    dsir.resample(
        out_dir="orca-sampled", num_to_sample=10000, cache_dir=".cache/resampled"
    )


def populate_db(conn):
    slim_orca = datasets.concatenate_datasets(
        [
            datasets.Dataset.from_json(path)
            for path in glob.glob("orca-sampled/*.jsonl")
            if os.stat(path).st_size
        ]
    )

    # Populate the sqlite DB.
    c = conn.cursor()
    skipped = 0
    for item in tqdm(slim_orca):
        prompt_parts = list(
            filter(
                lambda x: x.get("from", x.get("role")) in ("user", "human", "system"),
                item["conversations"],
            )
        )
        prompt = "\n".join(
            [part.get("value", part.get("content")) for part in prompt_parts]
        ).lower()
        if "translat" in prompt or len(prompt) / len(prompt.encode()) <= 0.95:
            skipped += 1
            continue
        c.execute("SELECT id FROM ultraboros WHERE id = ?", (item["id"],))
        row = c.fetchone()
        if not row:
            c.execute(
                "INSERT INTO ultraboros (id, source, conversation) VALUES (?, ?, ?)",
                (item["id"], "slimorca", json.dumps(item["conversations"])),
            )
            conn.commit()
    logger.warning(f"Skipped {skipped} translation tasks...")


async def main():
    # Load the items that have not been translated yet.
    conn = sqlite3.connect("ultraboros.db")
    sample_slimorca(conn)
    populate_db(conn)
    c = conn.cursor()
    c.execute(
        """
CREATE TABLE IF NOT EXISTS refusal_check (
    id VARCHAR PRIMARY KEY,
    response TEXT,
    refusal BOOLEAN
);
"""
    )

    c.execute(
        "SELECT id, conversation FROM ultraboros WHERE source = 'slimorca' AND conversation_ja IS NULL"
    )
    rows = c.fetchall()
    if not rows:
        return
    logger.info(f"Need to translate {len(rows)} rows from slimorca...")

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
        for item in rows:
            await _concurrency_check()
            conv = json.loads(item[1])
            tasks.append(
                asyncio.create_task(translate_turns(conn, item[0], conv, client))
            )
        if tasks:
            await asyncio.wait(tasks)
    conn.close()


asyncio.run(main())
