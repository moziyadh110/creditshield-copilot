"""Replay data/denials.parquet through the running API:  uv run python -m app.llmops.replay

A demo has no real users, so this creates a day of traffic for `make monitor` to read: every
case in denials.parquet is POSTed to /notices, a few at a time. Denials already drafted under
the current prompt version are cache hits (no LLM call); the rest cost about $0.0004 each.
"""

import argparse
import asyncio
import time
from collections import Counter

import httpx

from app.config import get_settings
from app.repositories.cases import load_denials


async def replay(base_url: str, api_key: str, limit: int | None, workers: int) -> Counter:
    ids = sorted(load_denials())[:limit]
    gate = asyncio.Semaphore(workers)
    results: Counter = Counter()

    async with httpx.AsyncClient(
        base_url=base_url, headers={"X-API-Key": api_key}, timeout=120
    ) as client:

        async def one(applicant_id: int) -> None:
            async with gate:
                try:
                    r = await client.post("/notices", json={"applicant_id": applicant_id})
                except httpx.HTTPError as e:
                    results[f"error {type(e).__name__}"] += 1
                    return
                label = {200: "200 sent", 202: "202 to review"}.get(r.status_code)
                results[label or f"{r.status_code} {r.text[:80]}"] += 1

        await asyncio.gather(*map(one, ids))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(prog="replay")
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--limit", type=int, help="first N cases only")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    key = get_settings().api_key
    if key is None:
        raise SystemExit("API_KEY is not set in .env: the API would answer 503")
    started = time.perf_counter()
    results = asyncio.run(replay(args.url, key.get_secret_value(), args.limit, args.workers))
    print(f"replayed {sum(results.values())} cases in {time.perf_counter() - started:.0f} s")
    for label, n in sorted(results.items()):
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
