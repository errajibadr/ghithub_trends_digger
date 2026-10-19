scripts/guard_bench_client.py
----
"""Prompt-injection guard benchmark: one markdown table row per judge model.

Run from the repository root with the guard judge configured in the environment
(ORCHESTRATOR_PROMPT_INJECTION_GUARD_PROVIDER / _BASE_URL / _API_KEY / _REASONING_EFFORT, or .env).
GUARD_BENCH_MODELS lists the models to compare; unset, the configured _MODEL is used.

    GUARD_BENCH_MODELS=model-a,model-b GUARD_BENCH_REPEATS=3 uv run scripts/guard_bench_client.py
"""

import asyncio
import importlib.util
import os
import statistics
import time

from dotenv import load_dotenv

load_dotenv()
spec = importlib.util.spec_from_file_location("harness", "tests/test_ai_engine/agents/orchestrator/test_prompt_extraction_guard_recall.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

REPEATS = int(os.getenv("GUARD_BENCH_REPEATS", "3"))


async def bench(model_name: str | None) -> str:
    settings = h.PromptInjectionGuardSettings(**({"model": model_name} if model_name else {}))
    mw = h.PromptInjectionGuardMiddleware(settings=settings)
    lat, errors = [], []
    classify = mw._classify_async

    async def timed(messages, model):
        start = time.perf_counter()
        try:
            return await classify(messages, model)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            raise
        finally:
            lat.append(time.perf_counter() - start)

    mw._classify_async = timed
    rows = h._load_labeled()
    await h._guard_outcome(mw, rows[0])  # warm-up, not counted
    lat.clear()
    errors.clear()
    outcomes = {r["qid"]: [await h._guard_outcome(mw, r) for _ in range(REPEATS)] for r in rows}
    attacks = [o for r in rows if r["must_block"] for o in outcomes[r["qid"]]]
    legit = [o for r in rows if not r["must_block"] for o in outcomes[r["qid"]]]
    benign = [o for r in rows if r["is_benign"] for o in outcomes[r["qid"]]]
    unstable = [q for q, o in outcomes.items() if len(set(o)) > 1]
    missed = [r["qid"] for r in rows if r["must_block"] and outcomes[r["qid"]].count("block") < REPEATS]
    for e in errors[:3]:
        print(f"[{settings.model}] {e}")
    blocks = attacks.count("block")
    return (
        f"| {settings.model} | {settings.reasoning_effort or '-'} | {blocks / len(attacks):.0%} ({blocks}/{len(attacks)}) "
        f"| {legit.count('block')}/{len(legit)} | {benign.count('advise')}/{len(benign)} | {len(errors)}/{len(lat)} "
        f"| {', '.join(unstable) or 'yes'} | {statistics.median(lat):.2f} | {statistics.quantiles(lat, n=20)[18]:.2f} "
        f"| {', '.join(missed) or '-'} |"
    )


async def main() -> None:
    models = [m.strip() for m in os.getenv("GUARD_BENCH_MODELS", "").split(",") if m.strip()] or [None]
    lines = [await bench(m) for m in models]
    print(f"\nGuard benchmark, {REPEATS} passes. Failed calls are unscreened turns (the guard is fail-open).\n")
    print("| Judge | Reasoning | Recall | False blocks | Advisory on benign | Failed calls | Stable (else flipping rows) | p50 s | p95 s | Missed attacks |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(lines))


asyncio.run(main())

-------

