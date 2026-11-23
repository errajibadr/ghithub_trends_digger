probes/decision_models/systemone_probe.py
----
"""Probe a decision model endpoint: System One contract, latency, and LangChain decision model compatibility.

Read-only and safe to rerun: the probe sends a handful of classification
requests about synthetic text and prints the answers. It never prints the API
key, and it sends no real data.

Prerequisites
    - Python 3.12. The contract checks use the standard library only.
    - The endpoint's base URL as callers reach it, without the ``/v1`` suffix
      (``http://<vllm-host>:8000``). A vLLM deployment serves a decision model
      on ``POST /v1/systemone``, the TypeSafe System One API.
    - The served model name (``--model``, as listed by ``GET /v1/models``).
    - The API key, when the server requires one, in ``PROBE_DECISION_API_KEY``.
    - Optional, for section 5: ``langchain-core>=1.8`` (``langchain_core.decisions``)
      and ``langchain-typesafe``. Missing packages skip the section, they do
      not fail the probe.

Ways to run
        export PROBE_DECISION_API_KEY=<key>        # only when the server requires one
        python systemone_probe.py --url http://<vllm-host>:8000 --model <served-model>

    More latency samples: ``--repeat 20``. A gateway certificate the machine does
    not trust: ``--no-ssl-verify``. Extra header (repeatable): ``--header Name=value``.

What it answers, section by section
    1. models: ``GET /v1/models`` answers and lists ``--model``
    2. contract: one request with a ``noul``, a ``choice`` and a ``score`` question
       returns one answer per question with the right type and value ranges
       (noul in [0, 1], choice among the criteria keys, score within the levels,
       probabilities summing to 1, confidence in [0, 1]), plus ``usage``;
       an object ``state`` is accepted as well as a string
    3. errors: an unknown question type is refused with a 4xx, not a 5xx
    4. latency: p50 and p95 over ``--repeat`` calls
    5. langchain: the raw response validates as ``langchain_core.decisions.ClassifierResponse``;
       ``TypeSafeClassifier`` pointed at the endpoint answers through ``invoke``
       and ``ainvoke``, and whether its async path is native or a thread

Exit code 0 when every check passed, 1 when one failed, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any


STATE = "My payouts have failed for three days and my customers are complaining. Please fix this today."
QUESTIONS: dict[str, Any] = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {
            "billing": "Payments, invoices, payouts or refunds",
            "technical": "Bugs, outages or integrations",
            "sales": "Pricing, upgrades or new accounts",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
    "is_urgent": {"type": "noul", "instructions": "Does the message express a time-sensitive need?"},
}
# What a sound model should answer on STATE; a mismatch is reported, not failed.
EXPECTED = {"department": "billing", "is_urgent_above": 0.5}

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f': {detail}' if detail else ''}")
    if not ok:
        failures.append(label)
    return ok


def note(label: str) -> None:
    print(f"  [INFO] {label}")


class Endpoint:
    def __init__(self, url: str, api_key: str | None, headers: dict[str, str], verify: bool, timeout: float) -> None:
        self.url = url.rstrip("/").removesuffix("/v1")
        self.headers = {"Content-Type": "application/json", **headers}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.ctx = None if verify else ssl._create_unverified_context()  # noqa: S323 - opt-in flag
        self.timeout = timeout

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any, float]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.url}{path}", data=data, headers=self.headers, method=method)
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.ctx) as resp:  # noqa: S310 - operator URL
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        elapsed = time.perf_counter() - start
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw[:200].decode(errors="replace")
        return status, payload, elapsed


def section_models(ep: Endpoint, model: str) -> None:
    print("\n1. models")
    try:
        status, body, _ = ep.call("GET", "/v1/models")
    except urllib.error.URLError as exc:
        check(False, "GET /v1/models reachable", str(exc.reason))
        return
    if not check(status == 200, "GET /v1/models answers 200", f"status {status}"):
        return
    ids = [m.get("id") for m in (body or {}).get("data", []) if isinstance(m, dict)]
    check(model in ids, f"model '{model}' is served", f"served: {ids}")


def validate_answers(body: Any, questions: dict[str, Any]) -> None:
    answers = body.get("answers") if isinstance(body, dict) else None
    if not check(isinstance(answers, dict), "response has an 'answers' object"):
        return
    check(set(answers) == set(questions), "one answer per question", f"got {sorted(answers)}")
    for name, question in questions.items():
        answer = answers.get(name)
        if not isinstance(answer, dict):
            continue
        kind = question["type"]
        if answer.get("type") == "refusal":
            check(False, f"{name}: answered, not refused", json.dumps(answer)[:200])
            continue
        check(answer.get("type") == kind, f"{name}: answer type is '{kind}'", f"got {answer.get('type')}")
        conf = answer.get("confidence")
        if conf is not None:
            check(0.0 <= conf <= 1.0, f"{name}: confidence in [0, 1]", f"{conf}")
        probs = answer.get("probabilities")
        if isinstance(probs, dict) and probs:
            total = sum(probs.values())
            check(math.isclose(total, 1.0, abs_tol=0.02), f"{name}: probabilities sum to 1", f"{total:.4f}")
        if kind == "noul":
            value = answer.get("noul")
            check(isinstance(value, int | float) and 0.0 <= value <= 1.0, f"{name}: noul is a probability", f"{value}")
        elif kind == "choice":
            check(answer.get("choice") in question["criteria"], f"{name}: choice is a criteria key", f"{answer.get('choice')}")
            if isinstance(probs, dict):
                check(set(probs) <= set(question["criteria"]), f"{name}: probabilities keyed by labels", f"{sorted(probs)}")
        elif kind == "score":
            value, top = answer.get("score"), len(question["criteria"]) - 1
            check(isinstance(value, int | float) and 0.0 <= value <= top, f"{name}: score within [0, {top}]", f"{value}")
            check(isinstance(answer.get("legend"), dict), f"{name}: score has a legend (LangChain requires it)", f"{answer.get('legend')}")
    usage = body.get("usage")
    check(isinstance(usage, dict) and "input_tokens" in usage, "response has usage.input_tokens", json.dumps(usage)[:120])
    note(f"other top-level fields: {sorted(k for k in body if k not in ('answers', 'usage'))}")


def section_contract(ep: Endpoint, model: str) -> dict[str, Any] | None:
    print("\n2. contract")
    status, body, elapsed = ep.call("POST", "/v1/systemone", {"model": model, "state": STATE, "questions": QUESTIONS})
    if not check(status == 200, "POST /v1/systemone answers 200", f"status {status}" + ("" if status == 200 else f", {json.dumps(body)[:300]}")):
        return None
    note(f"answered in {elapsed * 1000:.0f} ms")
    validate_answers(body, QUESTIONS)
    answers = body.get("answers", {})
    for name, answer in answers.items():
        value = {k: v for k, v in answer.items() if k in ("noul", "choice", "score", "confidence")}
        note(f"{name}: {value}")
    if answers.get("department", {}).get("choice") != EXPECTED["department"]:
        note(f"quality: department expected '{EXPECTED['department']}'")
    if (answers.get("is_urgent", {}).get("noul") or 0) <= EXPECTED["is_urgent_above"]:
        note("quality: is_urgent expected above 0.5")

    status, obj_body, _ = ep.call(
        "POST",
        "/v1/systemone",
        {"model": model, "state": {"customer_tier": "enterprise", "message": STATE}, "questions": {"is_urgent": QUESTIONS["is_urgent"]}},
    )
    check(status == 200 and "is_urgent" in (obj_body or {}).get("answers", {}), "object state accepted", f"status {status}")
    return body


def section_errors(ep: Endpoint, model: str) -> None:
    print("\n3. errors")
    bad = {"model": model, "state": STATE, "questions": {"x": {"type": "not_a_type", "instructions": "?"}}}
    status, body, _ = ep.call("POST", "/v1/systemone", bad)
    check(400 <= status < 500, "unknown question type refused with 4xx", f"status {status}, {json.dumps(body)[:200]}")


def section_latency(ep: Endpoint, model: str, repeat: int) -> None:
    print("\n4. latency")
    samples = []
    for _ in range(repeat):
        status, _, elapsed = ep.call("POST", "/v1/systemone", {"model": model, "state": STATE, "questions": QUESTIONS})
        if status == 200:
            samples.append(elapsed * 1000)
    if not check(len(samples) == repeat, f"{repeat} calls answered 200", f"{len(samples)} ok"):
        return
    p95 = statistics.quantiles(samples, n=20)[18] if len(samples) >= 2 else samples[0]
    note(f"3 questions per call: p50 {statistics.median(samples):.0f} ms, p95 {p95:.0f} ms, max {max(samples):.0f} ms")


def section_langchain(ep: Endpoint, model: str, raw: dict[str, Any] | None, api_key: str | None) -> None:
    print("\n5. langchain")
    try:
        from langchain_core.decisions import BaseDecisionModel, ClassifierResponse
    except ImportError:
        note("langchain_core.decisions not installed (needs langchain-core>=1.8): section skipped")
        return
    if raw is not None:
        try:
            parsed = ClassifierResponse.model_validate({"model": model, **raw})
            check(True, "raw response validates as ClassifierResponse", f"answers {sorted(parsed.answers)}")
        except Exception as exc:  # noqa: BLE001 - report any validation failure
            check(False, "raw response validates as ClassifierResponse", str(exc)[:300])
    try:
        from langchain_typesafe import TypeSafeClassifier
    except ImportError:
        note("langchain-typesafe not installed: classifier checks skipped")
        return
    from langchain_core.decisions import Choice, Noul, Score

    try:
        # The classifier refuses an empty key; a vLLM server without one ignores it.
        classifier = TypeSafeClassifier(model=model, base_url=ep.url, api_key=api_key or "unused", timeout=ep.timeout)
    except Exception as exc:  # noqa: BLE001
        check(False, "TypeSafeClassifier builds against the endpoint", str(exc)[:300])
        return
    kinds = {"noul": Noul, "choice": Choice, "score": Score}
    typed = {name: kinds[q["type"]](**{k: v for k, v in q.items() if k != "type"}) for name, q in QUESTIONS.items()}
    request = {"state": STATE, "questions": typed}
    try:
        result = classifier.invoke(request)
        check(set(result.answers) == set(QUESTIONS), "invoke answers every question", f"{sorted(result.answers)}")
        note(f"invoke: department={result.choices['department'].choice}, is_urgent={result.nouls['is_urgent'].noul:.3f}")
    except Exception as exc:  # noqa: BLE001
        check(False, "invoke answers", f"{type(exc).__name__}: {str(exc)[:300]}")
    try:
        result = asyncio.run(classifier.ainvoke(request))
        check(set(result.answers) == set(QUESTIONS), "ainvoke answers every question")
    except Exception as exc:  # noqa: BLE001
        check(False, "ainvoke answers", f"{type(exc).__name__}: {str(exc)[:300]}")
    native = type(classifier)._aclassify is not BaseDecisionModel._aclassify
    note(f"async path: {'native (_aclassify overridden)' if native else 'sync client in a thread'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="endpoint base URL, without /v1")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--repeat", type=int, default=5, help="latency samples (default 5)")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per request (default 30)")
    parser.add_argument("--header", action="append", default=[], help="extra header Name=value (repeatable)")
    parser.add_argument("--no-ssl-verify", action="store_true", help="skip TLS certificate verification")
    args = parser.parse_args()
    if args.repeat < 1 or any("=" not in h for h in args.header):
        parser.error("--repeat must be >= 1 and --header must be Name=value")
    headers = dict(h.split("=", 1) for h in args.header)
    api_key = os.environ.get("PROBE_DECISION_API_KEY") or None

    ep = Endpoint(args.url, api_key, headers, not args.no_ssl_verify, args.timeout)
    print(f"endpoint {ep.url}  model {args.model}  api key {'set' if api_key else 'not set'}")
    section_models(ep, args.model)
    raw = section_contract(ep, args.model)
    if raw is not None:
        section_errors(ep, args.model)
        section_latency(ep, args.model, args.repeat)
    section_langchain(ep, args.model, raw, api_key)

    print(f"\n{'all checks passed' if not failures else f'{len(failures)} failed: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

-------

