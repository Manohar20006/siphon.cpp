"""High-level decisions on top of llama-server's /decide.

A request names the options; Siphon owns the prompt so every caller gets the validated layout
(2026-10-12 tool-selection benchmark: bit-exact vs the unoptimized path, ~0.6 s per decision):

    system : decision-function instruction
    user   : Options:            <- shared part: computed once on the server and reused
             A. name: description
             ...
             State:
             <context>
             <input>             <- per request
             Question: <question>
             Answer with the letter only.

The shared part is handed to the server as `prefix_tokens`, aligned down to the 24-token prompt
chunk (the server reuses a saved prefix only when the prompt covers the chunk holding its end).
Probabilities are a softmax over the option letters.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

SYSTEM = ("You are a decision function. You will be given a state and one question. "
          "Reply with the answer label only: no words, no punctuation, no explanation.")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROMPT_CHUNK = 24
DEFAULT_THRESHOLD = 0.9
_PLACEHOLDER = "⁣SIPHON_INPUT⁣"


class DecisionError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class _Layout:
    head: str             # rendered prompt text before the input
    tail: str             # rendered prompt text after the input
    prefix_tokens: int
    names: list[str]


def _post(base_url: str, path: str, body: dict, api_key: str = "", timeout: float = 600.0) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + path, json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        if 400 <= exc.code < 500:   # e.g. the prompt is longer than the decision context: the caller's to fix
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            raise DecisionError(400, f"request rejected by the decision engine: {detail}") from exc
        raise DecisionError(502, f"decision engine returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        raise DecisionError(503, f"decision engine unreachable: {exc}") from exc


def parse_request(body: dict) -> tuple[str, str, list[tuple[str, str]], str, float]:
    """Validate a /v1/decide body -> (input, question, options, context, threshold)."""
    if not isinstance(body, dict):
        raise DecisionError(400, "body must be a JSON object")
    text = body.get("input")
    question = body.get("question")
    if not isinstance(text, str) or not text.strip():
        raise DecisionError(400, '"input" (non-empty string) is required')
    if not isinstance(question, str) or not question.strip():
        raise DecisionError(400, '"question" (non-empty string) is required')
    raw = body.get("options")
    if not isinstance(raw, list) or not 2 <= len(raw) <= len(LETTERS):
        raise DecisionError(400, f'"options" must be a list of 2 to {len(LETTERS)} options')
    options: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            name, desc = item, ""
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            name, desc = item["name"], item.get("description", "")
            if not isinstance(desc, str):
                raise DecisionError(400, "option descriptions must be strings")
        else:
            raise DecisionError(400, 'each option must be a string or {"name": ..., "description": ...}')
        name = name.strip()
        if not name or "\n" in name or "\n" in desc:
            raise DecisionError(400, "option names must be non-empty single-line strings")
        options.append((name, desc.strip()))
    if len({n for n, _ in options}) != len(options):
        raise DecisionError(400, "option names must be unique")
    context = body.get("context", "")
    if not isinstance(context, str):
        raise DecisionError(400, '"context" must be a string')
    threshold = body.get("threshold", DEFAULT_THRESHOLD)
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise DecisionError(400, '"threshold" must be a number between 0 and 1')
    return text, question.strip(), options, context.strip(), float(threshold)


class DecisionEngine:
    """Talks to one decision-capable llama-server. Thread-safe; the server serializes decisions."""

    def __init__(self, base_url: str, backend_key: str = "", alias: str = ""):
        self.base_url = base_url
        self.backend_key = backend_key
        self.alias = alias
        self._lock = threading.Lock()
        self._layouts: dict[str, _Layout] = {}
        self._label_ids: list[int] | None = None

    # -- setup ---------------------------------------------------------------
    def _letters(self) -> list[int]:
        if self._label_ids is None:
            ids = []
            for letter in LETTERS:
                toks = _post(self.base_url, "/tokenize", {"content": letter, "add_special": False,
                                                          "parse_special": False}, self.backend_key)["tokens"]
                if len(toks) != 1:
                    raise DecisionError(500, f"option letter {letter!r} is not a single token for this model")
                ids.append(toks[0])
            self._label_ids = ids
        return self._label_ids

    def _render(self, user: str) -> str:
        return _post(self.base_url, "/apply-template", {
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False}}, self.backend_key)["prompt"]

    def _tokens(self, prompt: str) -> list[int]:
        return _post(self.base_url, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True},
                     self.backend_key)["tokens"]

    @staticmethod
    def _user(options, context: str, text: str, question: str) -> str:
        listed = "\n".join(f"{LETTERS[i]}. {n}: {d}" if d else f"{LETTERS[i]}. {n}" for i, (n, d) in enumerate(options))
        state = f"{context}\n{text}" if context else text
        return f"Options:\n{listed}\n\nState:\n{state}\n\nQuestion: {question}\nAnswer with the letter only."

    def _layout(self, options, context: str, question: str) -> _Layout:
        key = hashlib.sha256(json.dumps([options, context, question]).encode()).hexdigest()
        layout = self._layouts.get(key)
        if layout is not None:
            return layout
        rendered = self._render(self._user(options, context, _PLACEHOLDER, question))
        if rendered.count(_PLACEHOLDER) != 1:
            raise DecisionError(500, "chat template altered the request text; cannot build a stable prompt")
        head, tail = rendered.split(_PLACEHOLDER)
        # shared prefix = tokens common to two different inputs, minus the boundary token, chunk-aligned
        a = self._tokens(head + "a" + tail)
        b = self._tokens(head + "Z" + tail)
        common = 0
        for x, y in zip(a, b):
            if x != y:
                break
            common += 1
        prefix = max(0, common - 1) // PROMPT_CHUNK * PROMPT_CHUNK
        layout = _Layout(head, tail, prefix, [n for n, _ in options])
        if len(self._layouts) > 64:
            self._layouts.clear()
        self._layouts[key] = layout
        return layout

    # -- API -----------------------------------------------------------------
    def decide(self, body: dict) -> dict:
        t0 = time.perf_counter()
        text, question, options, context, threshold = parse_request(body)
        with self._lock:
            labels = self._letters()[:len(options)]
            layout = self._layout(options, context, question)
        r = _post(self.base_url, "/decide", {"prompt": layout.head + text + layout.tail, "token_ids": labels,
                                             "prefix_tokens": layout.prefix_tokens}, self.backend_key)
        logprobs = r["logprobs"]
        top = max(logprobs)
        weights = [math.exp(x - top) for x in logprobs]
        total = sum(weights)
        probs = {name: w / total for name, w in zip(layout.names, weights)}
        choice = max(probs, key=probs.get)
        return {
            "object": "decision",
            "model": self.alias,
            "choice": choice,
            "confidence": probs[choice],
            "confident": probs[choice] >= threshold,
            "threshold": threshold,
            "probabilities": probs,
            "usage": {"prompt_tokens": r.get("prompt_tokens"), "cached_tokens": r.get("prefix_tokens"),
                      "cache_hit": r.get("prefix_hit")},
            "timing": {"engine_ms": r.get("decide_ms"), "total_ms": (time.perf_counter() - t0) * 1000},
        }

    def warmup(self, body: dict) -> dict:
        """Build the shared prefix for an option set ahead of real traffic (first decision is ~3 s)."""
        body = dict(body)
        body.setdefault("input", "warmup")
        result = self.decide(body)
        return {"object": "decision.warmup", "model": self.alias, "warmed": True,
                "timing": result["timing"], "cached_tokens": result["usage"]["cached_tokens"]}


def backend_key_from_env() -> str:
    return os.environ.get("SIPHON_BACKEND_API_KEY", "")
