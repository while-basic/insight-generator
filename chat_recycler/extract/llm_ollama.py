from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import requests

from chat_recycler.extract.schemas import EventRow, VariableRow
from chat_recycler.utils.hashing import sha256_text


class OllamaError(RuntimeError):
    pass


def _ollama_payload(model: str, prompt: str) -> Dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }


def call_ollama(model: str, prompt: str) -> Dict[str, Any]:
    response = requests.post("http://localhost:11434/api/generate", json=_ollama_payload(model, prompt), timeout=60)
    if response.status_code != 200:
        raise OllamaError(f"Ollama error: {response.status_code}")
    return response.json()


def build_prompt(text: str) -> str:
    return (
        "Extract events and variables from the conversation. "
        "Return strict JSON with keys 'events' and 'variables' "
        "that match the EventRow and VariableRow schemas.\n\n"
        f"Conversation:\n{text}"
    )


def extract_with_ollama(text: str, model: str) -> Tuple[Optional[List[EventRow]], Optional[List[VariableRow]], Dict[str, str]]:
    prompt = build_prompt(text)
    receipt = {
        "model": model,
        "prompt_hash": sha256_text(prompt),
        "response_hash": "",
    }
    payload = call_ollama(model, prompt)
    response_text = payload.get("response")
    if not response_text:
        raise OllamaError("Empty response from Ollama")
    receipt["response_hash"] = sha256_text(response_text)
    data = json.loads(response_text)
    events = [EventRow(**item) for item in data.get("events", [])]
    variables = [VariableRow(**item) for item in data.get("variables", [])]
    return events, variables, receipt
