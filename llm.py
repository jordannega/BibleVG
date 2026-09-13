import json
import os
from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, LLM_PROVIDER


def _call_anthropic(system: str, user: str, max_tokens: int = 400) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _call_openai(system: str, user: str, max_tokens: int = 400) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def call_llm(system: str, user: str, max_tokens: int = 400) -> str:
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        return _call_anthropic(system, user, max_tokens)
    if OPENAI_API_KEY:
        return _call_openai(system, user, max_tokens)
    raise RuntimeError("No LLM provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")


SELECT_SYSTEM = """You are a Bible verse selector. Given a user's situation and up to 8 candidate verses, choose the ONE that best fits.

Rules:
- Choose only from the candidates.
- Prefer pastoral fit over fame.
- Return JSON only: {"verse_id": <int>, "reason": "<short>"}"""


REFLECT_SYSTEM = """You are a warm, humble pastoral companion. Write a short reflection (2-4 sentences, under 80 words).

Rules:
- First sentence: acknowledge the situation with empathy.
- Do not quote any other verse.
- Do not speak for God.
- Do not promise outcomes ("God will heal you").
- No medical, legal, or financial advice.
- End with one gentle encouragement. No prayer unless asked."""


def select_verse(user_prompt: str, intent_theme: str, candidates: list[dict]) -> dict:
    candidate_block = "\n".join(
        f'{c["id"]}. "{c["text"]}" — {c["reference"]}' for c in candidates
    )
    user_msg = f"""User situation: {user_prompt}
Theme: {intent_theme}

Candidates:
{candidate_block}

Return JSON."""
    raw = call_llm(SELECT_SYSTEM, user_msg, max_tokens=200)
    # strip code fences if any
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except Exception:
        return {"verse_id": candidates[0]["id"], "reason": "fallback"}
    return data


def reflect(user_prompt: str, theme: str, verse: dict, translation: str) -> str:
    user_msg = f"""User situation: {user_prompt}
Theme: {theme}
Chosen verse: "{verse['text']}" ({verse['reference']}, {translation})

Write the reflection."""
    return call_llm(REFLECT_SYSTEM, user_msg, max_tokens=250).strip()