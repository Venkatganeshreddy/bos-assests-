from __future__ import annotations

import requests

SYSTEM_PROMPT = """You are the BOS Assets Chatbot.

You answer questions only from the supplied BOS asset library excerpts.

Rules:
- If the sources do not contain the answer, say that plainly.
- Cite factual statements inline with source tags like [Source 1].
- Use the exact file names and paths from the context when you reference assets.
- If a source file name or path clearly matches the user's request, prioritize that source instead of saying the answer is unavailable.
- Do not claim something is missing when a matching file is present in the retrieved context.
- When the user asks for copy, messaging, or drafts, you may synthesize a proposal from the BOS materials, but label it clearly as a proposed draft and still cite the source blocks that informed it.
- Do not invent missing campaign names, dates, file contents, or links.
- Keep answers structured and practical.

Context:
{context}
"""


def ask_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    context: str,
) -> str:
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing.")

    if not context.strip():
        return "I could not find relevant BOS assets for that question in the indexed folder."

    payload_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    payload_messages.extend(messages)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": payload_messages,
            "temperature": 0.2,
            "max_tokens": 1400,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:300]}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
