from __future__ import annotations

import re

import requests

SYSTEM_PROMPT = """You are the BOS Assets Chatbot.

You answer questions only from the supplied BOS asset library excerpts.

Rules:
- If the sources do not contain the answer, say that plainly.
- Cite factual statements inline with source tags like [Source 1].
- Every factual sentence must include at least one valid source tag.
- Only cite sources that exist in the provided context.
- Use the exact file names and paths from the context when you reference assets.
- If a source file name or path clearly matches the user's request, prioritize that source instead of saying the answer is unavailable.
- Do not claim something is missing when a matching file is present in the retrieved context.
- When the user asks for copy, messaging, or drafts, you may synthesize a proposal from the BOS materials, but label it clearly as a proposed draft and still cite the source blocks that informed it.
- Do not invent missing campaign names, dates, file contents, or links.
- Keep answers structured and practical.
- If you cannot support a claim from context, explicitly say you cannot confirm it from indexed sources.

Context:
{context}
"""


def ask_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    context: str,
    source_count: int,
) -> str:
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing.")

    if not context.strip():
        return "I could not find relevant BOS assets for that question in the indexed folder."

    payload_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    payload_messages.extend(messages)
    payload_messages.append(
        {
            "role": "user",
            "content": (
                "Return only source-grounded content. "
                "Use inline tags like [Source 1]. "
                "Do not use any Source index above "
                f"{source_count}. "
                "If the user asks for tabular rows (for example curriculum, semester, or mapped-format data), "
                "return a markdown table directly from cited rows and do not invent rows."
            ),
        }
    )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": payload_messages,
            "temperature": 0,
            "max_tokens": 1400,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:300]}")

    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()
    if _is_valid_grounded_answer(answer, source_count):
        return answer

    return (
        "I can only answer from indexed files with exact citations. "
        "I could not generate a fully source-grounded answer for this query. "
        "Please refine the question and I will respond only with verifiable sources."
    )


def _is_valid_grounded_answer(answer: str, source_count: int) -> bool:
    if not answer.strip():
        return False
    cited = [int(match) for match in re.findall(r"\[\s*Source\s+(\d+)\s*\]", answer, flags=re.IGNORECASE)]
    if not cited:
        return False
    return all(1 <= source_id <= source_count for source_id in cited)
