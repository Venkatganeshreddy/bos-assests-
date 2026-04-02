from __future__ import annotations

import base64
import io
from typing import Any

import requests
from PIL import Image, ImageSequence

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def extract_text_from_image(
    image_bytes: bytes,
    api_key: str,
    model: str,
    mime_type: str,
    filename: str,
) -> str:
    normalized_bytes, normalized_mime = _prepare_image_bytes(image_bytes, mime_type)
    data_url = _to_data_url(normalized_bytes, normalized_mime)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all readable text from this image for search indexing. "
                            "Return only the extracted text. Preserve headings, short lists, "
                            "labels, and line breaks when possible. Do not summarize."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                        "imageUrl": {"url": data_url},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 2500,
    }

    response_json = _post_openrouter(payload, api_key)
    message = response_json["choices"][0]["message"]
    content = message.get("content", "")
    text = _message_content_to_text(content).strip()
    if not text:
        raise ValueError(f"OCR returned no readable text for {filename}.")
    return text


def extract_text_from_pdf(
    pdf_bytes: bytes,
    api_key: str,
    model: str,
    filename: str,
) -> str:
    data_url = _to_data_url(pdf_bytes, "application/pdf")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Parse this PDF for OCR and reply with only OK.",
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": data_url,
                            "fileData": data_url,
                        },
                    },
                ],
            }
        ],
        "plugins": [
            {
                "id": "file-parser",
                "pdf": {
                    "engine": "mistral-ocr",
                },
            }
        ],
        "temperature": 0,
        "max_tokens": 20,
    }

    response_json = _post_openrouter(payload, api_key)
    message = response_json["choices"][0]["message"]
    text_from_annotations = _extract_text_from_annotations(message.get("annotations", []))
    if text_from_annotations.strip():
        return text_from_annotations

    content = message.get("content", "")
    fallback = _message_content_to_text(content).strip()
    if not fallback or fallback.lower() == "ok":
        raise ValueError(f"PDF OCR returned no parsed text for {filename}.")
    return fallback


def _post_openrouter(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for OCR.")

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter OCR error {response.status_code}: {response.text[:300]}")

    return response.json()


def _prepare_image_bytes(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        if getattr(image, "is_animated", False):
            image = ImageSequence.Iterator(image).__next__()

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        max_dimension = 1600
        width, height = image.size
        longest_side = max(width, height)
        if longest_side > max_dimension:
            scale = max_dimension / float(longest_side)
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(new_size)

        output = io.BytesIO()
        if mime_type in {"image/png", "image/webp"} and image.mode == "RGBA":
            image.save(output, format="PNG")
            return output.getvalue(), "image/png"

        image = image.convert("RGB")
        image.save(output, format="JPEG", quality=92, optimize=True)
        return output.getvalue(), "image/jpeg"


def _to_data_url(file_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _extract_text_from_annotations(annotations: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for annotation in annotations or []:
        if annotation.get("type") != "file":
            continue
        file_info = annotation.get("file", {})
        for content_part in file_info.get("content", []):
            if content_part.get("type") == "text" and content_part.get("text"):
                parts.append(content_part["text"])

    return "\n\n".join(parts).strip()


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts)

    return str(content or "")
