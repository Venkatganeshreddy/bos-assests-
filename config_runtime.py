from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st


def get_setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets[name]
        return str(value)
    except Exception:
        return os.getenv(name, default)


def get_secret_mapping(*names: str) -> dict[str, Any] | None:
    for name in names:
        try:
            raw_secret = st.secrets[name]
            return json.loads(json.dumps(dict(raw_secret)))
        except Exception:
            continue
    return None


def get_json_env(*names: str) -> dict[str, Any] | None:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value:
            return json.loads(raw_value)
    return None
