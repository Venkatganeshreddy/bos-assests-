from __future__ import annotations

import os
from collections import Counter
from datetime import datetime
from html import escape

import streamlit as st
from dotenv import load_dotenv

from config_runtime import get_setting
from drive_loader import get_service_account_email_from_config, index_drive_folder_with_options
from llm import ask_openrouter
from retrieval import build_chunks, build_context, build_index, retrieve, serialize_sources

load_dotenv()


DEFAULT_FOLDER_URL = get_setting("DRIVE_FOLDER_URL")
DEFAULT_MODEL = get_setting("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
DEFAULT_OCR_MODEL = ""
DEFAULT_OCR_ENABLED = False
OPENROUTER_API_KEY = get_setting("OPENROUTER_API_KEY")

st.set_page_config(
    page_title="BOS Assets Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');

        /* ── Keyframes ── */
        @keyframes bos-fade-up {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes bos-pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50%      { opacity: .55; transform: scale(1.18); }
        }
        @keyframes bos-shimmer {
            0%   { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes bos-gradient-shift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* ── Base ── */
        .stApp {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.78)),
                radial-gradient(circle at top left,  rgba(207,177,133,0.18), transparent 32%),
                radial-gradient(circle at bottom right, rgba(42,85,73,0.12), transparent 35%),
                linear-gradient(180deg, #f6f2ea 0%, #eee6d9 100%);
            color: #163127;
        }
        header[data-testid="stHeader"] { background: transparent; }
        div[data-testid="stToolbar"]   { display: none; }
        .stDeployButton                { display: none; }
        footer                         { visibility: hidden; }

        html, body, [class*="css"] { font-family: 'Manrope', sans-serif !important; }
        h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.02em; }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }

        /* Tighten Streamlit's default element gaps */
        .block-container > div > div > div > div {
            margin-bottom: 0;
        }
        .element-container { margin-bottom: 0.4rem !important; }

        /* ── Sidebar ── */
        .bos-sidebar-title {
            font-family: 'Fraunces', serif;
            font-size: 1.6rem;
            color: #173127;
            margin: 0 0 0.25rem 0;
            line-height: 1;
        }
        .bos-sidebar-subtitle {
            color: #637267;
            font-size: 0.9rem;
            line-height: 1.45;
            margin-bottom: 0.9rem;
        }

        .bos-sidebar-card {
            border-radius: 22px;
            background: linear-gradient(135deg, #173c31 0%, #215044 60%, #1a5a45 100%);
            background-size: 200% 200%;
            animation: bos-gradient-shift 8s ease infinite;
            padding: 1rem 1rem 0.95rem;
            color: #f7f2eb;
            margin-bottom: 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 8px 24px rgba(23,49,39,0.18);
            transition: box-shadow 0.3s ease;
        }
        .bos-sidebar-card:hover {
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 12px 32px rgba(23,49,39,0.24);
        }
        .bos-sidebar-card-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: rgba(247,242,235,0.72);
            font-weight: 800;
        }
        .bos-sidebar-card-value {
            font-size: 1.2rem;
            font-weight: 800;
            margin-top: 0.45rem;
            line-height: 1.15;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .bos-sidebar-card-meta {
            margin-top: 0.65rem;
            color: rgba(247,242,235,0.86);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        /* Status dot */
        .bos-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            display: inline-block;
            flex-shrink: 0;
        }
        .bos-dot--live {
            background: #4ade80;
            box-shadow: 0 0 8px rgba(74,222,128,0.6);
            animation: bos-pulse 2s ease-in-out infinite;
        }
        .bos-dot--idle {
            background: rgba(247,242,235,0.4);
        }

        .bos-sidebar-skip {
            padding: 0.65rem 0;
            border-top: 1px solid rgba(19,50,39,0.08);
        }
        .bos-sidebar-skip:first-child { padding-top: 0; border-top: none; }
        .bos-sidebar-skip a {
            color: #1d5a48 !important;
            font-weight: 700;
            text-decoration: none;
        }
        .bos-sidebar-skip a:hover { text-decoration: underline; }
        .bos-sidebar-skip-reason {
            margin-top: 0.22rem;
            color: #6a786e;
            font-size: 0.84rem;
            line-height: 1.45;
            word-break: break-word;
        }

        /* ── Hero ── */
        .bos-shell {
            border: 1px solid rgba(19,50,39,0.08);
            background: linear-gradient(135deg, rgba(255,252,247,0.96), rgba(248,244,237,0.92));
            border-radius: 28px;
            padding: 1.6rem 1.7rem;
            box-shadow: 0 22px 60px rgba(36,49,42,0.08);
            margin-bottom: 0.8rem;
            animation: bos-fade-up 0.5s ease-out both;
            position: relative;
            overflow: hidden;
        }
        .bos-shell::before {
            content: '';
            position: absolute;
            top: -50%; right: -20%;
            width: 60%; height: 200%;
            background: radial-gradient(ellipse, rgba(207,177,133,0.08), transparent 70%);
            pointer-events: none;
        }
        .bos-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            font-weight: 700;
            color: #8a5a39;
            margin-bottom: 0.55rem;
        }
        .bos-hero {
            display: grid;
            grid-template-columns: minmax(0, 1.8fr) minmax(220px, 0.62fr);
            gap: 1.2rem;
            align-items: start;
        }
        .bos-title {
            font-size: clamp(2.1rem, 4vw, 3.6rem);
            line-height: 0.98;
            color: #193127;
            margin: 0;
        }
        .bos-sub {
            margin-top: 0.9rem;
            color: #4b5d53;
            max-width: 40rem;
            font-size: 0.98rem;
            line-height: 1.55;
        }

        /* Chips */
        .bos-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1rem;
        }
        .bos-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.52rem 0.74rem;
            border-radius: 999px;
            background: rgba(245,239,229,0.95);
            border: 1px solid rgba(19,50,39,0.08);
            color: #264237;
            font-size: 0.82rem;
            font-weight: 600;
            backdrop-filter: blur(4px);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .bos-chip:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(33,42,38,0.08);
        }

        /* Status card (hero) */
        .bos-status-card {
            background: linear-gradient(180deg, #13372d 0%, #1c493b 100%);
            color: #f4efe7;
            border-radius: 20px;
            padding: 0.95rem 1rem;
            min-height: 0;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 12px 32px rgba(19,55,45,0.18);
            transition: transform 0.25s ease;
        }
        .bos-status-card:hover { transform: translateY(-2px); }
        .bos-status-label {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.7rem;
            color: rgba(243,235,222,0.72);
            font-weight: 700;
        }
        .bos-status-value {
            margin-top: 0.42rem;
            font-size: 1.2rem;
            font-weight: 800;
            line-height: 1.1;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }
        .bos-status-meta {
            margin-top: 0.65rem;
            color: rgba(243,235,222,0.84);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        /* ── Quickstart ── */
        .bos-quickstart-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.9rem;
            margin-bottom: 0.8rem;
        }
        .bos-quickstep {
            border-radius: 22px;
            background: rgba(255,253,249,0.96);
            border: 1px solid rgba(19,50,39,0.08);
            padding: 0.95rem 1rem;
            box-shadow: 0 10px 28px rgba(33,42,38,0.05);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            animation: bos-fade-up 0.5s ease-out both;
        }
        .bos-quickstep:nth-child(1) { animation-delay: 0.08s; }
        .bos-quickstep:nth-child(2) { animation-delay: 0.16s; }
        .bos-quickstep:nth-child(3) { animation-delay: 0.24s; }
        .bos-quickstep:hover {
            transform: translateY(-3px);
            box-shadow: 0 16px 40px rgba(33,42,38,0.10);
        }
        .bos-quickstep-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px; height: 28px;
            border-radius: 50%;
            background: linear-gradient(135deg, #e8ddd0, #f0e6d8);
            color: #8a5a39;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0;
        }
        .bos-quickstep-title {
            color: #173127;
            font-size: 1rem;
            font-weight: 800;
            margin-top: 0.55rem;
        }
        .bos-quickstep-copy {
            color: #66766b;
            margin-top: 0.32rem;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        /* ── Metric cards ── */
        .bos-metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.9rem;
            margin-bottom: 0.8rem;
        }
        .bos-card {
            border-radius: 22px;
            background: rgba(255,253,249,0.96);
            border: 1px solid rgba(19,50,39,0.08);
            padding: 1.1rem 1.15rem;
            min-height: 122px;
            box-shadow: 0 10px 28px rgba(33,42,38,0.05);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            animation: bos-fade-up 0.45s ease-out both;
        }
        .bos-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 16px 40px rgba(33,42,38,0.10);
        }
        .bos-label {
            color: #75624b;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.68rem;
            font-weight: 700;
        }
        .bos-value {
            font-size: 2.05rem;
            font-weight: 800;
            color: #173127;
            margin-top: 0.45rem;
            line-height: 1;
            background: linear-gradient(135deg, #173127, #1d5a48);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .bos-meta {
            color: #627267;
            margin-top: 0.5rem;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        /* ── Empty state ── */
        .bos-empty {
            border-radius: 24px;
            border: 1.5px dashed rgba(19,50,39,0.16);
            background: rgba(255,251,244,0.8);
            padding: 2rem 2rem;
            color: #516258;
            line-height: 1.7;
            text-align: center;
            max-width: 540px;
            margin: 0.8rem auto 0.8rem;
            animation: bos-fade-up 0.5s ease-out both;
        }
        .bos-empty strong { color: #22392f; }
        .bos-empty-icon {
            font-size: 2.8rem;
            margin-bottom: 0.6rem;
            display: block;
            opacity: 0.7;
        }
        .bos-simple-list {
            margin-top: 1rem;
            color: #214036;
            line-height: 2;
            padding-left: 0;
            list-style: none;
            text-align: left;
            display: inline-block;
        }
        .bos-simple-list li {
            position: relative;
            padding-left: 1.6rem;
        }
        .bos-simple-list li::before {
            content: attr(data-step);
            position: absolute;
            left: 0;
            top: 0.1em;
            width: 20px; height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #e8ddd0, #f0e6d8);
            color: #8a5a39;
            font-size: 0.65rem;
            font-weight: 800;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* ── Chat ── */
        .bos-chat-heading {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 1rem;
            margin: 0.6rem 0 0.75rem;
            padding-top: 0.8rem;
            border-top: 1px solid rgba(19,50,39,0.06);
            animation: bos-fade-up 0.4s ease-out both;
        }
        .bos-chat-title {
            font-family: 'Fraunces', serif;
            font-size: 1.35rem;
            color: #173127;
            margin: 0;
        }
        .bos-chat-copy {
            color: #68786d;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Source cards */
        .bos-source {
            padding: 0.85rem 0.9rem;
            border: 1px solid rgba(19,50,39,0.06);
            border-radius: 14px;
            background: rgba(248,245,240,0.6);
            margin-bottom: 0.55rem;
            transition: background 0.2s ease, box-shadow 0.2s ease;
        }
        .bos-source:hover {
            background: rgba(248,245,240,0.9);
            box-shadow: 0 4px 14px rgba(33,42,38,0.06);
        }
        .bos-source-title {
            color: #173127;
            font-weight: 700;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .bos-source-icon {
            width: 18px; height: 18px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.6rem;
            font-weight: 800;
            color: white;
            flex-shrink: 0;
            text-transform: uppercase;
        }
        .bos-source-icon--pdf  { background: #e74c3c; }
        .bos-source-icon--doc  { background: #4285f4; }
        .bos-source-icon--sheet { background: #0f9d58; }
        .bos-source-icon--slide { background: #f4b400; color: #333; }
        .bos-source-icon--text { background: #6b7280; }
        .bos-source-icon--default { background: #8a5a39; }
        .bos-source-path {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.78rem;
            color: #5e6d63;
            margin-top: 0.22rem;
            word-break: break-word;
            padding-left: 1.65rem;
        }
        .bos-source-meta {
            color: #6f7e73;
            margin-top: 0.25rem;
            font-size: 0.82rem;
            padding-left: 1.65rem;
        }
        .bos-source-link {
            margin-top: 0.35rem;
            padding-left: 1.65rem;
        }
        .bos-source-link a {
            color: #1d5a48 !important;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.82rem;
            transition: color 0.2s;
        }
        .bos-source-link a:hover {
            color: #2a7a60 !important;
            text-decoration: underline;
        }

        /* ── Streamlit overrides ── */
        [data-testid="stSidebar"] {
            background: rgba(252,248,241,0.98);
            border-right: 1px solid rgba(19,50,39,0.08);
        }
        [data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        [data-testid="stSidebar"] .element-container { margin-bottom: 0.25rem !important; }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            color: #1c352b !important;
            background: transparent !important;
        }
        [data-testid="stSidebar"] .stCaption { color: #617166 !important; }

        div.stButton > button {
            border-radius: 14px;
            font-weight: 700;
            min-height: 2.8rem;
            border: 1px solid rgba(19,50,39,0.12);
            box-shadow: none;
            transition: all 0.25s ease;
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(180deg, #1d5a48 0%, #174636 100%);
            color: #f8f3eb !important;
            opacity: 1 !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(180deg, #23725c 0%, #1d5a48 100%);
            box-shadow: 0 6px 20px rgba(23,70,54,0.25);
            transform: translateY(-1px);
        }
        div.stButton > button[kind="primary"] * {
            color: #f8f3eb !important;
            fill: #f8f3eb !important;
        }
        div.stButton > button[kind="secondary"] {
            background: #f4eee4;
            color: #1e362c;
        }
        div.stButton > button[kind="secondary"]:hover {
            background: #ede5d8;
            transform: translateY(-1px);
        }
        div.stButton > button:hover {
            border-color: rgba(19,50,39,0.24);
        }

        div[data-testid="stTextInput"] input,
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input,
        textarea {
            border-radius: 14px;
            background: #fffdf9 !important;
            color: #173127 !important;
            caret-color: #173127 !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="stTextInput"] input:focus,
        textarea:focus {
            box-shadow: 0 0 0 2px rgba(29,90,72,0.15) !important;
        }
        div[data-testid="stTextInput"] input::placeholder,
        textarea::placeholder { color: #708176 !important; }
        div[data-baseweb="input"],
        div[data-baseweb="base-input"] {
            background: #fffdf9 !important;
            border-radius: 14px !important;
            border: 1px solid rgba(19,50,39,0.12) !important;
        }

        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] p {
            color: #1d352b !important;
            background: transparent !important;
        }
        [data-testid="stCheckbox"] { margin-top: 0.25rem; margin-bottom: 0.35rem; }
        [data-testid="stCheckbox"] input[type="checkbox"] {
            accent-color: #1d5a48 !important;
            width: 1rem !important;
            height: 1rem !important;
        }
        [data-baseweb="checkbox"] { gap: 0.55rem !important; }
        [data-baseweb="checkbox"] > div:first-child {
            border-color: rgba(29,90,72,0.5) !important;
            background: #fffdf9 !important;
        }
        [data-baseweb="checkbox"] svg { color: #1d5a48 !important; fill: #1d5a48 !important; }

        [data-testid="stChatInput"] { background: transparent !important; }
        [data-testid="stChatInput"] textarea {
            background: #fffdf9 !important;
            color: #173127 !important;
        }
        [data-testid="stChatInput"] > div {
            background: rgba(255,253,249,0.98) !important;
            border: 1px solid rgba(19,50,39,0.1) !important;
            border-radius: 18px !important;
            box-shadow: 0 10px 28px rgba(33,42,38,0.05);
            transition: box-shadow 0.25s ease;
        }
        [data-testid="stChatInput"] > div:focus-within {
            box-shadow: 0 10px 28px rgba(33,42,38,0.05), 0 0 0 2px rgba(29,90,72,0.12) !important;
        }

        /* Chat messages */
        div[data-testid="stChatMessage"] {
            background: rgba(255,253,249,0.96);
            border: 1px solid rgba(19,50,39,0.08);
            border-radius: 22px;
            padding: 0.75rem 0.95rem;
            box-shadow: 0 10px 28px rgba(33,42,38,0.05);
            animation: bos-fade-up 0.3s ease-out both;
            margin-bottom: 0.5rem !important;
        }
        /* Tighten Streamlit's default gap between chat messages */
        div[data-testid="stChatMessageContainer"] > div {
            gap: 0.5rem !important;
        }
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] strong {
            color: #173127 !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(19,50,39,0.08);
            border-radius: 18px;
            background: rgba(255,253,249,0.92);
            transition: box-shadow 0.2s ease;
        }
        div[data-testid="stExpander"]:hover {
            box-shadow: 0 4px 16px rgba(33,42,38,0.06);
        }

        /* ── Responsive ── */
        @media (max-width: 1100px) {
            .bos-hero { grid-template-columns: 1fr; }
            .bos-metrics-grid { grid-template-columns: repeat(2, 1fr); }
            .bos-quickstart-grid { grid-template-columns: repeat(3, 1fr); }
        }
        @media (max-width: 780px) {
            .bos-hero,
            .bos-metrics-grid,
            .bos-quickstart-grid { grid-template-columns: 1fr; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_folder_index(
    folder_url: str,
    progress_callback=None,
) -> dict:
    return index_drive_folder_with_options(
        folder_url=folder_url,
        progress_callback=progress_callback,
    )


def build_runtime_bundle(folder_payload: dict) -> dict:
    chunks = build_chunks(folder_payload["documents"])
    vectorizer, matrix = build_index(chunks)

    return {
        **folder_payload,
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@st.cache_resource(ttl=3600, show_spinner=False)
def _cached_index(folder_url: str) -> dict:
    """Build and cache the folder index so it persists across sessions."""
    folder_payload = index_drive_folder_with_options(folder_url=folder_url)
    return build_runtime_bundle(folder_payload)


@st.cache_resource(show_spinner=False)
def _index_flag() -> dict:
    """Mutable container shared across sessions to track if indexing happened."""
    return {"done": False}


def _has_been_indexed() -> bool:
    return _index_flag()["done"]


def _mark_indexed() -> None:
    _index_flag()["done"] = True



def render_hero(bundle: dict | None) -> None:
    ready = bundle is not None
    status_title = "Ready" if ready else "Not indexed"
    dot_class = "bos-dot--live" if ready else "bos-dot--idle"
    status_meta = (
        f"{len(bundle['documents']):,} files indexed &middot; {len(bundle['chunks']):,} chunks ready"
        if ready
        else "Click Index BOS Folder to activate search."
    )
    folder_chip = bundle["folder_name"] if ready else "No folder indexed yet"
    st.markdown(
        f"""
        <div class="bos-shell">
            <div class="bos-hero">
                <div>
                    <div class="bos-kicker">BOS Asset Intelligence</div>
                    <h1 class="bos-title">Search assets, copy systems, decks, and source files from one clean workspace.</h1>
                    <div class="bos-sub">
                        This internal dashboard indexes a single BOS Drive folder, turns readable files into
                        retrieval-ready search chunks, and answers with citations so the team can verify exactly
                        where each answer came from.
                    </div>
                    <div class="bos-chip-row">
                        <div class="bos-chip">Folder: {escape(folder_chip)}</div>
                        <div class="bos-chip">Mode: file-backed answers</div>
                        <div class="bos-chip">OCR disabled</div>
                    </div>
                </div>
                <div class="bos-status-card">
                    <div>
                        <div class="bos-status-label">Workspace Status</div>
                        <div class="bos-status-value">
                            <span class="bos-dot {dot_class}"></span>
                            {escape(status_title)}
                        </div>
                    </div>
                    <div class="bos-status-meta">
                        {status_meta}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(bundle: dict) -> None:
    kind_counts = Counter(document.source_kind for document in bundle["documents"])
    top_formats = ", ".join(
        f"{name} {count}" for name, count in kind_counts.most_common(3)
    ) or "No formats indexed yet"
    metrics = [
        ("Indexed files", f"{len(bundle['documents']):,}", "readable files available"),
        ("Search chunks", f"{len(bundle['chunks']):,}", "retrieval-ready content blocks"),
        ("Skipped files", f"{len(bundle['skipped']):,}", "unsupported, empty, or blocked"),
        ("Top formats", str(len(kind_counts)), top_formats),
    ]

    cards = "".join(
        f'<div class="bos-card">'
        f'<div class="bos-label">{escape(label)}</div>'
        f'<div class="bos-value">{escape(value)}</div>'
        f'<div class="bos-meta">{escape(meta)}</div>'
        f"</div>"
        for label, value, meta in metrics
    )
    st.markdown(f'<div class="bos-metrics-grid">{cards}</div>', unsafe_allow_html=True)


def render_quickstart(bundle: dict | None) -> None:
    if bundle:
        steps = [
            ("1", "Update folder", "Change or add files in Drive whenever needed."),
            ("2", "Re-index", "Click Index BOS Folder to refresh the searchable library."),
            ("3", "Ask clearly", "Use file names, campaigns, or keywords for better answers."),
        ]
    else:
        steps = [
            ("1", "Use configured folder", "The app is pinned to the BOS Drive folder."),
            ("2", "Click Index BOS Folder", "The app will read files and prepare search."),
            ("3", "Start chatting", "Ask about assets once indexing finishes."),
        ]

    cards = "".join(
        f'<div class="bos-quickstep">'
        f'<div class="bos-quickstep-num">{escape(num)}</div>'
        f'<div class="bos-quickstep-title">{escape(value)}</div>'
        f'<div class="bos-quickstep-copy">{escape(meta)}</div>'
        f"</div>"
        for num, value, meta in steps
    )
    st.markdown(f'<div class="bos-quickstart-grid">{cards}</div>', unsafe_allow_html=True)




def render_empty_state() -> None:
    st.markdown(
        """
        <div class="bos-empty">
            <span class="bos-empty-icon">&#128218;</span>
            <strong>Welcome to BOS Assets</strong><br/>
            Share the target Drive folder with the service-account email,
            then click <strong>Index BOS Folder</strong> to get started.
            <ul class="bos-simple-list">
                <li data-step="1">Share the folder with the service-account email</li>
                <li data-step="2">Click <strong>Index BOS Folder</strong> in the sidebar</li>
                <li data-step="3">Ask your question and get cited answers</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


_SOURCE_ICON_MAP = {
    "pdf": ("pdf", "PDF"),
    "google-doc": ("doc", "DOC"),
    "docx": ("doc", "DOC"),
    "google-sheet": ("sheet", "XLS"),
    "spreadsheet-file": ("sheet", "XLS"),
    "google-slide": ("slide", "PPT"),
    "pptx": ("slide", "PPT"),
    "text-file": ("text", "TXT"),
    "svg": ("text", "SVG"),
}


def render_sources(sources: list[dict[str, str]]) -> None:
    for source in sources:
        kind = source.get("kind", "")
        icon_class, icon_label = _SOURCE_ICON_MAP.get(kind, ("default", kind[:3].upper() or "?"))
        link = (
            f'<div class="bos-source-link"><a href="{escape(source["link"])}" target="_blank">Open source file &#8599;</a></div>'
            if source["link"]
            else ""
        )
        st.markdown(
            (
                '<div class="bos-source">'
                f'<div class="bos-source-title">'
                f'<span class="bos-source-icon bos-source-icon--{icon_class}">{escape(icon_label)}</span>'
                f'{escape(source["label"])}'
                f'</div>'
                f'<div class="bos-source-path">{escape(source["path"])}</div>'
                f'<div class="bos-source-meta">{escape(source["kind"])} &middot; chunk {escape(source["chunk"])}</div>'
                f"{link}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def history_for_model(messages: list[dict], keep_last: int = 6) -> list[dict[str, str]]:
    trimmed = [message for message in messages if message["role"] == "user"][-keep_last:]
    return [{"role": "user", "content": message["content"]} for message in trimmed]


def render_sidebar_status(bundle: dict | None) -> None:
    if bundle:
        title = "Library online"
        dot_class = "bos-dot--live"
        meta = (
            f"{len(bundle['documents']):,} files indexed &middot; "
            f"{len(bundle['chunks']):,} chunks ready"
        )
    else:
        title = "Awaiting index"
        dot_class = "bos-dot--idle"
        meta = "Connect the Drive folder, then click Index BOS Folder to activate search."

    st.markdown(
        f"""
        <div class="bos-sidebar-card">
            <div class="bos-sidebar-card-label">BOS Workspace</div>
            <div class="bos-sidebar-card-value">
                <span class="bos-dot {dot_class}"></span>
                {escape(title)}
            </div>
            <div class="bos-sidebar-card-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_skipped(bundle: dict | None) -> None:
    if not bundle or not bundle.get("skipped"):
        return

    with st.expander(f"Unable to index ({len(bundle['skipped'])})", expanded=True):
        for item in bundle["skipped"]:
            link = item.get("link", "")
            if link:
                title = f'<a href="{escape(link)}" target="_blank">{escape(item["name"])}</a>'
            else:
                title = escape(item["name"])

            st.markdown(
                (
                    '<div class="bos-sidebar-skip">'
                    f"{title}"
                    f'<div class="bos-sidebar-skip-reason">{escape(item["reason"])}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "folder_bundle" not in st.session_state:
    st.session_state.folder_bundle = None

try:
    service_account_email = get_service_account_email_from_config()
    credentials_error = ""
except Exception as exc:
    service_account_email = ""
    credentials_error = str(exc)

with st.sidebar:
    st.markdown('<div class="bos-sidebar-title">BOS Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="bos-sidebar-subtitle">This app is locked to the BOS assets folder. Click index, then start asking questions.</div>',
        unsafe_allow_html=True,
    )
    render_sidebar_status(st.session_state.folder_bundle)
    folder_url = DEFAULT_FOLDER_URL
    if folder_url:
        st.caption("Configured BOS folder")
        st.code(folder_url, language=None)
    else:
        st.error("DRIVE_FOLDER_URL is missing in app secrets.")

    with st.expander("Advanced settings", expanded=False):
        model_name = st.text_input("OpenRouter model", value=DEFAULT_MODEL)
        enable_ocr = st.checkbox(
            "Enable OCR for images and scanned PDFs",
            value=DEFAULT_OCR_ENABLED,
            help="Uses OpenRouter during indexing for image files and scanned PDF documents.",
        )
        st.caption(
            f"OCR status: {'On' if enable_ocr else 'Off'}"
            + (" • slower indexing" if enable_ocr else "")
        )
        ocr_model_name = st.text_input("OCR model", value=DEFAULT_OCR_MODEL)

    if "enable_ocr" not in locals():
        enable_ocr = DEFAULT_OCR_ENABLED
        model_name = DEFAULT_MODEL
        ocr_model_name = DEFAULT_OCR_MODEL

    if service_account_email:
        with st.expander("Access details", expanded=False):
            st.caption("Share the folder with this service account")
            st.code(service_account_email, language=None)
    elif credentials_error:
        st.error(credentials_error)
    index_clicked = st.button("Index BOS Folder", type="primary", use_container_width=True)
    clear_clicked = st.button("Clear Chat", use_container_width=True)

    if clear_clicked:
        st.session_state.messages = []

    st.caption(
        "Supports Docs, Sheets, Slides, PDF, DOCX, PPTX, XLSX, CSV, TXT, JSON, SVG"
        + (" plus OCR for images and scanned PDFs." if enable_ocr else ".")
    )
    render_sidebar_skipped(st.session_state.folder_bundle)

if index_clicked:
    if not folder_url.strip():
        st.error("Set DRIVE_FOLDER_URL in Streamlit secrets first.")
    else:
        try:
            _cached_index.clear()
            _index_flag()["done"] = False
            with st.spinner("Indexing the BOS asset folder — this may take a moment..."):
                st.session_state.folder_bundle = _cached_index(folder_url.strip())
                _mark_indexed()
                folder_name = st.session_state.folder_bundle.get("folder_name", "BOS folder")
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": (
                            f"I indexed **{folder_name}** and I’m ready to answer "
                            "questions with file-backed citations."
                        ),
                    }
                ]
        except Exception as exc:
            st.error(str(exc))

# On page load, check if a previous session already built the index.
# _cached_index uses @st.cache_resource so cache hits are instant.
# We track whether indexing has ever completed via a cache_resource flag
# to avoid triggering a full index on cold starts.
if not st.session_state.folder_bundle and folder_url and folder_url.strip() and not credentials_error:
    if _has_been_indexed():
        try:
            st.session_state.folder_bundle = _cached_index(folder_url.strip())
        except Exception:
            pass

bundle = st.session_state.folder_bundle

render_hero(bundle)
render_quickstart(bundle)

if bundle:
    render_metrics(bundle)

    with st.expander("Index details", expanded=False):
        st.write(f"Folder: {bundle['folder_name']}")
        st.write(f"Indexed at: {bundle['indexed_at']}")
        st.write(f"Folder link: {bundle['folder_link']}")
        st.write("OCR enabled: No")
        if bundle["skipped"]:
            st.write("Skipped files")
            for item in bundle["skipped"][:25]:
                st.write(f"- {item['name']}: {item['reason']}")
else:
    render_empty_state()

st.markdown(
    """
    <div class="bos-chat-heading">
        <div>
            <div class="bos-chat-title">Workspace Conversation</div>
            <div class="bos-chat-copy">Ask precise questions and verify the answer against the returned BOS sources.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources", expanded=False):
                render_sources(message["sources"])

prompt = st.chat_input(
    "Ask about BOS assets, copy references, brand files, decks, or source material"
    if bundle
    else "Click `Index BOS Folder` to start the conversation",
    disabled=not bool(bundle),
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not bundle:
            answer = "The BOS folder has not been indexed yet."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        elif not OPENROUTER_API_KEY:
            answer = "OPENROUTER_API_KEY is missing, so I can search sources but cannot generate an answer yet."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            with st.spinner("Searching BOS sources..."):
                retrieved = retrieve(
                    query=prompt,
                    chunks=bundle["chunks"],
                    vectorizer=bundle["vectorizer"],
                    matrix=bundle["matrix"],
                )
                context = build_context(retrieved)
                sources = serialize_sources(retrieved)
                model_messages = history_for_model(st.session_state.messages[:-1])
                answer = ask_openrouter(
                    api_key=OPENROUTER_API_KEY,
                    model=model_name.strip() or DEFAULT_MODEL,
                    messages=model_messages + [{"role": "user", "content": prompt}],
                    context=context,
                )

            st.markdown(answer)
            if sources:
                with st.expander("Sources", expanded=False):
                    render_sources(sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )
