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

        .stApp {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.78), rgba(255, 255, 255, 0.78)),
                radial-gradient(circle at top left, rgba(207, 177, 133, 0.18), transparent 32%),
                radial-gradient(circle at bottom right, rgba(42, 85, 73, 0.12), transparent 35%),
                linear-gradient(180deg, #f6f2ea 0%, #eee6d9 100%);
            color: #163127;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        div[data-testid="stToolbar"] {
            display: none;
        }

        .stDeployButton {
            display: none;
        }

        footer {
            visibility: hidden;
        }

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif !important;
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif !important;
            letter-spacing: -0.02em;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }

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
            background: linear-gradient(180deg, #173c31 0%, #215044 100%);
            padding: 1rem 1rem 0.95rem;
            color: #f7f2eb;
            margin-bottom: 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }

        .bos-sidebar-card-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: rgba(247, 242, 235, 0.72);
            font-weight: 800;
        }

        .bos-sidebar-card-value {
            font-size: 1.2rem;
            font-weight: 800;
            margin-top: 0.45rem;
            line-height: 1.15;
        }

        .bos-sidebar-card-meta {
            margin-top: 0.65rem;
            color: rgba(247, 242, 235, 0.86);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .bos-sidebar-skip {
            padding: 0.65rem 0;
            border-top: 1px solid rgba(19, 50, 39, 0.08);
        }

        .bos-sidebar-skip:first-child {
            padding-top: 0;
            border-top: none;
        }

        .bos-sidebar-skip a {
            color: #1d5a48 !important;
            font-weight: 700;
            text-decoration: none;
        }

        .bos-sidebar-skip a:hover {
            text-decoration: underline;
        }

        .bos-sidebar-skip-reason {
            margin-top: 0.22rem;
            color: #6a786e;
            font-size: 0.84rem;
            line-height: 1.45;
            word-break: break-word;
        }

        .bos-shell {
            border: 1px solid rgba(19, 50, 39, 0.08);
            background:
                linear-gradient(135deg, rgba(255, 252, 247, 0.96), rgba(248, 244, 237, 0.92));
            border-radius: 28px;
            padding: 1.6rem 1.7rem;
            box-shadow: 0 22px 60px rgba(36, 49, 42, 0.08);
            margin-bottom: 1rem;
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
            background: rgba(245, 239, 229, 0.95);
            border: 1px solid rgba(19, 50, 39, 0.08);
            color: #264237;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .bos-status-card {
            background: linear-gradient(180deg, #13372d 0%, #1c493b 100%);
            color: #f4efe7;
            border-radius: 20px;
            padding: 0.95rem 1rem;
            min-height: 0;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        .bos-status-label {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.7rem;
            color: rgba(243, 235, 222, 0.72);
            font-weight: 700;
        }

        .bos-status-value {
            margin-top: 0.42rem;
            font-size: 1.2rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .bos-status-meta {
            margin-top: 0.65rem;
            color: rgba(243, 235, 222, 0.84);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .bos-quickstart {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.95rem;
            margin: 0.95rem 0 1.1rem;
        }

        .bos-quickstep {
            border-radius: 22px;
            background: rgba(255, 253, 249, 0.96);
            border: 1px solid rgba(19, 50, 39, 0.08);
            padding: 0.95rem 1rem;
            box-shadow: 0 10px 28px rgba(33, 42, 38, 0.05);
        }

        .bos-quickstep-num {
            color: #8a5a39;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-weight: 800;
        }

        .bos-quickstep-title {
            color: #173127;
            font-size: 1rem;
            font-weight: 800;
            margin-top: 0.4rem;
        }

        .bos-quickstep-copy {
            color: #66766b;
            margin-top: 0.32rem;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .bos-metrics {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.95rem;
            margin: 0.95rem 0 1.2rem;
        }

        .bos-card {
            border-radius: 22px;
            background: rgba(255, 253, 249, 0.96);
            border: 1px solid rgba(19, 50, 39, 0.08);
            padding: 1.1rem 1.15rem;
            min-height: 122px;
            box-shadow: 0 10px 28px rgba(33, 42, 38, 0.05);
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
        }

        .bos-meta {
            color: #627267;
            margin-top: 0.5rem;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .bos-panel {
            border-radius: 24px;
            background: rgba(255, 253, 249, 0.96);
            border: 1px solid rgba(19, 50, 39, 0.08);
            padding: 1.15rem 1.2rem;
            box-shadow: 0 10px 28px rgba(33, 42, 38, 0.05);
            height: 100%;
        }

        .bos-summary {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin-bottom: 1rem;
        }

        .bos-summary-card {
            border-radius: 22px;
            background: rgba(255, 253, 249, 0.96);
            border: 1px solid rgba(19, 50, 39, 0.08);
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 28px rgba(33, 42, 38, 0.05);
            min-height: 112px;
        }

        .bos-summary-label {
            color: #7a654b;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 800;
        }

        .bos-summary-value {
            margin-top: 0.5rem;
            color: #1f382f;
            font-size: 1rem;
            line-height: 1.5;
            word-break: break-word;
        }

        .bos-summary-value a {
            color: #1d5a48 !important;
            font-weight: 700;
            text-decoration: none;
        }

        .bos-summary-value a:hover {
            text-decoration: underline;
        }

        .bos-empty {
            border-radius: 24px;
            border: 1px dashed rgba(19, 50, 39, 0.16);
            background: rgba(255, 251, 244, 0.8);
            padding: 1.2rem 1.25rem;
            color: #516258;
            line-height: 1.7;
        }

        .bos-empty strong {
            color: #22392f;
        }

        .bos-simple-list {
            margin-top: 0.8rem;
            color: #214036;
            line-height: 1.7;
            padding-left: 1.1rem;
        }

        .bos-chat-heading {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 1rem;
            margin: 1.15rem 0 0.85rem;
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

        .bos-source {
            padding: 0.85rem 0;
            border-top: 1px solid rgba(19, 50, 39, 0.08);
        }

        .bos-source:first-child {
            border-top: none;
            padding-top: 0;
        }

        .bos-source-title {
            color: #173127;
            font-weight: 700;
            font-size: 0.98rem;
        }

        .bos-source-path {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.83rem;
            color: #5e6d63;
            margin-top: 0.22rem;
            word-break: break-word;
        }

        .bos-source-meta {
            color: #6f7e73;
            margin-top: 0.25rem;
            font-size: 0.87rem;
        }

        .bos-source-link a {
            color: #9a552f !important;
            text-decoration: none;
            font-weight: 700;
        }

        .bos-source-link a:hover {
            text-decoration: underline;
        }

        [data-testid="stSidebar"] {
            background: rgba(252, 248, 241, 0.98);
            border-right: 1px solid rgba(19, 50, 39, 0.08);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            color: #1c352b !important;
            background: transparent !important;
        }

        [data-testid="stSidebar"] .stCaption {
            color: #617166 !important;
        }

        div.stButton > button {
            border-radius: 14px;
            font-weight: 700;
            min-height: 2.8rem;
            border: 1px solid rgba(19, 50, 39, 0.12);
            box-shadow: none;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(180deg, #1d5a48 0%, #174636 100%);
            color: #f8f3eb !important;
            opacity: 1 !important;
        }

        div.stButton > button[kind="primary"] * {
            color: #f8f3eb !important;
            fill: #f8f3eb !important;
        }

        div.stButton > button[kind="secondary"] {
            background: #f4eee4;
            color: #1e362c;
        }

        div.stButton > button:hover {
            border-color: rgba(19, 50, 39, 0.24);
        }

        div[data-testid="stTextInput"] input,
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input,
        textarea {
            border-radius: 14px;
            background: #fffdf9 !important;
            color: #173127 !important;
            caret-color: #173127 !important;
        }

        div[data-testid="stTextInput"] input::placeholder,
        textarea::placeholder {
            color: #708176 !important;
        }

        div[data-baseweb="input"],
        div[data-baseweb="base-input"] {
            background: #fffdf9 !important;
            border-radius: 14px !important;
            border: 1px solid rgba(19, 50, 39, 0.12) !important;
        }

        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] p {
            color: #1d352b !important;
            background: transparent !important;
        }

        [data-testid="stCheckbox"] {
            margin-top: 0.25rem;
            margin-bottom: 0.35rem;
        }

        [data-testid="stCheckbox"] input[type="checkbox"] {
            accent-color: #1d5a48 !important;
            width: 1rem !important;
            height: 1rem !important;
        }

        [data-baseweb="checkbox"] {
            gap: 0.55rem !important;
        }

        [data-baseweb="checkbox"] > div:first-child {
            border-color: rgba(29, 90, 72, 0.5) !important;
            background: #fffdf9 !important;
        }

        [data-baseweb="checkbox"] svg {
            color: #1d5a48 !important;
            fill: #1d5a48 !important;
        }

        [data-testid="stChatInput"] {
            background: transparent !important;
        }

        [data-testid="stChatInput"] textarea {
            background: #fffdf9 !important;
            color: #173127 !important;
        }

        [data-testid="stChatInput"] > div {
            background: rgba(255, 253, 249, 0.98) !important;
            border: 1px solid rgba(19, 50, 39, 0.1) !important;
            border-radius: 18px !important;
            box-shadow: 0 10px 28px rgba(33, 42, 38, 0.05);
        }

        div[data-testid="stChatMessage"] {
            background: rgba(255, 253, 249, 0.96);
            border: 1px solid rgba(19, 50, 39, 0.08);
            border-radius: 22px;
            padding: 0.75rem 0.95rem;
            box-shadow: 0 10px 28px rgba(33, 42, 38, 0.05);
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] strong {
            color: #173127 !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(19, 50, 39, 0.08);
            border-radius: 18px;
            background: rgba(255, 253, 249, 0.92);
        }

        @media (max-width: 1100px) {
            .bos-hero,
            .bos-metrics,
            .bos-summary,
            .bos-quickstart {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 780px) {
            .bos-hero,
            .bos-metrics,
            .bos-summary,
            .bos-quickstart {
                grid-template-columns: 1fr;
            }
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


def render_hero(bundle: dict | None) -> None:
    ready = bundle is not None
    status_title = "Ready" if ready else "Not indexed"
    status_meta = (
        f"{len(bundle['documents']):,} files indexed • {len(bundle['chunks']):,} chunks ready"
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
                        <div class="bos-status-value">{escape(status_title)}</div>
                    </div>
                    <div class="bos-status-meta">
                        {escape(status_meta)}
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

    columns = st.columns(len(metrics), gap="medium")
    for column, (label, value, meta) in zip(columns, metrics):
        with column:
            st.markdown(
                (
                    '<div class="bos-card">'
                    f'<div class="bos-label">{escape(label)}</div>'
                    f'<div class="bos-value">{escape(value)}</div>'
                    f'<div class="bos-meta">{escape(meta)}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_quickstart(bundle: dict | None) -> None:
    if bundle:
        steps = [
            ("Step 1", "Update folder", "Change or add files in Drive whenever needed."),
            ("Step 2", "Re-index", "Click Index BOS Folder to refresh the searchable library."),
            ("Step 3", "Ask clearly", "Use file names, campaigns, or keywords for better answers."),
        ]
    else:
        steps = [
            ("Step 1", "Use configured folder", "The app is pinned to the BOS Drive folder."),
            ("Step 2", "Click Index BOS Folder", "The app will read files and prepare search."),
            ("Step 3", "Start chatting", "Ask about assets once indexing finishes."),
        ]

    columns = st.columns(len(steps), gap="medium")
    for column, (label, value, meta) in zip(columns, steps):
        with column:
            st.markdown(
                (
                    '<div class="bos-quickstep">'
                    f'<div class="bos-quickstep-num">{escape(label)}</div>'
                    f'<div class="bos-quickstep-title">{escape(value)}</div>'
                    f'<div class="bos-quickstep-copy">{escape(meta)}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_overview(bundle: dict) -> None:
    summary_items = [
        ("Folder", escape(bundle["folder_name"])),
        ("Indexed At", escape(bundle["indexed_at"])),
        (
            "Drive Link",
            f'<a href="{escape(bundle["folder_link"])}" target="_blank">Open source folder</a>',
        ),
        ("OCR Mode", "Disabled"),
    ]

    cards = "".join(
        f"""
        <div class="bos-summary-card">
            <div class="bos-summary-label">{label}</div>
            <div class="bos-summary-value">{value}</div>
        </div>
        """
        for label, value in summary_items
    )
    st.markdown(f'<div class="bos-summary">{cards}</div>', unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="bos-empty">
            <strong>Welcome.</strong><br/>
            To start the conversation, share the target Drive folder with the service-account email in
            the sidebar, confirm the folder URL, and click <strong>Index BOS Folder</strong>.<br/><br/>
            After indexing finishes, you can ask questions about BOS assets, decks, Sheets, Docs,
            PDFs with readable text, and other supported text-based files.
            <ul class="bos-simple-list">
                <li>Share the folder with the service-account email</li>
                <li>Click <strong>Index BOS Folder</strong></li>
                <li>Ask your question</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict[str, str]]) -> None:
    for source in sources:
        link = (
            f'<div class="bos-source-link"><a href="{escape(source["link"])}" target="_blank">Open source file</a></div>'
            if source["link"]
            else '<div class="bos-source-link">No direct link</div>'
        )
        st.markdown(
            (
                '<div class="bos-source">'
                f'<div class="bos-source-title">{escape(source["label"])}</div>'
                f'<div class="bos-source-path">{escape(source["path"])}</div>'
                f'<div class="bos-source-meta">{escape(source["kind"])} · chunk {escape(source["chunk"])}</div>'
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
        meta = (
            f"{len(bundle['documents']):,} files indexed • "
            f"{len(bundle['chunks']):,} chunks ready"
        )
    else:
        title = "Awaiting index"
        meta = "Connect the Drive folder, then click Index BOS Folder to activate search."

    st.markdown(
        f"""
        <div class="bos-sidebar-card">
            <div class="bos-sidebar-card-label">BOS Workspace</div>
            <div class="bos-sidebar-card-value">{escape(title)}</div>
            <div class="bos-sidebar-card-meta">{escape(meta)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_skipped(bundle: dict | None) -> None:
    if not bundle or not bundle.get("skipped"):
        return

    with st.expander(f"Unable to index ({len(bundle['skipped'])})", expanded=False):
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
    st.markdown('<div class="bos-sidebar-title">BOS Setup</div>', unsafe_allow_html=True)
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
            with st.spinner("Indexing the BOS asset folder..."):
                progress_text = st.empty()
                progress_bar = st.progress(0)

                def _update_progress(current: int, total: int, label: str = "") -> None:
                    ratio = 0 if total <= 0 else current / total
                    progress_bar.progress(ratio)
                    if total <= 0:
                        progress_text.caption("Scanning BOS assets...")
                    else:
                        progress_text.caption(f"Indexed {current}/{total}: {label}")

                folder_payload = load_folder_index(
                    folder_url=folder_url.strip(),
                    progress_callback=_update_progress,
                )
                progress_bar.empty()
                progress_text.empty()
                st.session_state.folder_bundle = build_runtime_bundle(folder_payload)
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": (
                            f"I indexed **{folder_payload['folder_name']}** and I’m ready to answer "
                            "questions with file-backed citations."
                        ),
                    }
                ]
        except Exception as exc:
            st.error(str(exc))

bundle = st.session_state.folder_bundle

render_hero(bundle)
render_quickstart(bundle)

if bundle:
    render_metrics(bundle)
    render_overview(bundle)

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
