# BOS Assets Chatbot

Separate Streamlit app for chatting with a single Google Drive folder of BOS brand assets.

## What it does

- Connects to one Drive folder using a Google service account
- Recursively indexes supported files inside that folder
- Chunks extracted text for retrieval
- Uses OpenRouter for final answer generation
- Shows source-backed citations for each answer
- Can OCR image files and scanned PDFs when OCR is enabled

## Supported file types

- Google Docs
- Google Sheets
- Google Slides
- PDF
- DOCX
- PPTX
- XLSX
- CSV / TSV
- TXT / MD / JSON
- SVG
- PNG / JPG / JPEG / WEBP / GIF / BMP / TIFF via OCR

## Setup

1. Create a Python environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY`.
3. Put your Google service account JSON at `credentials.json`, or set `GOOGLE_CREDENTIALS_PATH`.
4. Share the target Drive folder with the service account email.
5. Run `streamlit run app.py`.

## Streamlit Cloud

Add these app secrets in Streamlit Cloud:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `OCR_ENABLED`
- `OCR_MODEL`
- `DRIVE_FOLDER_URL`
- `google_service_account` or `gcp_service_account`

An example file is included at `.streamlit/secrets.toml.example`.

## Notes

- This app is intentionally separate from the sheets chatbot project.
- It is designed for one folder at a time, not for cross-project indexing.
- OCR uses OpenRouter during indexing, so scanned PDFs and image-heavy folders can increase indexing time and API cost.
