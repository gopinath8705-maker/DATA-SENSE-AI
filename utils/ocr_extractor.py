"""
utils/ocr_extractor.py
Extract tabular data from images (PNG, JPG) and PDFs using OCR + AI fallback.
"""

import pandas as pd
import numpy as np
import io
import base64
import re
from typing import Tuple, Optional


def image_to_base64(image_bytes: bytes) -> str:
    """Convert raw image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_table_from_image_ai(image_bytes: bytes, api_key: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Use GPT-4o Vision to extract a table from an image.
    Returns (DataFrame, status_message).
    """
    if not api_key:
        return None, "no_api_key"

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        b64 = image_to_base64(image_bytes)
        # Detect image type from magic bytes
        if image_bytes[:4] == b'\x89PNG':
            media_type = "image/png"
        elif image_bytes[:2] in (b'\xff\xd8', b'\xff\xe0', b'\xff\xe1'):
            media_type = "image/jpeg"
        else:
            media_type = "image/png"

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL tabular data from this image as a CSV. "
                            "Output ONLY the CSV with a header row, nothing else. "
                            "Use comma as delimiter. If there is no table, output: NO_TABLE"
                        )
                    }
                ]
            }]
        )

        raw = response.choices[0].message.content.strip()

        if "NO_TABLE" in raw.upper():
            return None, "no_table_found"

        # Parse CSV from response
        df = pd.read_csv(io.StringIO(raw))
        return df, "success"

    except ImportError:
        return None, "openai_not_installed"
    except Exception as e:
        return None, f"error: {e}"


def extract_table_from_image_pytesseract(image_bytes: bytes) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fallback: Use pytesseract OCR to extract text from image, then parse as table.
    """
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)

        # Try to parse pipe-delimited or whitespace-delimited table
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return None, "insufficient_text"

        # Try pandas read_csv with whitespace
        try:
            df = pd.read_csv(io.StringIO(text), sep=r'\s{2,}', engine='python')
            if df.shape[1] >= 2 and df.shape[0] >= 1:
                return df, "success"
        except Exception:
            pass

        # Fallback: split each line by tabs or multiple spaces
        rows = [re.split(r'\t|\s{2,}', line) for line in lines]
        max_cols = max(len(r) for r in rows)
        rows = [r + [''] * (max_cols - len(r)) for r in rows]
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df, "success"

    except ImportError:
        return None, "pytesseract_not_installed"
    except Exception as e:
        return None, f"error: {e}"


def extract_table_from_pdf(pdf_bytes: bytes) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Extract tables from a PDF using pdfplumber.
    """
    try:
        import pdfplumber

        frames = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        frames.append(df)

        if not frames:
            return None, "no_tables_found"

        combined = pd.concat(frames, ignore_index=True)
        return combined, "success"

    except ImportError:
        return None, "pdfplumber_not_installed"
    except Exception as e:
        return None, f"error: {e}"


def smart_extract(uploaded_file, api_key: str = "") -> Tuple[Optional[pd.DataFrame], str]:
    """
    Route file to the best extraction method based on type.
    Returns (DataFrame, message).
    """
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith(".pdf"):
        df, status = extract_table_from_pdf(raw_bytes)
        if status == "success":
            return df, "✅ Table extracted from PDF successfully."
        return None, f"Could not extract table from PDF: {status}"

    elif name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        # Try AI vision first
        if api_key:
            df, status = extract_table_from_image_ai(raw_bytes, api_key)
            if status == "success":
                return df, "✅ Table extracted via AI Vision (GPT-4o)."

        # Fallback to pytesseract
        df, status = extract_table_from_image_pytesseract(raw_bytes)
        if status == "success":
            return df, "✅ Table extracted via OCR (pytesseract)."

        return None, (
            "Could not extract a table from this image. "
            "For best results, add an OpenAI API key (uses GPT-4o Vision) "
            "or ensure pytesseract is installed."
        )

    return None, "Unsupported file type for OCR extraction."
