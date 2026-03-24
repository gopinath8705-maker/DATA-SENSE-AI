"""
utils/voice_query.py
Voice query support — record audio in browser, transcribe, then answer.
Uses Streamlit's audio recorder component + OpenAI Whisper for transcription.
"""

import streamlit as st
import tempfile
import os
from typing import Optional, Tuple


def render_voice_input() -> Optional[str]:
    """
    Render voice recording UI and return transcribed text (or None).
    Uses streamlit-audio-recorder if available, else shows manual upload fallback.
    """
    transcribed = None

    # Try streamlit-audio-recorder
    try:
        from audiorecorder import audiorecorder
        audio = audiorecorder("🎙️ Click to Record", "⏹️ Stop Recording", key="voice_recorder")

        if len(audio) > 0:
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio.export(tmp.name, format="wav")
                tmp_path = tmp.name

            st.audio(tmp_path)
            st.session_state["last_audio_path"] = tmp_path
            return tmp_path  # Will be transcribed separately

    except ImportError:
        pass

    # Fallback: file uploader
    st.caption("🎙️ Upload an audio file (MP3/WAV) to use voice queries.")
    audio_file = st.file_uploader(
        "Upload audio query",
        type=["wav", "mp3", "m4a", "ogg"],
        key="voice_upload",
        label_visibility="collapsed"
    )

    if audio_file:
        with tempfile.NamedTemporaryFile(suffix=f".{audio_file.name.split('.')[-1]}", delete=False) as tmp:
            tmp.write(audio_file.read())
            return tmp.name

    return None


def transcribe_audio(audio_path: str, api_key: str) -> Tuple[Optional[str], str]:
    """
    Transcribe audio file using OpenAI Whisper.
    Returns (transcription_text, status_message).
    """
    if not api_key:
        return None, "⚠️ OpenAI API key required for voice transcription."

    if not audio_path or not os.path.exists(audio_path):
        return None, "Audio file not found."

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        text = transcript.strip() if isinstance(transcript, str) else transcript.text.strip()
        return text, f"✅ Transcribed: *\"{text}\"*"

    except ImportError:
        return None, "OpenAI package not installed."
    except Exception as e:
        return None, f"Transcription error: {e}"


def simulate_voice_demo(question: str = "Which column has the highest average value?") -> str:
    """Return a demo question for users without a microphone or API key."""
    return question
