from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import sounddevice as sd
import whisper

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.constants import START

# LangGraph imports
from langgraph.graph import StateGraph, END, MessagesState

# ============ TTS (pyttsx3) utilities ============
import atexit
try:
    import pyttsx3  # type: ignore
except Exception as _e:
    pyttsx3 = None  # fallback if not installed yet
    print(f"[Voice] pyttsx3 not available: {_e}", file=sys.stderr)

_TTS_ENGINE = None
_TTS_LOCK = threading.Lock()

def _get_tts_engine():
    global _TTS_ENGINE
    if _TTS_ENGINE is not None:
        return _TTS_ENGINE
    if pyttsx3 is None:
        return None
    try:
        eng = pyttsx3.init()
        # Attempt to set a reasonable rate; ignore if unsupported
        try:
            rate = eng.getProperty("rate")
            if isinstance(rate, int):
                eng.setProperty("rate", max(150, min(200, rate)))
        except Exception:
            pass
        _TTS_ENGINE = eng

        def _cleanup():
            try:
                if _TTS_ENGINE is not None:
                    _TTS_ENGINE.stop()
            except Exception:
                pass
        atexit.register(_cleanup)
        return _TTS_ENGINE
    except Exception as e:
        print(f"[Voice] Failed to initialize TTS engine: {e}", file=sys.stderr)
        return None

def _tts_speak_blocking(text: str):
    """Speak the given text using pyttsx3; blocks until finished. Thread-safe via lock."""
    text = (text or "").strip()
    if not text:
        return
    eng = _get_tts_engine()
    if eng is None:
        return
    with _TTS_LOCK:
        try:
            eng.say(text)
            eng.runAndWait()
        except Exception as e:
            print(f"[Voice] TTS speak error: {e}", file=sys.stderr)


def init_llm(model: str, temperature: float, max_tokens: Optional[int]) -> ChatOpenAI:
    """Initialize ChatOpenAI client configured to use OpenRouter.

    Follows guidance from https://openrouter.ai/docs/community/lang-chain
    and mirrors conventions used elsewhere in this repo.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1"

    # Recommended extra headers
    default_headers: Dict[str, str] = {}
    site = os.environ.get("OPENROUTER_SITE_URL")
    app = os.environ.get("OPENROUTER_APP_NAME")
    if site:
        default_headers["HTTP-Referer"] = site
    if app:
        default_headers["X-Title"] = app

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "base_url": base_url,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if default_headers:
        kwargs["default_headers"] = default_headers
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)

    return ChatOpenAI(**kwargs)


def build_graph(llm: ChatOpenAI):
    """Build a simple LangGraph: START -> model -> END."""
    graph = StateGraph(MessagesState)

    async def call_model(state: MessagesState):
        # LLM is a ChatModel; pass the conversation so far and get one response
        response = await llm.ainvoke(state["messages"])  # returns an AIMessage
        return {"messages": [response]}

    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)

    return graph.compile()


# ============ Whisper mic input utilities ============

_WHISPER_CACHE: Dict[str, Any] = {}


def _get_whisper_model(name: str):
    name = name or "base"
    if name in _WHISPER_CACHE:
        return _WHISPER_CACHE[name]
    try:
        model = whisper.load_model(name)
        _WHISPER_CACHE[name] = model
        return model
    except Exception as e:
        print(f"[Voice] Failed to load Whisper model '{name}': {e}", file=sys.stderr)
        raise


def transcribe_from_mic_until_enter(whisper_model: str = "base", sample_rate: int = 16000, language: str = "ru", volume_threshold: float = 0.02) -> str:
    """Capture audio from microphone, show live Whisper transcription, stop on Enter and return final text.

    - Uses sounddevice to stream mic audio into a buffer.
    - In parallel, repeatedly runs Whisper on the accumulated buffer and prints incremental text to the same line.
    - When the user presses Enter, stops recording, prints a newline, and returns the final transcription.
    """
    model = _get_whisper_model(whisper_model)

    buf_lock = threading.Lock()
    chunks: List[np.ndarray] = []
    stop_event = threading.Event()

    def _enter_watcher():
        try:
            # Read one line; Enter will end it
            sys.stdin.readline()
        except Exception:
            pass
        finally:
            stop_event.set()

    enter_thread = threading.Thread(target=_enter_watcher, daemon=True)
    enter_thread.start()

    def _audio_callback(indata, frames, time_info, status):  # noqa: D401 unused-args
        if status:
            print(f"\n[Voice] Audio status: {status}", file=sys.stderr)
        mono = indata
        try:
            if mono.ndim > 1:
                mono = mono[:, 0]
        except Exception:
            pass
        mono = np.asarray(mono, dtype=np.float32).copy()
        # Volume gate: append only if RMS >= threshold
        try:
            rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        except Exception:
            rms = 0.0
        if rms >= float(volume_threshold):
            with buf_lock:
                chunks.append(mono)

    # Start mic stream
    try:
        stream = sd.InputStream(callback=_audio_callback, channels=1, samplerate=sample_rate, dtype="float32")
        stream.start()
    except Exception as e:
        print(f"[Voice] Could not open microphone stream: {e}", file=sys.stderr)
        return ""

    last_printed = ""
    print("You (speak, press Enter to send): ", end="", flush=True)

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
            with buf_lock:
                if not chunks:
                    continue
                audio = np.concatenate(chunks, axis=0)
            try:
                # Whisper expects 16kHz mono float32; we record as such
                result = model.transcribe(audio, fp16=False, language=language, task="transcribe")
                text = (result.get("text") or "").strip()
                if text and text != last_printed:
                    # Update the same line
                    print("\rYou (speak, press Enter to send): " + text, end="", flush=True)
                    last_printed = text
            except Exception as e:
                print(f"\n[Voice] Transcribe (live) error: {e}", file=sys.stderr)
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    # Final transcription
    print()  # newline after Enter
    with buf_lock:
        audio = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.float32)
    if audio.size == 0:
        return ""
    try:
        result = model.transcribe(audio, fp16=False, language=language, task="transcribe")
        final_text = (result.get("text") or "").strip()
        return final_text
    except Exception as e:
        print(f"[Voice] Transcribe (final) error: {e}", file=sys.stderr)
        return last_printed or ""


async def stream_once(app, messages: List[BaseMessage]) -> str:
    """Stream a single turn through the graph and print tokens as they arrive.

    Returns the full assistant response content.
    """
    full_text_parts: List[str] = []

    # Stream internal events; token chunks will come via chat model stream events
    async for event in app.astream_events({"messages": messages}, version="v2"):
        ev = event.get("event")
        if ev in ("on_chat_model_stream", "on_llm_stream"):
            data = event.get("data", {})
            chunk = data.get("chunk")
            # chunk may be AIMessageChunk or LLMResult chunk; try to get text
            text = None
            if hasattr(chunk, "content"):
                text = chunk.content
            elif isinstance(chunk, dict):
                text = chunk.get("content")
            if text:
                full_text_parts.append(str(text))
                print(str(text), end="", flush=True)
        elif ev in ("on_chat_model_end", "on_llm_end"):
            # End of streaming for this model call; print newline if anything was printed
            if full_text_parts:
                print()
    return "".join(full_text_parts)


async def interactive_chat(app, system_prompt: str, whisper_model: str = "base", sample_rate: int = 16000, language: str = "ru", volume_threshold: float = 0.02):
    print("Interactive voice chat started. Speak and press Enter to send. Press Ctrl+C to exit.")
    history: List[BaseMessage] = [SystemMessage(system_prompt)] if system_prompt else []
    try:
        while True:
            text = transcribe_from_mic_until_enter(
                whisper_model=whisper_model,
                sample_rate=sample_rate,
                language=language,
                volume_threshold=volume_threshold,
            )
            text = (text or "").strip()
            if not text:
                continue
            history.append(HumanMessage(text))
            print("Assistant: ", end="", flush=True)
            reply = await stream_once(app, history)
            # Speak the reply via TTS
            try:
                await asyncio.to_thread(_tts_speak_blocking, reply)
            except Exception as e:
                print(f"[Voice] TTS error: {e}", file=sys.stderr)
            # Persist assistant message in history for multi-turn context
            from langchain_core.messages import AIMessage
            history.append(AIMessage(content=reply))
    except KeyboardInterrupt:
        print("\nExiting chat.")


async def single_turn(app, system_prompt: str, message: str):
    messages: List[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(system_prompt))
    messages.append(HumanMessage(message))
    print("Assistant: ", end="", flush=True)
    reply = await stream_once(app, messages)
    try:
        await asyncio.to_thread(_tts_speak_blocking, reply)
    except Exception as e:
        print(f"[Voice] TTS error: {e}", file=sys.stderr)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async streaming voice chat agent using LangGraph, OpenRouter, and Whisper"
    )
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Model ID for OpenRouter")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for the response (model-dependent limits apply)",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful AI assistant.",
        help="System prompt/instructions",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="If provided, run a single-turn chat for this message instead of interactive REPL",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model to use for speech recognition (e.g., tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate (Hz), default 16000",
    )
    parser.add_argument(
        "--language",
        default="ru",
        help="Language code for Whisper transcription (e.g., ru). Default: ru",
    )
    parser.add_argument(
        "--volume-threshold",
        type=float,
        default=0.02,
        help="RMS threshold to accept mic frames (0..1). Higher = more strict. Default: 0.02",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    llm = init_llm(args.model, args.temperature, args.max_tokens)

    app = build_graph(llm)

    async def runner():
        if args.message:
            await single_turn(app, args.system, args.message)
        else:
            await interactive_chat(
                app,
                args.system,
                whisper_model=args.whisper_model,
                sample_rate=args.sample_rate,
                language=args.language,
                volume_threshold=args.volume_threshold,
            )

    try:
        asyncio.run(runner())
    except RuntimeError as e:
        # Fallback for nested event loops if any environment already has a loop
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(runner())
        except Exception:
            print(f"[ERROR] Async runtime failed: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
