"""WAV loading and resampling to the 24 kHz mono PCM16 format Voice Agent sessions expect."""

from __future__ import annotations

import array
import sys
import wave
from pathlib import Path

from .settings import long_path

TARGET_RATE = 24000


def load_pcm16(path: Path | str, target_rate: int = TARGET_RATE) -> bytes:
    """Read a PCM WAV (8/16/32-bit, mono or stereo, any rate) and return mono 16-bit PCM at ``target_rate``."""
    with wave.open(long_path(path), "rb") as wav:
        channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        if wav.getcomptype() != "NONE":
            raise ValueError(f"{path}: compressed WAV is not supported")
        raw = wav.readframes(frames)
    if width == 2:
        samples = array.array("h", raw)
        if sys.byteorder == "big":
            samples.byteswap()
    elif width == 1:
        samples = array.array("h", ((b - 128) << 8 for b in raw))
    elif width == 4:
        wide = array.array("i", raw)
        if sys.byteorder == "big":
            wide.byteswap()
        samples = array.array("h", (v >> 16 for v in wide))
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    if channels > 1:
        samples = array.array("h", (sum(samples[i:i + channels]) // channels for i in range(0, len(samples), channels)))
    if rate != target_rate:
        samples = _resample(samples, rate, target_rate)
    out = samples.tobytes() if sys.byteorder == "little" else _swap(samples)
    return out


def _swap(samples: array.array) -> bytes:
    copy = array.array("h", samples)
    copy.byteswap()
    return copy.tobytes()


def _resample(samples: array.array, src_rate: int, dst_rate: int) -> array.array:
    """Linear-interpolation resampler (adequate for speech recognition input)."""
    if not samples:
        return samples
    ratio = src_rate / dst_rate
    n_out = int(len(samples) / ratio)
    last = len(samples) - 1
    out = array.array("h", bytes(2 * n_out))
    for i in range(n_out):
        x = i * ratio
        j = int(x)
        frac = x - j
        a = samples[j]
        b = samples[j + 1] if j < last else a
        out[i] = int(a + (b - a) * frac)
    return out


def duration_seconds(pcm: bytes, rate: int = TARGET_RATE) -> float:
    return len(pcm) / (2 * rate)
