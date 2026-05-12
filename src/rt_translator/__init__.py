"""BreezeMate · 微伴 -- realtime offline speech subtitling and translation.

Package name is kept as ``rt_translator`` for backwards compatibility
with existing installs / saved configs / cached vosk model directories.
The user-facing product name is BreezeMate.
"""

import os as _os
import sys as _sys
import warnings as _warnings

# Soundcard fires SoundcardRuntimeWarning("data discontinuity in recording")
# whenever WASAPI hands us a chunk whose timestamp doesn't line up with the
# previous one. It is cosmetic -- a sub-millisecond gap -- and we can't act
# on it. Silence it at the *class* level on every platform so it does not
# pollute the log file or stderr (which would scramble rich.live).
for _backend in ("mediafoundation", "pulseaudio", "coreaudio"):
    try:
        _mod = __import__(f"soundcard.{_backend}", fromlist=["SoundcardRuntimeWarning"])
        _warnings.simplefilter("ignore", _mod.SoundcardRuntimeWarning)
    except Exception:
        pass

# Belt-and-suspenders message-based filter (covers any future backend or
# if soundcard's class import path changes).
_warnings.filterwarnings("ignore", message=r"data discontinuity in recording")

# numpy 2.x on some Windows builds hits an OpenBLAS thread-allocation
# failure when default BLAS thread pool is spun up. Our numpy use is
# lightweight (resampling 50 ms PCM frames), so a single BLAS thread is
# plenty. Set this BEFORE any numpy import. Users can still override by
# exporting the variable explicitly.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

# On Chinese / Japanese / Korean locales the Windows console defaults to a
# legacy codepage (cp936 / cp932 / cp949) which can't encode characters
# like (R), (TM), or punctuation found in device names returned by WASAPI.
# Force UTF-8 on the stdio streams so rich's renderer never trips over
# them. errors='replace' keeps us safe even on a terminal that can't draw
# the resulting bytes.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(_sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

__version__ = "0.1.0"
