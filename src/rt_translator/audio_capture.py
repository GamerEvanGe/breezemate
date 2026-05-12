"""Threaded audio capture.

Supports two sources via the same ``soundcard`` API:

* ``loopback`` (Speaker.recorder) -- WASAPI loopback on Windows, captures
  whatever the speakers are currently playing.
* ``mic`` (Microphone.recorder) -- standard microphone / line-in capture.

In either case we:

1. Resample to 16 kHz mono via ``soxr.ResampleStream`` (streaming-safe).
2. Clip and convert to int16 PCM (little-endian by default on x86/x64).
3. Push raw bytes into an ``asyncio.Queue`` on the main event loop via
   ``run_coroutine_threadsafe`` -- never call into asyncio directly from
   the recorder thread.

The recorder thread is daemonic so a hung ``stop()`` won't keep the
process alive at shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import threading
import warnings
from typing import Callable, Optional

import numpy as np
import soundcard as sc
import soxr

from .config import AudioConfig
from .device_picker import DeviceInfo
from .events import ConnectionStatus, PipelineEvent

log = logging.getLogger(__name__)

TARGET_RATE = 16_000
TARGET_CHANNELS = 1


class AudioCaptureError(RuntimeError):
    pass


# Process-wide guard so two AudioCapture instances can never have
# overlapping live recorder threads. See class docstring for the
# crash-history rationale.
_ACTIVE_RECORDER_LOCK = threading.Lock()
_ACTIVE_RECORDER: "Optional[threading.Thread]" = None


def _resolve_device(info: DeviceInfo):
    """Return the underlying soundcard ``_Microphone`` object for a DeviceInfo.

    Both ``loopback`` and ``mic`` sources are exposed by soundcard as
    ``_Microphone`` instances -- the only difference is the ``isloopback``
    flag and the API used to enumerate them:

    * ``loopback``: ``sc.get_microphone(id, include_loopback=True)`` which
      returns the WASAPI loopback mic for a speaker.
    * ``mic``: ``sc.get_microphone(id)`` which returns a normal capture
      device.
    """
    is_loopback = info.source == "loopback"

    # First try exact id lookup.
    try:
        return sc.get_microphone(info.id, include_loopback=is_loopback)
    except Exception:
        pass

    # ID match failed (device unplugged or id format changed across
    # soundcard versions). Fall back to name match against the live list,
    # then the system default.
    try:
        for m in sc.all_microphones(include_loopback=is_loopback):
            if bool(getattr(m, "isloopback", False)) != is_loopback:
                continue
            if m.name == info.name:
                return m
    except Exception:
        pass

    if is_loopback:
        # Use the default speaker's loopback mic.
        default_spk_name = sc.default_speaker().name
        for m in sc.all_microphones(include_loopback=True):
            if getattr(m, "isloopback", False) and m.name == default_spk_name:
                return m
        # Last resort: the first loopback mic we can find.
        for m in sc.all_microphones(include_loopback=True):
            if getattr(m, "isloopback", False):
                return m
        raise RuntimeError("No loopback microphone available on this system.")

    return sc.default_microphone()


class AudioCapture:
    """Background-threaded audio capture producing 16 kHz mono int16 bytes.

    Only one ``AudioCapture`` instance is allowed to have a live
    recorder thread at a time *process-wide*. This is enforced by
    ``_ACTIVE_RECORDER`` -- a class-level slot that holds a weakref
    to the current owner. If a previous pipeline's recorder thread
    is still alive (e.g. stuck in a slow ``recorder_cm.record()`` C
    call) and a new pipeline tries to start, ``start()`` will raise
    rather than opening a second WASAPI recorder on the same device.
    That second recorder is exactly what produced the BEX64 /
    0xc0000409 CRT fast-fail crashes we used to see on quick
    stop/start cycles: two threads racing inside MediaFoundation
    corrupted the heap, and ucrtbase's invariant checker tripped a
    few seconds later.

    ``tee_callback`` (optional) is invoked on the recorder thread with
    every PCM chunk in addition to the chunk being pushed onto the
    asyncio queue. This is how the local Vosk preview ASR receives
    audio without needing its own capture pipeline.

    The callback MUST be non-blocking (queue.put_nowait, etc.). It runs
    on the audio capture thread; a slow callback will starve the
    recorder and trigger ``data discontinuity`` warnings.
    """

    def __init__(
        self,
        cfg: AudioConfig,
        device: DeviceInfo,
        queue: "asyncio.Queue[bytes]",
        status_queue: "Optional[asyncio.Queue[PipelineEvent]]" = None,
        tee_callback: "Optional[Callable[[bytes], None]]" = None,
    ) -> None:
        self.cfg = cfg
        self.device_info = device
        self.queue = queue
        self.status_queue = status_queue
        self.tee_callback = tee_callback
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.last_exception: Optional[BaseException] = None

    @property
    def description(self) -> str:
        return f"{self.device_info.source}:{self.device_info.name}"

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        global _ACTIVE_RECORDER
        with _ACTIVE_RECORDER_LOCK:
            # Cross-instance check: is *any* recorder thread (possibly
            # one owned by a previous AudioCapture instance whose
            # stop() returned with a stuck record() call) still alive?
            prev = _ACTIVE_RECORDER
            if prev is not None and prev.is_alive():
                raise RuntimeError(
                    "AudioCapture: a previous recorder thread is still running. "
                    "Wait for it to exit before starting a new capture (this "
                    "guard prevents the WASAPI race that caused the BEX64 / "
                    "0xc0000409 crashes)."
                )
            # Same-instance sanity check (mostly redundant with the
            # process-wide one, but keeps the existing error if a
            # caller mis-uses a single instance).
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(
                    "AudioCapture.start called while this instance's recorder "
                    "thread is still alive."
                )
            self._thread = None
            self._loop = loop
            self._stop_event.clear()
            t = threading.Thread(
                target=self._run, name="rt-audio-capture", daemon=True
            )
            self._thread = t
            _ACTIVE_RECORDER = t
            t.start()
        log.info("Audio capture started (%s)", self.description)

    def stop(self) -> None:
        """Signal the recorder thread to stop and block until it dies.

        We give the recorder a generous timeout (10 s by default).
        ``soundcard.recorder.record()`` is a blocking C call into
        WASAPI/MediaFoundation; on most systems it returns within
        50-100 ms once ``_stop_event`` is checked, but a stalled
        device or a slow driver can occasionally hold it for longer.
        If the thread STILL hasn't exited by then we deliberately
        keep the reference alive instead of dropping it -- that
        prevents start() from kicking off a second recorder on the
        same WASAPI device, which would otherwise race the zombie
        thread and corrupt the heap (CRT __fastfail in ucrtbase).
        """
        global _ACTIVE_RECORDER
        self._stop_event.set()
        t = self._thread
        if t is None:
            return
        # 10 s is well above any normal Windows audio API latency.
        # We've seen rare 2-3 s stalls in the wild; 10 s gives them
        # plenty of headroom while still bounding shutdown time.
        t.join(timeout=10.0)
        with _ACTIVE_RECORDER_LOCK:
            if t.is_alive():
                log.error(
                    "Audio capture thread did not exit within 10 s; keeping "
                    "the process-wide active-recorder reference so the next "
                    "start() refuses instead of creating a zombie. Source=%s",
                    self.description,
                )
                # NB: do NOT clear self._thread or _ACTIVE_RECORDER --
                # leaving them set is what causes the next start() to
                # raise instead of opening a second WASAPI handle.
                return
            self._thread = None
            if _ACTIVE_RECORDER is t:
                _ACTIVE_RECORDER = None
        log.info("Audio capture stopped")

    def _run(self) -> None:
        com_initialised = False
        if platform.system() == "Windows":
            try:
                import pythoncom

                pythoncom.CoInitialize()
                com_initialised = True
            except Exception as e:
                log.debug("CoInitialize failed (will continue): %s", e)

        try:
            # Soundcard chatters with SoundcardRuntimeWarning during loopback
            # whenever WASAPI hands us a discontinuous chunk. The package-level
            # warning filters should catch them, but this thread-local
            # catch_warnings is the final safety net so they can never leak
            # to stderr / the live UI.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._capture_loop()
        except Exception as e:
            self.last_exception = e
            log.exception("Audio capture thread crashed: %s", e)
            self._post_status(
                ConnectionStatus(
                    state="error",
                    detail=f"audio capture: {type(e).__name__}: {e}",
                )
            )
        finally:
            if com_initialised:
                try:
                    import pythoncom

                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def _capture_loop(self) -> None:
        device = _resolve_device(self.device_info)

        # Negotiate working config before entering the with-block. _open_recorder
        # only constructs the cm; it doesn't enter it, so we can still try
        # fallbacks if __enter__ raises.
        last_err: Optional[BaseException] = None
        recorder_cm = None
        actual_rate = actual_channels = actual_block = 0

        is_loopback = self.device_info.source == "loopback"
        device_channels = getattr(device, "channels", None)
        # WASAPI's engine period is ~10ms; using blocksize=10ms makes us race
        # against scheduling jitter and triggers SoundcardRuntimeWarning(
        # "data discontinuity"). Use ~50ms blocks to give the capture thread
        # plenty of slack while still feeling instantaneous.
        block_frames_loopback = 2_400      # 50ms @ 48kHz
        block_frames_mic = 1_600           # 100ms @ 16kHz (mic devices tend
                                           # to deliver larger native buffers)

        candidates = (
            [
                (48_000, 2, block_frames_loopback),
                (48_000, device_channels, block_frames_loopback),
                (44_100, 2, int(44_100 * 0.05)),
                (44_100, device_channels, int(44_100 * 0.05)),
            ]
            if is_loopback
            else [
                (16_000, 1, block_frames_mic),
                (48_000, 1, int(48_000 * 0.1)),
                (44_100, 1, int(44_100 * 0.1)),
            ]
        )

        for rate, ch, bsz in candidates:
            if ch is None or ch <= 0:
                continue
            kwargs = {"samplerate": rate, "channels": ch, "blocksize": bsz}
            try:
                log.info("Opening recorder %s on %s", kwargs, self.description)
                cm = device.recorder(**kwargs)
                # Probe by entering; if it fails, try next candidate.
                cm.__enter__()
                recorder_cm = cm
                actual_rate, actual_channels, actual_block = rate, ch, bsz
                log.info(
                    "Recorder opened: rate=%d channels=%d blocksize=%d",
                    actual_rate, actual_channels, actual_block,
                )
                break
            except Exception as e:
                last_err = e
                log.warning("Recorder open failed for %s: %r", kwargs, e)
                continue

        if recorder_cm is None:
            raise RuntimeError(
                f"Could not open recorder for {self.description}. "
                f"Last error: {last_err!r}"
            )

        resampler = (
            soxr.ResampleStream(actual_rate, TARGET_RATE, 1, dtype="float32")
            if actual_rate != TARGET_RATE
            else None
        )

        try:
            while not self._stop_event.is_set():
                data = recorder_cm.record(numframes=actual_block)
                if data is None or data.size == 0:
                    continue

                if data.ndim > 1 and data.shape[1] > 1:
                    mono = data.mean(axis=1).astype(np.float32, copy=False)
                else:
                    mono = data.reshape(-1).astype(np.float32, copy=False)

                if resampler is not None:
                    resampled = resampler.resample_chunk(mono, last=False)
                else:
                    resampled = mono

                if resampled.size == 0:
                    continue

                clipped = np.clip(resampled, -1.0, 1.0)
                pcm16 = (clipped * 32767.0).astype(np.int16)
                payload = pcm16.tobytes()
                self._enqueue(payload)
                # Fan-out to the local preview ASR if one was wired up.
                # Callback must be non-blocking; we still wrap in
                # try/except so a buggy hook can't kill capture.
                tee = self.tee_callback
                if tee is not None:
                    try:
                        tee(payload)
                    except Exception:
                        log.exception("Audio tee callback raised; ignoring")
        finally:
            try:
                recorder_cm.__exit__(None, None, None)
            except Exception:
                pass

    def _post_status(self, ev: PipelineEvent) -> None:
        if self.status_queue is None or self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.status_queue.put(ev), self._loop)
        except Exception:
            pass

    def _enqueue(self, payload: bytes) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.queue.put(payload), loop)
        except RuntimeError:
            # Loop was closed between checks; harmless on shutdown.
            pass
        except Exception as e:
            log.warning("Failed to enqueue audio chunk: %s", e)
