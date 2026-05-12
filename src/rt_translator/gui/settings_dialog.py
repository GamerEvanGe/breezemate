"""Settings dialog: provider profiles, API keys, ASR & translator config.

The dialog is *purely a view*: it copies the incoming ``AppConfig`` into
local widgets, lets the user edit, and on ``Accept`` returns a new
``AppConfig`` plus mutates ``SecretStore`` for any keys the user typed.
The caller is responsible for persisting the new config and restarting
the pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    ASRConfig,
    AppConfig,
    LocalASRConfig,
    ProviderEndpoint,
    SubtitleWindowConfig,
    TranslatorConfig,
)
from ..providers.asr import vosk_model
from ..providers.presets import (
    ProviderPreset,
    chat_presets,
)
from ..secrets import get_secret_store

log = logging.getLogger(__name__)


class _ColorButton(QPushButton):
    """Small swatch button that opens ``QColorDialog`` on click.

    The button face is painted with the current colour and the hex
    value is shown as the label, so the user can see at a glance both
    what they picked and what string will be persisted to the config.

    Use ``current_color()`` to read the latest hex value when saving.
    """

    color_changed = Signal(str)

    def __init__(self, initial_hex: str, parent=None) -> None:
        super().__init__(parent)
        self._hex = self._normalise(initial_hex)
        self.setMinimumWidth(110)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._open_dialog)
        self._refresh()

    @staticmethod
    def _normalise(hex_rgb: str) -> str:
        c = QColor(hex_rgb)
        if not c.isValid():
            c = QColor("#ffffff")
        return c.name(QColor.NameFormat.HexRgb)  # "#rrggbb"

    def current_color(self) -> str:
        return self._hex

    def set_color(self, hex_rgb: str) -> None:
        new_hex = self._normalise(hex_rgb)
        if new_hex == self._hex:
            return
        self._hex = new_hex
        self._refresh()
        self.color_changed.emit(self._hex)

    def _open_dialog(self) -> None:
        dlg = QColorDialog(QColor(self._hex), self)
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, False)
        if dlg.exec() == QColorDialog.DialogCode.Accepted:
            picked = dlg.selectedColor()
            if picked.isValid():
                self.set_color(picked.name(QColor.NameFormat.HexRgb))

    def _refresh(self) -> None:
        c = QColor(self._hex)
        # Auto-pick a readable text colour for the swatch so the hex
        # string stays legible regardless of which colour was chosen.
        luminance = (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()) / 255.0
        text = "#000" if luminance > 0.55 else "#fff"
        self.setText(self._hex)
        self.setStyleSheet(
            f"QPushButton {{ background-color: {self._hex}; color: {text}; "
            f"border: 1px solid #555; padding: 4px 10px; border-radius: 4px; "
            f"font-family: 'Consolas','Menlo',monospace; }}"
        )


class _VoskDownloadWorker(QThread):
    """QThread that calls vosk_model.download_model with progress signals.

    Lives only for the duration of one download. Cancellation goes via
    ``requestInterruption()`` -- we check that flag inside the progress
    callback and raise to abort the urllib stream.
    """

    progress = Signal(int, int)  # downloaded, total
    error = Signal(str)
    finished_ok = Signal(str)  # model path

    def __init__(self, model_id: str, parent=None) -> None:
        super().__init__(parent)
        self._model_id = model_id

    def run(self) -> None:
        def cb(downloaded: int, total: int) -> None:
            if self.isInterruptionRequested():
                raise RuntimeError("Download cancelled by user")
            self.progress.emit(downloaded, total)

        try:
            path = vosk_model.download_model(
                self._model_id, progress=cb, force=True
            )
        except Exception as e:
            log.exception("Vosk download failed")
            self.error.emit(str(e))
            return
        self.finished_ok.emit(str(path))


class _ProviderPanel(QWidget):
    """Translator / chat provider row: preset + key + base URL + model.

    The ASR backend is no longer chosen here -- speech recognition runs
    exclusively on the embedded offline Vosk engine (see the dedicated
    "语音识别" tab). This panel is therefore single-purpose now.
    """

    def __init__(
        self,
        title: str,
        available: list[ProviderPreset],
        current_profile: str,
        current_model: str,
        current_endpoint: ProviderEndpoint,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._available = available

        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.preset_combo = QComboBox(self)
        for p in available:
            self.preset_combo.addItem(p.label, userData=p.id)
        idx = max(0, next((i for i, p in enumerate(available) if p.id == current_profile), 0))
        self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        self.notes_label = QLabel("")
        self.notes_label.setWordWrap(True)
        self.notes_label.setStyleSheet("color: #888;")

        self.base_url_edit = QLineEdit(current_endpoint.base_url)
        self.base_url_edit.setPlaceholderText("https://...")

        self.api_key_edit = QLineEdit("")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("(留空 = 沿用已保存的 key)")

        self.show_key_btn = QPushButton("显示")
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self._toggle_key_visibility)
        key_row = QHBoxLayout()
        key_row.addWidget(self.api_key_edit, 1)
        key_row.addWidget(self.show_key_btn, 0)
        key_row_w = QWidget(self)
        key_row_w.setLayout(key_row)
        key_row.setContentsMargins(0, 0, 0, 0)

        self.signup_btn = QPushButton("获取 API Key")
        self.signup_btn.clicked.connect(self._open_signup_url)

        self.model_combo = QComboBox(self)
        self.model_combo.setEditable(True)
        self.model_combo.setCurrentText(current_model)

        # Title row.
        layout.addRow(QLabel(f"<b>{title}</b>"))
        layout.addRow("服务商:", self.preset_combo)
        layout.addRow("", self.notes_label)
        layout.addRow("API Base URL:", self.base_url_edit)
        layout.addRow("API Key:", key_row_w)
        layout.addRow("", self.signup_btn)
        layout.addRow("模型:", self.model_combo)

        self._on_preset_changed(self.preset_combo.currentIndex())
        # Pre-fill key placeholder if one is already stored.
        store = get_secret_store()
        env_name = current_endpoint.api_key_env
        if store.has(env_name):
            self.api_key_edit.setPlaceholderText("(已保存。留空表示不修改)")

        # Re-apply the user's current model after preset-change wiped it.
        if current_model:
            self.model_combo.setCurrentText(current_model)

    def _toggle_key_visibility(self, on: bool) -> None:
        self.api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
        )
        self.show_key_btn.setText("隐藏" if on else "显示")

    def _on_preset_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._available):
            return
        preset = self._available[idx]
        # Only override base_url / model dropdown when the user hasn't
        # already typed something custom for them. Specifically: if the
        # base_url is empty or matches *some* preset, swap to the new
        # one's default; otherwise leave the user's value alone.
        existing = self.base_url_edit.text().strip()
        if not existing or any(existing == p.base_url for p in self._available):
            self.base_url_edit.setText(preset.base_url)

        self.api_key_edit.setEnabled(preset.auth_required)
        self.show_key_btn.setEnabled(preset.auth_required)
        self.signup_btn.setEnabled(bool(preset.signup_url))

        # Refresh suggested-model dropdown. Always preserve whatever the
        # user typed -- the combobox is editable, so a model not in the
        # preset list is still a valid choice.
        current_text = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for m in preset.suggested_chat_models or ():
            self.model_combo.addItem(m)
        if current_text:
            self.model_combo.setCurrentText(current_text)
        self.model_combo.blockSignals(False)

        self.notes_label.setText(preset.notes or "")
        self.signup_btn.setEnabled(bool(preset.signup_url))
        self.signup_btn.setVisible(bool(preset.signup_url))

        store = get_secret_store()
        if store.has(preset.api_key_env):
            self.api_key_edit.setPlaceholderText(
                f"(已保存到 {preset.api_key_env}。留空表示不修改)"
            )
        else:
            self.api_key_edit.setPlaceholderText(
                f"(将保存为 {preset.api_key_env})"
            )

    def _open_signup_url(self) -> None:
        idx = self.preset_combo.currentIndex()
        if 0 <= idx < len(self._available):
            url = self._available[idx].signup_url
            if url:
                QDesktopServices.openUrl(QUrl(url))

    def current_preset(self) -> ProviderPreset:
        return self._available[self.preset_combo.currentIndex()]

    def current_model(self) -> str:
        return self.model_combo.currentText().strip()

    def current_base_url(self) -> str:
        return self.base_url_edit.text().strip()

    def current_api_key(self) -> str:
        return self.api_key_edit.text()


class SettingsDialog(QDialog):
    """Modal settings editor. Use ``exec()`` and read ``result_config()``."""

    config_saved = Signal(AppConfig)

    def __init__(self, cfg: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BreezeMate · 微伴 — 设置")
        self.setMinimumWidth(560)
        self._cfg = cfg.model_copy(deep=True)

        outer = QVBoxLayout(self)
        tabs = QTabWidget(self)
        outer.addWidget(tabs, 1)

        tabs.addTab(self._build_general_tab(), "通用")
        tabs.addTab(self._build_translator_tab(), "翻译模型")
        tabs.addTab(self._build_asr_tab(), "语音识别")
        tabs.addTab(self._build_subtitle_tab(), "字幕浮窗")

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    # ------------------------------------------------------------------ Tabs

    def _build_general_tab(self) -> QWidget:
        w = QWidget(self)
        form = QFormLayout(w)

        self.mode_combo = QComboBox(w)
        self.mode_combo.addItem("仅外语字幕 (ASR only)", userData="asr_only")
        self.mode_combo.addItem("外语 + 翻译", userData="translate")
        self.mode_combo.setCurrentIndex(1 if self._cfg.mode == "translate" else 0)

        self.target_lang_edit = QLineEdit(self._cfg.translator.target_lang, w)
        self.target_lang_edit.setPlaceholderText("zh / ja / es / ...")
        self.target_lang_edit.setMaximumWidth(160)

        self.context_window_spin = QSpinBox(w)
        self.context_window_spin.setRange(0, 10)
        self.context_window_spin.setValue(self._cfg.translator.context_window)

        form.addRow("模式:", self.mode_combo)
        form.addRow("译文目标语言:", self.target_lang_edit)
        form.addRow("翻译上下文窗口 (句):", self.context_window_spin)

        return w

    def _build_translator_tab(self) -> QWidget:
        endpoint = self._cfg.translator_endpoint()
        self._translator_panel = _ProviderPanel(
            title="翻译 / Chat 模型",
            available=chat_presets(),
            current_profile=self._cfg.translator.provider,
            current_model=self._cfg.translator.model,
            current_endpoint=endpoint,
            parent=self,
        )
        return self._translator_panel

    def _build_asr_tab(self) -> QWidget:
        """ASR settings.

        Two backends:

        * 本地 Vosk -- offline, free, decent on small models. The
          language picker + model dropdown + download button at the
          top configure Vosk. When this is the canonical backend,
          Vosk also drives the live preview row.
        * OpenAI Realtime -- networked, billed per minute, higher
          quality. When selected, Vosk still runs as a *preview-only*
          companion so the user keeps the words-as-they-are-spoken
          UX without the cloud round-trip latency. The Vosk settings
          below stay relevant in both modes (they configure the
          preview engine in the cloud case).
        """
        w = QWidget(self)
        outer = QVBoxLayout(w)

        # --- Canonical backend selector ----------------------------
        backend_form = QFormLayout()
        backend_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.asr_backend_combo = QComboBox(w)
        self.asr_backend_combo.addItem(
            "本地 Vosk (离线, 免费, 默认)", userData="vosk_local"
        )
        self.asr_backend_combo.addItem(
            "OpenAI Realtime API (云端, 高精度, 计费)",
            userData="openai_realtime",
        )
        # Pre-select based on saved config.
        idx = self.asr_backend_combo.findData(self._cfg.asr.provider)
        self.asr_backend_combo.setCurrentIndex(max(0, idx))
        self.asr_backend_combo.currentIndexChanged.connect(
            self._on_asr_backend_changed
        )
        backend_form.addRow("识别后端:", self.asr_backend_combo)

        # OpenAI Realtime model (only meaningful when that backend is
        # selected; widget visibility is toggled in
        # ``_on_asr_backend_changed`` below).
        self.openai_asr_model_combo = QComboBox(w)
        self.openai_asr_model_combo.setEditable(True)
        for m in ("gpt-4o-mini-transcribe", "gpt-4o-transcribe"):
            self.openai_asr_model_combo.addItem(m)
        if (
            self._cfg.asr.provider == "openai_realtime"
            and self._cfg.asr.model
            and not self._cfg.asr.model.startswith("vosk-")
        ):
            self.openai_asr_model_combo.setCurrentText(self._cfg.asr.model)
        self.openai_asr_model_label = QLabel("OpenAI 转写模型:")
        backend_form.addRow(
            self.openai_asr_model_label, self.openai_asr_model_combo
        )

        self.openai_asr_hint = QLabel(
            "<small><i>提示：复用「翻译模型」标签页中已保存的 "
            "<code>OPENAI_API_KEY</code>；如果没填过，请先到那里填一次。</i></small>"
        )
        self.openai_asr_hint.setStyleSheet("color: #888;")
        self.openai_asr_hint.setWordWrap(True)
        backend_form.addRow("", self.openai_asr_hint)

        outer.addLayout(backend_form)

        # --- Vosk section (still relevant in both modes) -----------
        vosk_header = QLabel(
            "<b>本地 Vosk 模型设置</b><br>"
            "<i><span style='color:#888'>"
            "Vosk 在「本地 Vosk」模式下负责完整识别 + 句末切句；"
            "在「OpenAI Realtime」模式下作为预览引擎，只显示「正在听到的词」"
            "（云端最终给出权威转写）。"
            "</span></i>"
        )
        vosk_header.setWordWrap(True)
        outer.addWidget(vosk_header)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # --- Language selector --------------------------------------
        self.asr_lang_combo = QComboBox(w)
        for code, label in vosk_model.available_languages():
            self.asr_lang_combo.addItem(f"{label}  ({code})", userData=code)
        # Pre-select based on the currently configured model's language;
        # fall back to the ``asr.language`` config field; else English.
        initial_lang = (
            vosk_model.language_of(self._cfg.local_asr.model)
            or self._cfg.asr.language
            or "en"
        )
        lang_idx = max(0, self.asr_lang_combo.findData(initial_lang))
        self.asr_lang_combo.setCurrentIndex(lang_idx)
        self.asr_lang_combo.currentIndexChanged.connect(self._on_asr_language_changed)
        form.addRow("识别语言:", self.asr_lang_combo)

        # --- Model selector (filtered by language) ------------------
        self.asr_model_combo = QComboBox(w)
        self.asr_model_combo.currentIndexChanged.connect(self._refresh_asr_model_status)
        form.addRow("Vosk 模型:", self.asr_model_combo)

        # --- Status + download row ---------------------------------
        self.asr_model_status_label = QLabel("")
        self.asr_model_status_label.setStyleSheet("color: #666;")
        self.asr_model_status_label.setWordWrap(True)
        form.addRow("", self.asr_model_status_label)

        btn_row = QHBoxLayout()
        self.asr_download_btn = QPushButton("下载模型")
        self.asr_download_btn.clicked.connect(self._on_download_vosk_model)
        btn_row.addWidget(self.asr_download_btn, 1)

        self.asr_open_dir_btn = QPushButton("打开模型目录")
        self.asr_open_dir_btn.setToolTip(
            "用资源管理器打开本地 Vosk 模型目录。"
            "如果官方下载受限，可以从 alphacephei.com 手动下载 zip 并解压到此目录。"
        )
        self.asr_open_dir_btn.clicked.connect(self._open_vosk_models_dir)
        btn_row.addWidget(self.asr_open_dir_btn, 0)
        form.addRow("", btn_row)

        # --- Silence cutoff ----------------------------------------
        self.asr_finalize_spin = QDoubleSpinBox(w)
        self.asr_finalize_spin.setRange(0.2, 10.0)
        self.asr_finalize_spin.setSingleStep(0.1)
        self.asr_finalize_spin.setDecimals(1)
        self.asr_finalize_spin.setSuffix(" s")
        self.asr_finalize_spin.setValue(self._cfg.local_asr.finalize_after_silence_s)
        self.asr_finalize_spin.setToolTip(
            "讲话停顿多久才认为一句话讲完，并把累积文本送去翻译。"
            "（仅「本地 Vosk」模式下生效；云端模式由服务端 VAD 决定。）"
        )
        form.addRow("句末静默时长:", self.asr_finalize_spin)

        # --- Force-finalize duration cap ---------------------------
        # When the speaker just keeps talking with no clear pause,
        # the live-preview row would otherwise accumulate forever and
        # eventually grow taller than the overlay window (which
        # breaks the scroll layout). This cap forces a TranscriptFinal
        # / OpenAI commit when the elapsed time of the current
        # utterance exceeds it, regardless of silence. The translator
        # picks up the chunk WITH the previous turn as context, so
        # mid-sentence cuts still produce a coherent translation.
        self.asr_max_duration_spin = QDoubleSpinBox(w)
        self.asr_max_duration_spin.setRange(2.0, 60.0)
        self.asr_max_duration_spin.setSingleStep(0.5)
        self.asr_max_duration_spin.setDecimals(1)
        self.asr_max_duration_spin.setSuffix(" s")
        self.asr_max_duration_spin.setValue(
            float(getattr(self._cfg.asr, "preview_max_duration_s", 8.0))
        )
        self.asr_max_duration_spin.setToolTip(
            "实时字幕单条最长时长。超过此时长后，无论是否检测到句尾，"
            "都会强制把当前文本送去翻译，避免悬浮窗里的实时字幕越积越长、"
            "把整窗排版顶飞。同时翻译时会带上上一句作为上下文，"
            "保证半句话也能翻译通顺。"
        )
        form.addRow("实时字幕最长:", self.asr_max_duration_spin)

        outer.addLayout(form)

        footer = QLabel(
            "<small>Vosk 模型从 alphacephei.com 拉取，约 30-80MB（small 档），"
            "可离线使用。预览模式下推荐使用 small 模型以保证响应速度。</small>"
        )
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #888;")
        outer.addWidget(footer)
        outer.addStretch(1)

        # Initial model list + status refresh + backend visibility.
        self._populate_asr_model_combo(preferred=self._cfg.local_asr.model)
        self._on_asr_backend_changed()
        return w

    def _on_asr_backend_changed(self) -> None:
        """Toggle visibility of the OpenAI-Realtime-specific widgets
        based on the selected canonical backend.

        The Vosk widgets stay visible in both modes -- they configure
        the preview engine when the canonical is cloud."""
        is_cloud = self.asr_backend_combo.currentData() == "openai_realtime"
        self.openai_asr_model_label.setVisible(is_cloud)
        self.openai_asr_model_combo.setVisible(is_cloud)
        self.openai_asr_hint.setVisible(is_cloud)

    # ------------------------------------------------------------------ ASR helpers

    def _on_asr_language_changed(self) -> None:
        # Language changed -> repopulate the model dropdown from the
        # filtered catalog. Try to keep the currently-selected model
        # if it happens to belong to the new language; otherwise the
        # first entry in the new list wins.
        current = self.asr_model_combo.currentData()
        new_lang = self.asr_lang_combo.currentData()
        if not new_lang:
            return
        models = vosk_model.models_for_language(new_lang)
        preferred = current if current in models else None
        self._populate_asr_model_combo(preferred=preferred)

    def _populate_asr_model_combo(self, preferred: Optional[str] = None) -> None:
        lang = self.asr_lang_combo.currentData()
        self.asr_model_combo.blockSignals(True)
        self.asr_model_combo.clear()
        if lang:
            for mid, meta in vosk_model.models_for_language(lang).items():
                self.asr_model_combo.addItem(
                    f"{meta['quality']}  ·  ~{meta['size_mb']}MB  ·  {mid}",
                    userData=mid,
                )
        if preferred:
            idx = self.asr_model_combo.findData(preferred)
            if idx >= 0:
                self.asr_model_combo.setCurrentIndex(idx)
        self.asr_model_combo.blockSignals(False)
        self._refresh_asr_model_status()

    def _refresh_asr_model_status(self) -> None:
        mid = self.asr_model_combo.currentData()
        if not mid:
            self.asr_model_status_label.setText("当前语言暂无可下载的模型。")
            self.asr_download_btn.setEnabled(False)
            return
        self.asr_download_btn.setEnabled(True)
        present = vosk_model.is_model_present(mid)
        if present:
            self.asr_model_status_label.setText(
                f"✓ 已下载到 {vosk_model.model_path(mid)}"
            )
            self.asr_download_btn.setText("重新下载")
        else:
            meta = vosk_model.list_known_models().get(mid, {})
            self.asr_model_status_label.setText(
                f"✗ 尚未下载（约 {meta.get('size_mb', '?')}MB）"
            )
            self.asr_download_btn.setText("下载模型")

    def _on_download_vosk_model(self) -> None:
        mid = self.asr_model_combo.currentData()
        if not mid:
            return
        meta = vosk_model.list_known_models().get(mid, {})
        size_mb = meta.get("size_mb", "?")

        confirm = QMessageBox.question(
            self,
            "下载 Vosk 模型",
            f"将从 alphacephei.com 下载约 {size_mb}MB 的 Vosk 模型到\n\n"
            f"{vosk_model.model_path(mid)}\n\n继续吗?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        progress = QProgressDialog("正在下载 Vosk 模型...", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.show()

        worker = _VoskDownloadWorker(mid)
        worker.progress.connect(
            lambda d, t: progress.setValue(int(d * 100 / t) if t else 0)
        )
        worker.error.connect(
            lambda msg: (
                progress.close(),
                QMessageBox.critical(self, "下载失败", msg),
            )
        )
        worker.finished_ok.connect(
            lambda path: (
                progress.setValue(100),
                progress.close(),
                self._refresh_asr_model_status(),
                QMessageBox.information(self, "下载完成", f"已下载到\n{path}"),
            )
        )
        progress.canceled.connect(worker.requestInterruption)
        worker.start()
        # Keep a reference so the QThread isn't GC'd mid-download.
        self._active_download_worker = worker

    def _open_vosk_models_dir(self) -> None:
        path = vosk_model.models_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _build_subtitle_tab(self) -> QWidget:
        w = QWidget(self)
        form = QFormLayout(w)
        sw = self._cfg.subtitle_window

        self.subtitle_show_check = QCheckBox("启动后显示字幕浮窗", w)
        self.subtitle_show_check.setChecked(True)

        self.subtitle_top_check = QCheckBox("字幕窗口始终置顶", w)
        self.subtitle_top_check.setChecked(sw.always_on_top)

        self.subtitle_clickthrough_check = QCheckBox("鼠标穿透 (点击不影响窗口)", w)
        self.subtitle_clickthrough_check.setChecked(sw.click_through)

        self.subtitle_opacity_spin = QDoubleSpinBox(w)
        self.subtitle_opacity_spin.setRange(0.0, 1.0)
        self.subtitle_opacity_spin.setSingleStep(0.05)
        self.subtitle_opacity_spin.setDecimals(2)
        self.subtitle_opacity_spin.setValue(sw.background_opacity)
        self.subtitle_opacity_spin.setToolTip(
            "字幕浮窗背后那块圆角深色板的透明度；\n"
            "0 = 完全透明（看不到底板），1 = 不透明。\n"
            "和「文字透明度」是独立的。"
        )

        self.subtitle_text_opacity_spin = QDoubleSpinBox(w)
        self.subtitle_text_opacity_spin.setRange(0.0, 1.0)
        self.subtitle_text_opacity_spin.setSingleStep(0.05)
        self.subtitle_text_opacity_spin.setDecimals(2)
        self.subtitle_text_opacity_spin.setValue(sw.text_opacity)
        self.subtitle_text_opacity_spin.setToolTip(
            "原文 / 译文 / 实时预览的文字透明度；\n"
            "0 = 完全透明（文字不可见），1 = 不透明。\n"
            "和「背景透明度」是独立的。"
        )

        self.subtitle_trans_font_spin = QSpinBox(w)
        self.subtitle_trans_font_spin.setRange(8, 72)
        self.subtitle_trans_font_spin.setSuffix(" pt")
        self.subtitle_trans_font_spin.setValue(sw.translation_font_size_pt)

        self.subtitle_asr_font_spin = QSpinBox(w)
        self.subtitle_asr_font_spin.setRange(8, 72)
        self.subtitle_asr_font_spin.setSuffix(" pt")
        self.subtitle_asr_font_spin.setValue(sw.asr_font_size_pt)

        self.subtitle_max_rows_spin = QSpinBox(w)
        self.subtitle_max_rows_spin.setRange(1, 12)
        self.subtitle_max_rows_spin.setSuffix(" 行")
        self.subtitle_max_rows_spin.setValue(sw.max_visible_entries)

        self.subtitle_row_spacing_spin = QSpinBox(w)
        self.subtitle_row_spacing_spin.setRange(0, 40)
        self.subtitle_row_spacing_spin.setSuffix(" px")
        self.subtitle_row_spacing_spin.setValue(sw.row_spacing_px)
        self.subtitle_row_spacing_spin.setToolTip(
            "句子之间的固定像素间距；不随字号变化。"
        )

        self.subtitle_slide_ms_spin = QSpinBox(w)
        self.subtitle_slide_ms_spin.setRange(0, 2000)
        self.subtitle_slide_ms_spin.setSuffix(" ms")
        self.subtitle_slide_ms_spin.setSingleStep(20)
        self.subtitle_slide_ms_spin.setValue(sw.slide_animation_ms)
        self.subtitle_slide_ms_spin.setToolTip(
            "新句子滑入动画时长。设为 0 关闭动画，立即出现。"
        )

        # Colour pickers. Three semantic slots: finalised EN, finalised
        # CN, and the live "currently being recognised" preview line.
        self.subtitle_asr_color_btn = _ColorButton(sw.asr_color, w)
        self.subtitle_translation_color_btn = _ColorButton(sw.translation_color, w)
        self.subtitle_preview_color_btn = _ColorButton(sw.preview_color, w)

        reset_btn = QPushButton("恢复默认配色", w)
        reset_btn.clicked.connect(self._reset_subtitle_colors)

        form.addRow(self.subtitle_show_check)
        form.addRow(self.subtitle_top_check)
        form.addRow(self.subtitle_clickthrough_check)
        form.addRow("背景透明度 (0=透明,1=不透明):", self.subtitle_opacity_spin)
        form.addRow("文字透明度 (0=透明,1=不透明):", self.subtitle_text_opacity_spin)
        form.addRow("译文字号:", self.subtitle_trans_font_spin)
        form.addRow("原文字号:", self.subtitle_asr_font_spin)
        form.addRow("最多保留行数:", self.subtitle_max_rows_spin)
        form.addRow("行间距 (固定像素):", self.subtitle_row_spacing_spin)
        form.addRow("滑入动画时长:", self.subtitle_slide_ms_spin)
        form.addRow("原文字幕颜色:", self.subtitle_asr_color_btn)
        form.addRow("译文颜色:", self.subtitle_translation_color_btn)
        form.addRow("实时预览行颜色:", self.subtitle_preview_color_btn)
        form.addRow("", reset_btn)

        return w

    def _reset_subtitle_colors(self) -> None:
        """Re-apply factory-default colours from the pydantic schema.

        Useful when the user has fiddled their way into an unreadable
        combination and wants a clean slate without exiting the dialog.
        """
        from ..config import SubtitleWindowConfig  # local import: avoid cycles

        defaults = SubtitleWindowConfig()
        self.subtitle_asr_color_btn.set_color(defaults.asr_color)
        self.subtitle_translation_color_btn.set_color(defaults.translation_color)
        self.subtitle_preview_color_btn.set_color(defaults.preview_color)

    # ------------------------------------------------------------------ Accept

    def _on_accept(self) -> None:
        cfg = self._cfg.model_copy(deep=True)

        # --- General ---
        cfg.mode = self.mode_combo.currentData()
        cfg.translator = TranslatorConfig(
            provider=cfg.translator.provider,  # overwritten below
            model=cfg.translator.model,
            target_lang=self.target_lang_edit.text().strip() or "zh",
            context_window=self.context_window_spin.value(),
            timeout_s=cfg.translator.timeout_s,
        )

        # --- Translator provider ---
        tp = self._translator_panel
        t_preset = tp.current_preset()
        t_endpoint = ProviderEndpoint(
            base_url=tp.current_base_url() or t_preset.base_url,
            api_key_env=t_preset.api_key_env,
            auth_required=t_preset.auth_required,
        )
        cfg.providers[t_preset.id] = t_endpoint
        cfg.translator = cfg.translator.model_copy(
            update={"provider": t_preset.id, "model": tp.current_model() or cfg.translator.model}
        )
        if tp.current_api_key().strip():
            get_secret_store().set(t_preset.api_key_env, tp.current_api_key())

        # --- ASR (Vosk + optional cloud canonical) ---
        picked_model = (
            self.asr_model_combo.currentData() or cfg.local_asr.model
        )
        picked_lang = self.asr_lang_combo.currentData() or cfg.asr.language or "en"
        cfg.local_asr = LocalASRConfig(
            model=picked_model,
            min_partial_chars=cfg.local_asr.min_partial_chars,
            finalize_after_silence_s=self.asr_finalize_spin.value(),
        )
        backend = self.asr_backend_combo.currentData() or "vosk_local"
        max_duration = self.asr_max_duration_spin.value()
        if backend == "openai_realtime":
            openai_model = (
                self.openai_asr_model_combo.currentText().strip()
                or "gpt-4o-mini-transcribe"
            )
            cfg.asr = ASRConfig(
                provider="openai_realtime",
                model=openai_model,
                language=picked_lang,
                preview_max_duration_s=max_duration,
            )
        else:
            cfg.asr = ASRConfig(
                provider="vosk_local",
                model=picked_model,
                language=picked_lang,
                preview_max_duration_s=max_duration,
            )

        # Friendly guards.
        if not vosk_model.is_model_present(cfg.local_asr.model):
            QMessageBox.warning(
                self,
                "Vosk 模型未下载",
                f"所选 Vosk 模型 {cfg.local_asr.model} 尚未下载。\n"
                + (
                    "云端模式下 Vosk 用于实时预览行；不下载也能跑，但开始时不会有「正在听到的词」。"
                    if backend == "openai_realtime"
                    else "请回到「语音识别」标签页点击「下载模型」后再启动，否则下次开始时会报错。"
                ),
            )
        if backend == "openai_realtime":
            openai_ep = cfg.providers.get("openai") or ProviderEndpoint()
            if not get_secret_store().has(openai_ep.api_key_env):
                QMessageBox.warning(
                    self,
                    "缺少 OpenAI API Key",
                    "选了云端 OpenAI Realtime 但还没保存 OPENAI_API_KEY。\n"
                    "请到「翻译模型」标签页选 OpenAI 服务商并填一次 Key，"
                    "之后云端识别就能用了。",
                )

        # --- Subtitle ---
        cfg.subtitle_window = cfg.subtitle_window.model_copy(
            update={
                "always_on_top": self.subtitle_top_check.isChecked(),
                "click_through": self.subtitle_clickthrough_check.isChecked(),
                "background_opacity": self.subtitle_opacity_spin.value(),
                "text_opacity": self.subtitle_text_opacity_spin.value(),
                "translation_font_size_pt": self.subtitle_trans_font_spin.value(),
                "asr_font_size_pt": self.subtitle_asr_font_spin.value(),
                "max_visible_entries": self.subtitle_max_rows_spin.value(),
                "row_spacing_px": self.subtitle_row_spacing_spin.value(),
                "slide_animation_ms": self.subtitle_slide_ms_spin.value(),
                "asr_color": self.subtitle_asr_color_btn.current_color(),
                "translation_color": self.subtitle_translation_color_btn.current_color(),
                "preview_color": self.subtitle_preview_color_btn.current_color(),
            }
        )

        self._cfg = cfg
        self.config_saved.emit(cfg)
        self.accept()

    def result_config(self) -> AppConfig:
        return self._cfg

    def subtitle_should_show(self) -> bool:
        return self.subtitle_show_check.isChecked()
