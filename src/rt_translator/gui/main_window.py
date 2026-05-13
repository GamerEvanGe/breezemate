"""Main control window.

Compact panel with: device picker, mode toggle, start/stop button,
status indicator, settings button, subtitle-overlay toggle, and a small
recent-transcript history list for debugging.

Lifecycle ownership: this window creates and owns the
``PipelineController`` and the ``SubtitleWindow``, so closing it stops
everything.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStyle,
    QSystemTrayIcon,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..agents.context import SUPPORTED_EXTENSIONS as AGENT_CONTEXT_EXTS
from ..agents.registry import list_agent_modes
from ..config import (
    AgentConfig,
    AgentWindowConfig,
    AppConfig,
    AudioConfig,
    SubtitleWindowConfig,
    save_config,
)
from ..device_picker import (
    DeviceInfo,
    device_still_present,
    find_matching_device,
    list_devices,
    load_selection,
    save_selection,
)
from ..paths import asset_path
from .agent_window import AgentWindow
from .pipeline_controller import PipelineController
from .settings_dialog import SettingsDialog
from .subtitle_window import SubtitleWindow

log = logging.getLogger(__name__)


_STATUS_COLORS = {
    "idle": "#999",
    "connecting": "#f0ad4e",
    "reconnecting": "#f0ad4e",
    "connected": "#5cb85c",
    "disconnected": "#999",
    "error": "#d9534f",
}


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, config_path: Optional[Path] = None) -> None:
        super().__init__()
        self._cfg = cfg
        self._config_path = config_path
        self._devices: list[DeviceInfo] = []

        self.setWindowTitle("BreezeMate · 微伴 — 实时字幕 & 翻译")
        self.resize(560, 420)
        # Window icon (titlebar + taskbar). Falls back to the
        # QApplication-level icon if the bundled PNG is missing -- not
        # fatal, just slightly less branded.
        _icon = asset_path("breezemate.png") or asset_path("breezemate.ico")
        if _icon is not None:
            self.setWindowIcon(QIcon(str(_icon)))

        # --- Build UI ---
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        # Row 1: source + device dropdowns
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("音频源:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("系统播放 (Loopback)", userData="loopback")
        self.source_combo.addItem("麦克风 (Mic)", userData="mic")
        self.source_combo.currentIndexChanged.connect(self._refresh_devices)
        source_row.addWidget(self.source_combo, 0)
        source_row.addSpacing(8)
        source_row.addWidget(QLabel("设备:"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(220)
        source_row.addWidget(self.device_combo, 1)
        self.refresh_btn = QToolButton()
        self.refresh_btn.setText("⟳")
        self.refresh_btn.setToolTip("重新扫描音频设备")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        source_row.addWidget(self.refresh_btn, 0)
        outer.addLayout(source_row)

        # Row 2: mode + provider summary
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("仅外语字幕", userData="asr_only")
        self.mode_combo.addItem("外语 + 翻译", userData="translate")
        self.mode_combo.setCurrentIndex(1 if cfg.mode == "translate" else 0)
        mode_row.addWidget(self.mode_combo, 0)
        mode_row.addSpacing(12)
        self.provider_label = QLabel("")
        self.provider_label.setStyleSheet("color: #666;")
        mode_row.addWidget(self.provider_label, 1)
        outer.addLayout(mode_row)

        # Row 3: start/stop + settings + window toggles
        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  开始")
        self.start_btn.clicked.connect(self._on_start_clicked)
        self.start_btn.setMinimumHeight(36)
        font = self.start_btn.font()
        font.setPointSize(font.pointSize() + 1)
        font.setBold(True)
        self.start_btn.setFont(font)
        ctrl_row.addWidget(self.start_btn, 1)

        self.subtitle_btn = QPushButton("字幕浮窗")
        self.subtitle_btn.setCheckable(True)
        self.subtitle_btn.setChecked(True)
        self.subtitle_btn.toggled.connect(self._toggle_subtitle_window)
        ctrl_row.addWidget(self.subtitle_btn, 0)

        self.agent_btn = QPushButton("Agent 浮窗")
        self.agent_btn.setCheckable(True)
        self.agent_btn.setChecked(cfg.agent.enabled)
        self.agent_btn.toggled.connect(self._toggle_agent_window)
        ctrl_row.addWidget(self.agent_btn, 0)

        self.settings_btn = QPushButton("设置…")
        self.settings_btn.clicked.connect(self._open_settings)
        ctrl_row.addWidget(self.settings_btn, 0)
        outer.addLayout(ctrl_row)

        # Row 3b: agent controls (only meaningful when agent is enabled)
        agent_row = QHBoxLayout()
        self.agent_enable_check = QCheckBox("启用 Agent")
        self.agent_enable_check.setChecked(cfg.agent.enabled)
        self.agent_enable_check.toggled.connect(self._on_agent_enable_toggled)
        agent_row.addWidget(self.agent_enable_check, 0)

        agent_row.addWidget(QLabel("模式:"))
        self.agent_mode_combo = QComboBox()
        for mode_id, mode_label in list_agent_modes():
            self.agent_mode_combo.addItem(mode_label, userData=mode_id)
        self._select_agent_mode(cfg.agent.mode)
        self.agent_mode_combo.currentIndexChanged.connect(self._on_agent_mode_changed)
        agent_row.addWidget(self.agent_mode_combo, 1)

        self.agent_context_btn = QToolButton()
        self.agent_context_btn.setText("上传上下文…")
        self.agent_context_btn.setToolTip("添加 .txt / .md / .pdf / .docx 参考文件")
        self.agent_context_btn.clicked.connect(self._add_agent_context_files)
        agent_row.addWidget(self.agent_context_btn, 0)

        self.agent_context_clear_btn = QToolButton()
        self.agent_context_clear_btn.setText("清空")
        self.agent_context_clear_btn.setToolTip("移除所有已上传的上下文文件")
        self.agent_context_clear_btn.clicked.connect(self._clear_agent_context_files)
        agent_row.addWidget(self.agent_context_clear_btn, 0)
        outer.addLayout(agent_row)

        # Compact summary of currently-loaded context files. Plain
        # QLabel rather than a QListWidget so the main window stays
        # the same height whether 0 or 4 files are loaded.
        self.agent_context_label = QLabel("")
        self.agent_context_label.setStyleSheet("color: #888; font-size: 9pt;")
        self.agent_context_label.setWordWrap(True)
        outer.addWidget(self.agent_context_label)
        self._refresh_agent_context_label()

        # Row 4: status
        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {_STATUS_COLORS['idle']}; font-size: 16pt;")
        status_row.addWidget(self.status_dot, 0)
        self.status_text = QLabel("空闲")
        status_row.addWidget(self.status_text, 1)
        outer.addLayout(status_row)

        # Row 5: history (debug aid)
        outer.addWidget(QLabel("最近字幕:"))
        self.history_list = QListWidget()
        self.history_list.setUniformItemSizes(False)
        self.history_list.setWordWrap(True)
        outer.addWidget(self.history_list, 1)

        self.setCentralWidget(central)

        # --- Subtitle window ---
        self.subtitle_window = SubtitleWindow(cfg.subtitle_window)
        self.subtitle_window.settings_changed.connect(self._on_subtitle_settings_changed)
        self.subtitle_window.set_mode_label(cfg.mode)
        self.subtitle_window.show()

        # --- Agent window (hidden until user enables the agent) ---
        self.agent_window = AgentWindow(cfg.agent_window)
        self.agent_window.settings_changed.connect(self._on_agent_window_settings_changed)
        self._refresh_agent_window_label()
        if cfg.agent.enabled:
            self.agent_window.show()

        # --- Pipeline controller ---
        self.controller = PipelineController(self)
        self.controller.transcript_delta.connect(self.subtitle_window.on_transcript_delta)
        self.controller.transcript_final.connect(self.subtitle_window.on_transcript_final)
        self.controller.translation_delta.connect(self.subtitle_window.on_translation_delta)
        self.controller.translation_final.connect(self.subtitle_window.on_translation_final)
        self.controller.preview_delta.connect(self.subtitle_window.on_preview_delta)
        self.controller.preview_reset.connect(self.subtitle_window.on_preview_reset)

        # Agent: pre-cache the source text for the row header, then
        # plumb the agent events into the AgentWindow.
        self.controller.transcript_final.connect(self.agent_window.on_transcript_final)
        self.controller.agent_delta.connect(self.agent_window.on_agent_delta)
        self.controller.agent_final.connect(self.agent_window.on_agent_final)
        self.controller.agent_skipped.connect(self.agent_window.on_agent_skipped)

        self.controller.transcript_final.connect(self._on_transcript_final_log)
        self.controller.translation_final.connect(self._on_translation_final_log)
        self.controller.connection_status.connect(self._on_status_changed)
        # Four-state lifecycle: starting / started / stopping / stopped.
        # The two transient states (starting / stopping) repaint the
        # button immediately on click so the user never sees a "dead"
        # button waiting on a multi-second pipeline boot or teardown.
        self.controller.starting.connect(self._on_starting)
        self.controller.started.connect(self._on_started)
        self.controller.stopping.connect(self._on_stopping)
        self.controller.stopped.connect(self._on_stopped)
        self.controller.error.connect(self._on_error)

        # --- Tray icon (so the user can hide MainWindow without killing pipeline) ---
        self.tray = QSystemTrayIcon(self)
        self.tray.setToolTip("BreezeMate · 微伴")
        _tray_icon_path = asset_path("breezemate.png") or asset_path("breezemate.ico")
        if _tray_icon_path is not None:
            self.tray.setIcon(QIcon(str(_tray_icon_path)))
        elif QIcon.hasThemeIcon("media-record"):
            self.tray.setIcon(QIcon.fromTheme("media-record"))
        else:
            self.tray.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )
        self._build_tray_menu()
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

        # --- Initial state ---
        self._refresh_devices()
        self._update_provider_label()
        self._update_history_columns()

    # ------------------------------------------------------------------ Helpers

    def _build_tray_menu(self) -> None:
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        act_show = QAction("显示控制面板", self)
        act_show.triggered.connect(self._show_main)
        menu.addAction(act_show)

        act_subtitle = QAction("切换字幕浮窗", self)
        act_subtitle.triggered.connect(self._toggle_subtitle_window_quick)
        menu.addAction(act_subtitle)

        menu.addSeparator()
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(QApplication.quit)
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)

    def _show_main(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._show_main()

    def _toggle_subtitle_window_quick(self) -> None:
        on = not self.subtitle_window.isVisible()
        self.subtitle_btn.setChecked(on)

    def _update_provider_label(self) -> None:
        t = self._cfg.translator
        a = self._cfg.asr
        if self._cfg.mode == "translate":
            self.provider_label.setText(
                f"<small>ASR <b>{a.model}</b> · 翻译 <b>{t.provider}</b> / <b>{t.model}</b></small>"
            )
        else:
            self.provider_label.setText(f"<small>ASR <b>{a.model}</b></small>")

    def _update_history_columns(self) -> None:
        # Use a slightly smaller font for the history list so more rows fit.
        f = QFont(self.history_list.font())
        f.setPointSize(max(8, f.pointSize() - 1))
        self.history_list.setFont(f)

    def _refresh_devices(self) -> None:
        source = self.source_combo.currentData()
        try:
            all_devices = list_devices()
        except Exception as e:
            log.exception("Failed to enumerate audio devices")
            QMessageBox.warning(self, "音频设备", f"无法枚举音频设备: {e}")
            all_devices = []
        self._devices = [d for d in all_devices if d.source == source]
        self.device_combo.clear()
        for d in self._devices:
            label = f"{d.name}{'  (默认)' if d.is_default else ''}"
            self.device_combo.addItem(label, userData=d)

        # Pre-select the saved choice / config / default.
        preferred: Optional[DeviceInfo] = None
        if self._cfg.audio.source == source and self._cfg.audio.device_name:
            preferred = find_matching_device(source, self._cfg.audio.device_name)
        if preferred is None:
            saved = load_selection()
            if saved and saved.source == source and device_still_present(saved):
                preferred = saved
        if preferred is None:
            preferred = next((d for d in self._devices if d.is_default), None)

        if preferred is not None:
            for i in range(self.device_combo.count()):
                d = self.device_combo.itemData(i)
                if d.id == preferred.id and d.source == preferred.source:
                    self.device_combo.setCurrentIndex(i)
                    break

    def _selected_device(self) -> Optional[DeviceInfo]:
        d = self.device_combo.currentData()
        return d if isinstance(d, DeviceInfo) else None

    # ------------------------------------------------------------------ Actions

    def _on_start_clicked(self) -> None:
        if self.controller.is_running:
            self.controller.stop()
            return

        device = self._selected_device()
        if device is None:
            QMessageBox.warning(self, "无设备", "请先选择音频源和设备。")
            return

        try:
            self._cfg = self._cfg.model_copy(
                update={
                    "mode": self.mode_combo.currentData(),
                    "audio": AudioConfig(
                        source=device.source,
                        device_name=device.name,
                        chunk_ms=self._cfg.audio.chunk_ms,
                    ),
                }
            )
            # Validate that the chosen translator profile has a key
            # (unless auth_required is False). The ASR backend is
            # offline Vosk and needs no credentials.
            if self._cfg.mode == "translate":
                _ = self._cfg.translator_endpoint().resolve_api_key()
        except RuntimeError as e:
            QMessageBox.warning(
                self,
                "缺少 API Key",
                str(e) + "\n\n点击 \"设置…\" 填入 API Key 后重试。",
            )
            return
        except Exception as e:
            log.exception("Start failed")
            QMessageBox.critical(self, "启动失败", str(e))
            return

        save_selection(device)
        self.subtitle_window.set_mode_label(self._cfg.mode)
        if not self.subtitle_window.isVisible():
            self.subtitle_window.show()
            self.subtitle_btn.setChecked(True)
        self.history_list.clear()
        self.controller.start(self._cfg, device)

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._cfg, self)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            self._cfg = dlg.result_config()
            try:
                save_config(self._cfg, self._config_path)
            except Exception:
                log.exception("Failed to persist config")
            self.subtitle_window.apply_config(self._cfg.subtitle_window)
            self.subtitle_window.set_mode_label(self._cfg.mode)
            self.agent_window.apply_config(self._cfg.agent_window)
            self._refresh_agent_window_label()
            self.agent_enable_check.setChecked(self._cfg.agent.enabled)
            self.agent_btn.setChecked(self._cfg.agent.enabled)
            self._select_agent_mode(self._cfg.agent.mode)
            self._refresh_agent_context_label()
            if self._cfg.agent.enabled and not self.agent_window.isVisible():
                self.agent_window.show()
            elif not self._cfg.agent.enabled and self.agent_window.isVisible():
                self.agent_window.hide()
            self.mode_combo.setCurrentIndex(1 if self._cfg.mode == "translate" else 0)
            self._update_provider_label()
            if self.controller.is_running:
                QMessageBox.information(
                    self,
                    "已保存",
                    "设置已保存。新设置将在下次点击 \"开始\" 时生效。",
                )

    def _toggle_subtitle_window(self, on: bool) -> None:
        if on:
            self.subtitle_window.show()
            self.subtitle_btn.setText("隐藏字幕浮窗")
        else:
            self.subtitle_window.hide()
            self.subtitle_btn.setText("字幕浮窗")

    def _on_subtitle_settings_changed(self, new_cfg: SubtitleWindowConfig) -> None:
        self._cfg = self._cfg.model_copy(update={"subtitle_window": new_cfg})
        try:
            save_config(self._cfg, self._config_path)
        except Exception:
            log.exception("Failed to persist subtitle settings")

    # ------------------------------------------------------------------ Agent UI

    def _toggle_agent_window(self, on: bool) -> None:
        if on:
            self.agent_window.show()
            self.agent_btn.setText("隐藏 Agent 浮窗")
            # The user explicitly asked for the window; make sure the
            # pipeline-side agent is enabled too. Otherwise the window
            # would just sit empty.
            if not self._cfg.agent.enabled:
                self.agent_enable_check.setChecked(True)
        else:
            self.agent_window.hide()
            self.agent_btn.setText("Agent 浮窗")

    def _on_agent_enable_toggled(self, on: bool) -> None:
        if on == self._cfg.agent.enabled:
            return
        self._cfg = self._cfg.model_copy(
            update={"agent": self._cfg.agent.model_copy(update={"enabled": on})}
        )
        if on:
            self.agent_window.show()
            self.agent_btn.setChecked(True)
            self.agent_btn.setText("隐藏 Agent 浮窗")
        self._persist_config("agent toggled")
        if self.controller.is_running:
            QMessageBox.information(
                self,
                "Agent",
                "Agent 启用状态已更新。新设置将在下次点击 \"开始\" 时生效。",
            )

    def _on_agent_mode_changed(self, _index: int) -> None:
        mode = self.agent_mode_combo.currentData()
        if not mode or mode == self._cfg.agent.mode:
            return
        self._cfg = self._cfg.model_copy(
            update={"agent": self._cfg.agent.model_copy(update={"mode": mode})}
        )
        self._refresh_agent_window_label()
        self._persist_config("agent mode changed")

    def _refresh_agent_window_label(self) -> None:
        # Use the registry's human label for the active mode so the
        # window title and entry headers match the dropdown.
        label = next(
            (lbl for mid, lbl in list_agent_modes() if mid == self._cfg.agent.mode),
            "Agent",
        )
        self.agent_window.set_agent_label(label)

    def _select_agent_mode(self, mode: str) -> None:
        for i in range(self.agent_mode_combo.count()):
            if self.agent_mode_combo.itemData(i) == mode:
                self.agent_mode_combo.setCurrentIndex(i)
                return

    def _add_agent_context_files(self) -> None:
        # Build the file dialog filter from the loader's supported
        # extensions so the two stay in sync automatically.
        exts = " ".join(f"*{e}" for e in AGENT_CONTEXT_EXTS)
        filter_str = f"参考文档 ({exts});;所有文件 (*.*)"
        # Force the non-native Qt dialog. The Windows native picker
        # is a COM object that PyInstaller's frozen bootloader can
        # leave in a state where its first call hangs indefinitely
        # (clicking the button puts the whole app into "not
        # responding"). The pure-Qt dialog uses no COM and renders
        # consistently across the development and packaged builds.
        start_dir = str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择上下文文件",
            start_dir,
            filter_str,
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not paths:
            return
        existing = list(self._cfg.agent.context_files)
        seen = set(existing)
        for p in paths:
            if p not in seen:
                existing.append(p)
                seen.add(p)
        self._cfg = self._cfg.model_copy(
            update={
                "agent": self._cfg.agent.model_copy(update={"context_files": existing})
            }
        )
        self._refresh_agent_context_label()
        self._persist_config("agent context files added")

    def _clear_agent_context_files(self) -> None:
        if not self._cfg.agent.context_files:
            return
        self._cfg = self._cfg.model_copy(
            update={"agent": self._cfg.agent.model_copy(update={"context_files": []})}
        )
        self._refresh_agent_context_label()
        self._persist_config("agent context files cleared")

    def _refresh_agent_context_label(self) -> None:
        files = self._cfg.agent.context_files
        if not files:
            self.agent_context_label.setText("上下文: (未上传)")
            return
        names = [Path(p).name for p in files]
        if len(names) <= 3:
            joined = "、".join(names)
        else:
            joined = "、".join(names[:3]) + f" 等 {len(names)} 个文件"
        self.agent_context_label.setText(f"上下文: {joined}")

    def _on_agent_window_settings_changed(self, new_cfg: AgentWindowConfig) -> None:
        self._cfg = self._cfg.model_copy(update={"agent_window": new_cfg})
        self._persist_config("agent window cosmetics changed")

    def _persist_config(self, reason: str) -> None:
        try:
            save_config(self._cfg, self._config_path)
        except Exception:
            log.exception("Failed to persist config (%s)", reason)

    # ------------------------------------------------------------------ Slots

    def _on_starting(self) -> None:
        """Fired synchronously from ``controller.start()`` -- the worker
        thread may not have spawned yet, but the user clicked Start and
        we owe them visible feedback right now."""
        self.start_btn.setText("⏳  启动中…")
        self.start_btn.setEnabled(False)
        self.status_dot.setStyleSheet(
            f"color: {_STATUS_COLORS['connecting']}; font-size: 16pt;"
        )
        self.status_text.setText("正在启动…")

    def _on_started(self) -> None:
        """Fired from the worker thread once the asyncio loop is up
        and the main pipeline task has been scheduled. We may not be
        *connected* to the cloud yet -- ``connection_status`` events
        will refine the status text shortly."""
        self.start_btn.setText("■  停止")
        self.start_btn.setEnabled(True)
        self.status_dot.setStyleSheet(
            f"color: {_STATUS_COLORS['connecting']}; font-size: 16pt;"
        )
        self.status_text.setText("连接中…")
        self.tray.showMessage(
            "BreezeMate", "已开始转写", QSystemTrayIcon.MessageIcon.Information, 1500
        )

    def _on_stopping(self) -> None:
        """Fired synchronously from ``controller.stop()``. Pipeline
        teardown can take 2-3 s on Windows (WASAPI shutdown, OpenAI
        Realtime WS close); show the user we heard the click."""
        self.start_btn.setText("⏳  停止中…")
        self.start_btn.setEnabled(False)
        self.status_dot.setStyleSheet(
            f"color: {_STATUS_COLORS['connecting']}; font-size: 16pt;"
        )
        self.status_text.setText("正在停止…")

    def _on_stopped(self) -> None:
        self.start_btn.setText("▶  开始")
        self.start_btn.setEnabled(True)
        self.status_dot.setStyleSheet(f"color: {_STATUS_COLORS['idle']}; font-size: 16pt;")
        self.status_text.setText("空闲")

    def _on_error(self, msg: str) -> None:
        # Whatever state the button was in, make sure it's clickable
        # again so the user can retry. The ``stopped`` signal will fire
        # shortly after this too, but emitting both is harmless and
        # the user shouldn't see "启动中…" stuck after a failure.
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶  开始")
        self.status_dot.setStyleSheet(f"color: {_STATUS_COLORS['error']}; font-size: 16pt;")
        self.status_text.setText(f"错误: {msg}")
        QMessageBox.critical(self, "Pipeline 异常", msg)

    def _on_status_changed(self, state: str, detail: str) -> None:
        color = _STATUS_COLORS.get(state, _STATUS_COLORS["idle"])
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 16pt;")
        label = {
            "connecting": "连接中…",
            "reconnecting": "重新连接中…",
            "connected": "运行中 (已连接)",
            "disconnected": "已断开",
            "error": f"错误: {detail}" if detail else "错误",
        }.get(state, state)
        self.status_text.setText(label)

    def _on_transcript_final_log(self, item_id: str, text: str) -> None:
        self._append_history(f"[音] {text}")

    def _on_translation_final_log(self, item_id: str, text: str) -> None:
        self._append_history(f"[译] {text}")

    def _append_history(self, line: str) -> None:
        item = QListWidgetItem(line)
        self.history_list.addItem(item)
        # Cap to last 200 rows.
        while self.history_list.count() > 200:
            self.history_list.takeItem(0)
        self.history_list.scrollToBottom()

    # ------------------------------------------------------------------ Close

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        # Closing the main window hides it instead of quitting -- the
        # tray icon keeps the app alive. The user quits via the tray
        # menu's "退出".
        if self.tray.isVisible():
            self.hide()
            self.tray.showMessage(
                "BreezeMate",
                "已最小化到托盘。右键托盘图标可退出。",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
            event.ignore()
            return
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        try:
            self.controller.shutdown()
        except Exception:
            log.exception("Controller shutdown failed")
        try:
            self.subtitle_window.close()
        except Exception:
            pass
        try:
            self.agent_window.close()
        except Exception:
            pass
        try:
            self.tray.hide()
        except Exception:
            pass
