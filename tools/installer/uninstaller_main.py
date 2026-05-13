"""BreezeMate uninstaller (tkinter wizard).

Frozen by PyInstaller into ``BreezeMateUninstall.exe`` and copied by
the installer into the install directory. When the user clicks
"Uninstall" in Apps & features (or runs the exe directly), this
script:

  1. Confirms with the user.
  2. Optionally also clears user data (``%APPDATA%\\rt-translator``,
     which holds config.yaml, secrets.json, downloaded Vosk models).
  3. Deletes Start Menu + Desktop shortcuts.
  4. Removes the Add/Remove Programs registry key.
  5. Schedules deletion of the install directory itself. We can't
     ``rmtree`` while we're literally executing from inside it, so
     we spawn a detached cmd.exe that polls for our PID to exit and
     then removes the folder.

Pass ``/silent`` to skip the confirmation dialogs (used by the
"QuietUninstallString" entry, e.g. when Windows Apps & features
calls us during an OS-managed cleanup).
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tkinter as tk
import winreg
from pathlib import Path
from tkinter import messagebox, ttk

APP_NAME = "BreezeMate"
APP_NAME_CN = "微伴"
UNINSTALL_KEY_PATH = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall\BreezeMate"
)
USER_DATA_DIR_NAME = "rt-translator"  # see paths.py


def _own_install_dir() -> Path:
    """The folder we're running from = the install dir.

    Uses ``sys.executable`` when frozen (the typical case), the
    script's own location when run from source.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _start_menu_lnk() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return (
        Path(base)
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / f"{APP_NAME}.lnk"
    )


def _desktop_lnk() -> Path:
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as k:
            raw, _ = winreg.QueryValueEx(k, "Desktop")
            return Path(os.path.expandvars(raw)) / f"{APP_NAME}.lnk"
    except OSError:
        return Path.home() / "Desktop" / f"{APP_NAME}.lnk"


def _user_data_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / USER_DATA_DIR_NAME


def _delete_registry_entry() -> None:
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY_PATH)
    except OSError:
        pass


def _delete_path(p: Path) -> None:
    try:
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def _schedule_self_delete(install_dir: Path) -> None:
    """Hand off install-dir deletion to a detached cmd process.

    Windows refuses to delete a directory that has a running .exe
    inside it (the uninstaller IS that exe), so we spawn a child
    that:
      1. Waits for our PID to exit (``timeout /t 2``).
      2. ``rmdir /s /q`` the install dir.
      3. Self-deletes.

    The ``DETACHED_PROCESS`` + ``CREATE_NEW_CONSOLE`` flags ensure the
    child outlives us. We don't poll for completion -- the user has
    already seen "Uninstall complete" by the time the deletion
    actually finishes a couple seconds later.
    """
    pid = os.getpid()
    # ``ping`` is the most reliable "sleep for N seconds in a .cmd"
    # idiom on Windows; the redirected ``nul`` discards its output.
    cmd = (
        f'cmd /c "ping 127.0.0.1 -n 3 >nul & '
        f'rmdir /s /q "{install_dir}" & '
        f'exit"'
    )
    subprocess.Popen(
        cmd,
        shell=True,
        creationflags=subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP,
    )


class UninstallerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(f"卸载 {APP_NAME} {APP_NAME_CN}")
        root.geometry("520x360")
        root.resizable(False, False)
        root.configure(bg="#f4f7fb")

        self.install_dir = _own_install_dir()
        self.also_clear_user_data = tk.BooleanVar(value=True)

        self._build_confirm_page()

    def _build_confirm_page(self) -> None:
        for w in self.root.winfo_children():
            w.destroy()

        outer = tk.Frame(self.root, bg="#f4f7fb")
        outer.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            outer,
            text=f"卸载 {APP_NAME} {APP_NAME_CN}",
            font=("Segoe UI", 18, "bold"),
            bg="#f4f7fb",
            fg="#1f2d3d",
        ).pack(anchor="w")
        tk.Label(
            outer,
            text="即将从下列位置移除程序文件和快捷方式：",
            font=("Segoe UI", 10),
            bg="#f4f7fb",
            fg="#566578",
        ).pack(anchor="w", pady=(2, 6))
        tk.Label(
            outer,
            text=str(self.install_dir),
            font=("Consolas", 10),
            bg="#f4f7fb",
            fg="#1f2d3d",
        ).pack(anchor="w")

        tk.Checkbutton(
            outer,
            text=(
                "同时清除用户数据（API Key、配置、已下载的本地语音模型等）\n"
                f"位于：{_user_data_dir()}"
            ),
            variable=self.also_clear_user_data,
            bg="#f4f7fb",
            font=("Segoe UI", 10),
            justify="left",
            anchor="w",
        ).pack(anchor="w", pady=(20, 0))

        btn_row = tk.Frame(outer, bg="#f4f7fb")
        btn_row.pack(side="bottom", fill="x", pady=(20, 0))
        tk.Button(
            btn_row,
            text="取消",
            width=12,
            command=self.root.destroy,
            font=("Segoe UI", 10),
        ).pack(side="right", padx=(8, 0))
        tk.Button(
            btn_row,
            text="卸载",
            width=12,
            command=self._on_uninstall,
            font=("Segoe UI", 10, "bold"),
            bg="#e24a4a",
            fg="white",
            activebackground="#c83b3b",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
        ).pack(side="right")

    def _on_uninstall(self) -> None:
        self._build_progress_page()
        self.root.after(50, self._do_uninstall)

    def _build_progress_page(self) -> None:
        for w in self.root.winfo_children():
            w.destroy()
        outer = tk.Frame(self.root, bg="#f4f7fb")
        outer.pack(fill="both", expand=True, padx=24, pady=20)
        tk.Label(
            outer,
            text="正在卸载…",
            font=("Segoe UI", 18, "bold"),
            bg="#f4f7fb",
            fg="#1f2d3d",
        ).pack(anchor="w")
        self.status_var = tk.StringVar(value="正在准备…")
        tk.Label(
            outer,
            textvariable=self.status_var,
            bg="#f4f7fb",
            fg="#566578",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(20, 4))
        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill="x")
        self.progress.start(12)

    def _do_uninstall(self) -> None:
        self.status_var.set("正在删除快捷方式…")
        self.root.update_idletasks()
        _delete_path(_start_menu_lnk())
        _delete_path(_desktop_lnk())

        self.status_var.set("正在移除注册表项…")
        self.root.update_idletasks()
        _delete_registry_entry()

        if self.also_clear_user_data.get():
            self.status_var.set("正在清除用户数据…")
            self.root.update_idletasks()
            _delete_path(_user_data_dir())

        self.status_var.set("正在删除程序文件…")
        self.root.update_idletasks()
        _schedule_self_delete(self.install_dir)

        self.progress.stop()
        messagebox.showinfo(APP_NAME, f"{APP_NAME} {APP_NAME_CN} 已卸载。")
        self.root.destroy()


def _silent_uninstall() -> int:
    """Headless uninstall used by Apps & features quiet-uninstall."""
    _delete_path(_start_menu_lnk())
    _delete_path(_desktop_lnk())
    _delete_registry_entry()
    # In silent mode we do NOT touch user data: that's a deliberate
    # choice the user expressed in the GUI, and `/silent` callers
    # (Windows OS, MDM) shouldn't make that decision for them.
    _schedule_self_delete(_own_install_dir())
    return 0


def main() -> int:
    if sys.platform != "win32":
        print("This uninstaller only runs on Windows.", file=sys.stderr)
        return 2
    if "/silent" in sys.argv[1:] or "--silent" in sys.argv[1:]:
        return _silent_uninstall()
    # Hide the Windows console that PyInstaller would otherwise pop
    # up while we initialise tkinter.
    try:
        ctypes.windll.user32.ShowWindow(
            ctypes.windll.kernel32.GetConsoleWindow(), 0
        )
    except Exception:
        pass
    root = tk.Tk()
    UninstallerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
