"""BreezeMate Windows installer (tkinter wizard).

This script is what becomes ``BreezeMateSetup.exe`` after PyInstaller
packages it together with ``payload.zip`` (the pre-built one-folder
BreezeMate dist). When the user runs the setup .exe the PyInstaller
bootloader unpacks us to ``sys._MEIPASS``; this module then:

  1. Shows a single-page wizard (welcome blurb, install path picker,
     two shortcut checkboxes, Install / Cancel buttons).
  2. On Install: streams ``payload.zip`` into the chosen install
     directory with a progress bar.
  3. Writes a small marker file with the install path so the bundled
     uninstaller can clean up.
  4. Creates Start Menu (always) + Desktop (optional) shortcuts via a
     short PowerShell COM snippet -- this is the only way to make a
     real ``.lnk`` without pulling in pywin32 as a build dep.
  5. Registers an Add/Remove Programs entry under
     ``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall``
     pointing at the bundled uninstaller.
  6. Shows a finish page with "Launch BreezeMate" checkbox.

We deliberately install under ``%LOCALAPPDATA%\\Programs\\BreezeMate``
by default rather than ``Program Files``: that path is writable per-
user, so the installer does NOT need administrator elevation, which
also means no scary UAC prompt for end users.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import traceback
import winreg
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

APP_NAME = "BreezeMate"
APP_NAME_CN = "微伴"
APP_VERSION = "1.0.3"
APP_PUBLISHER = "BreezeMate"
APP_HELP_URL = "https://github.com/GamerEvanGe/breezemate"
UNINSTALL_KEY_PATH = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall\BreezeMate"
)
PAYLOAD_NAME = "payload.zip"
EXE_NAME = "BreezeMate.exe"
UNINSTALLER_NAME = "BreezeMateUninstall.exe"


def _bundled_path(name: str) -> Path:
    """Locate a file PyInstaller bundled next to the installer.

    When frozen, data files added via ``--add-data`` land under
    ``sys._MEIPASS``. In source checkouts (handy for debugging the
    installer UI) we fall back to ``<repo>/build/installer_payload/``.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / name
    here = Path(__file__).resolve().parent
    return here.parent.parent / "build" / "installer_payload" / name


def _default_install_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Programs" / APP_NAME


def _start_menu_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def _desktop_dir() -> Path:
    """Best-effort Desktop folder lookup.

    Falls back to ``%USERPROFILE%\\Desktop`` if the registry-based
    user-shell-folders query (which would correctly handle OneDrive-
    redirected desktops) fails for any reason.
    """
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as k:
            raw, _ = winreg.QueryValueEx(k, "Desktop")
            return Path(os.path.expandvars(raw))
    except OSError:
        return Path.home() / "Desktop"


def _make_shortcut(
    lnk_path: Path,
    target: Path,
    working_dir: Path,
    icon: Path,
    description: str,
) -> None:
    """Create a Windows ``.lnk`` via a tiny PowerShell COM call.

    Using PowerShell here keeps the installer free of pywin32 (which
    would add ~5 MB to the setup.exe and an extra import the installer
    doesn't otherwise need).
    """
    lnk_path.parent.mkdir(parents=True, exist_ok=True)
    ps = (
        "$ws = New-Object -ComObject WScript.Shell;"
        f"$s = $ws.CreateShortcut('{lnk_path}');"
        f"$s.TargetPath = '{target}';"
        f"$s.WorkingDirectory = '{working_dir}';"
        f"$s.IconLocation = '{icon}';"
        f"$s.Description = '{description}';"
        "$s.Save();"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
        capture_output=True,
    )


def _write_uninstall_registry(
    install_dir: Path,
    uninstaller_path: Path,
    icon_path: Path,
    estimated_size_kb: int,
) -> None:
    """Register BreezeMate under HKCU so it shows up in "Apps & features"."""
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY_PATH) as k:
        winreg.SetValueEx(k, "DisplayName", 0, winreg.REG_SZ, f"{APP_NAME} {APP_NAME_CN}")
        winreg.SetValueEx(k, "DisplayVersion", 0, winreg.REG_SZ, APP_VERSION)
        winreg.SetValueEx(k, "Publisher", 0, winreg.REG_SZ, APP_PUBLISHER)
        winreg.SetValueEx(k, "InstallLocation", 0, winreg.REG_SZ, str(install_dir))
        winreg.SetValueEx(k, "DisplayIcon", 0, winreg.REG_SZ, str(icon_path))
        winreg.SetValueEx(
            k, "UninstallString", 0, winreg.REG_SZ, f'"{uninstaller_path}"'
        )
        winreg.SetValueEx(
            k,
            "QuietUninstallString",
            0,
            winreg.REG_SZ,
            f'"{uninstaller_path}" /silent',
        )
        winreg.SetValueEx(k, "HelpLink", 0, winreg.REG_SZ, APP_HELP_URL)
        winreg.SetValueEx(k, "NoModify", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(k, "NoRepair", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(
            k, "EstimatedSize", 0, winreg.REG_DWORD, max(1, estimated_size_kb)
        )


def _extract_zip_with_progress(
    zip_path: Path,
    install_dir: Path,
    on_progress,
) -> None:
    """Stream-extract ``payload.zip`` while feeding ``on_progress`` a
    fraction in [0, 1] roughly every 1 MB written, so the wizard's
    progress bar moves smoothly without spamming the UI thread.
    """
    install_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        total_bytes = sum(m.file_size for m in members) or 1
        bytes_done = 0
        last_emit = 0
        for m in members:
            zf.extract(m, install_dir)
            bytes_done += m.file_size
            # Throttle UI updates to once every ~1 MB to avoid Tcl
            # event-queue saturation when extracting thousands of
            # tiny files (the vosk model dir has plenty of those).
            if bytes_done - last_emit > 1_048_576 or m is members[-1]:
                last_emit = bytes_done
                on_progress(bytes_done / total_bytes)


class InstallerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(f"{APP_NAME} {APP_NAME_CN} 安装向导")
        root.geometry("560x420")
        root.resizable(False, False)
        root.configure(bg="#f4f7fb")
        try:
            icon_path = _bundled_path("breezemate.ico")
            if icon_path.exists():
                root.iconbitmap(default=str(icon_path))
        except Exception:
            pass

        self.install_dir_var = tk.StringVar(value=str(_default_install_dir()))
        self.create_desktop_shortcut = tk.BooleanVar(value=True)
        self.create_start_menu_shortcut = tk.BooleanVar(value=True)
        self.launch_after_install = tk.BooleanVar(value=True)

        self._build_setup_page()

    # ------------------------------------------------------------------
    # Page 1: gather choices
    # ------------------------------------------------------------------
    def _build_setup_page(self) -> None:
        for w in self.root.winfo_children():
            w.destroy()

        outer = tk.Frame(self.root, bg="#f4f7fb")
        outer.pack(fill="both", expand=True, padx=24, pady=20)

        title = tk.Label(
            outer,
            text=f"欢迎安装 {APP_NAME} {APP_NAME_CN}",
            font=("Segoe UI", 18, "bold"),
            bg="#f4f7fb",
            fg="#1f2d3d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            outer,
            text=f"实时音频字幕与 Agent 助手 · v{APP_VERSION}",
            font=("Segoe UI", 10),
            bg="#f4f7fb",
            fg="#566578",
        )
        subtitle.pack(anchor="w", pady=(2, 18))

        # --- install location ---
        loc_frame = tk.LabelFrame(
            outer,
            text=" 安装位置 ",
            bg="#f4f7fb",
            fg="#1f2d3d",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
        )
        loc_frame.pack(fill="x")

        entry_row = tk.Frame(loc_frame, bg="#f4f7fb")
        entry_row.pack(fill="x")
        self.path_entry = tk.Entry(
            entry_row,
            textvariable=self.install_dir_var,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
        )
        self.path_entry.pack(side="left", fill="x", expand=True, ipady=4)
        tk.Button(
            entry_row,
            text="浏览…",
            command=self._on_browse,
            font=("Segoe UI", 9),
            width=8,
        ).pack(side="left", padx=(8, 0))

        tk.Label(
            loc_frame,
            text="默认安装到当前用户目录，无需管理员权限。",
            bg="#f4f7fb",
            fg="#788694",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(6, 0))

        # --- shortcuts ---
        sc_frame = tk.LabelFrame(
            outer,
            text=" 快捷方式 ",
            bg="#f4f7fb",
            fg="#1f2d3d",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
        )
        sc_frame.pack(fill="x", pady=(14, 0))

        tk.Checkbutton(
            sc_frame,
            text="在开始菜单创建快捷方式",
            variable=self.create_start_menu_shortcut,
            bg="#f4f7fb",
            font=("Segoe UI", 10),
            anchor="w",
        ).pack(fill="x", anchor="w")
        tk.Checkbutton(
            sc_frame,
            text="在桌面创建快捷方式",
            variable=self.create_desktop_shortcut,
            bg="#f4f7fb",
            font=("Segoe UI", 10),
            anchor="w",
        ).pack(fill="x", anchor="w")

        # --- buttons ---
        btn_row = tk.Frame(outer, bg="#f4f7fb")
        btn_row.pack(side="bottom", fill="x", pady=(20, 0))

        tk.Button(
            btn_row,
            text="取消",
            width=12,
            command=self.root.destroy,
            font=("Segoe UI", 10),
        ).pack(side="right", padx=(8, 0))

        self.install_btn = tk.Button(
            btn_row,
            text="安装",
            width=12,
            command=self._on_install,
            font=("Segoe UI", 10, "bold"),
            bg="#4a90e2",
            fg="white",
            activebackground="#3b7cc8",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
        )
        self.install_btn.pack(side="right")

    def _on_browse(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.install_dir_var.get(),
            title="选择安装位置",
            mustexist=False,
        )
        if path:
            # If the user picks an arbitrary folder, append our
            # subfolder name so we never extract straight into e.g.
            # "C:\\" or their Desktop.
            chosen = Path(path)
            if chosen.name != APP_NAME:
                chosen = chosen / APP_NAME
            self.install_dir_var.set(str(chosen))

    # ------------------------------------------------------------------
    # Page 2: install progress
    # ------------------------------------------------------------------
    def _build_progress_page(self) -> None:
        for w in self.root.winfo_children():
            w.destroy()

        outer = tk.Frame(self.root, bg="#f4f7fb")
        outer.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            outer,
            text="正在安装…",
            font=("Segoe UI", 18, "bold"),
            bg="#f4f7fb",
            fg="#1f2d3d",
        ).pack(anchor="w")
        tk.Label(
            outer,
            text="请稍候，安装过程通常只需几秒。",
            font=("Segoe UI", 10),
            bg="#f4f7fb",
            fg="#566578",
        ).pack(anchor="w", pady=(2, 24))

        self.progress = ttk.Progressbar(outer, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", pady=(6, 4))

        self.status_var = tk.StringVar(value="正在准备…")
        tk.Label(
            outer,
            textvariable=self.status_var,
            bg="#f4f7fb",
            fg="#566578",
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x")

    # ------------------------------------------------------------------
    # Page 3: done
    # ------------------------------------------------------------------
    def _build_done_page(self, install_dir: Path, exe_path: Path) -> None:
        for w in self.root.winfo_children():
            w.destroy()

        outer = tk.Frame(self.root, bg="#f4f7fb")
        outer.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(
            outer,
            text="安装完成",
            font=("Segoe UI", 18, "bold"),
            bg="#f4f7fb",
            fg="#1f2d3d",
        ).pack(anchor="w")
        tk.Label(
            outer,
            text=f"{APP_NAME} {APP_NAME_CN} 已安装到：",
            font=("Segoe UI", 10),
            bg="#f4f7fb",
            fg="#566578",
        ).pack(anchor="w", pady=(2, 4))
        tk.Label(
            outer,
            text=str(install_dir),
            font=("Consolas", 10),
            bg="#f4f7fb",
            fg="#1f2d3d",
        ).pack(anchor="w")

        tk.Checkbutton(
            outer,
            text=f"立即启动 {APP_NAME}",
            variable=self.launch_after_install,
            bg="#f4f7fb",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(24, 0))

        def on_finish():
            if self.launch_after_install.get():
                try:
                    # ``start`` + ``creationflags`` so the spawned GUI
                    # is fully detached from the installer process --
                    # otherwise closing the wizard window would also
                    # close BreezeMate.
                    subprocess.Popen(
                        [str(exe_path)],
                        cwd=str(install_dir),
                        creationflags=subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                except Exception as e:
                    messagebox.showwarning(
                        APP_NAME, f"未能启动 BreezeMate：{e}"
                    )
            self.root.destroy()

        tk.Button(
            outer,
            text="完成",
            width=12,
            command=on_finish,
            font=("Segoe UI", 10, "bold"),
            bg="#4a90e2",
            fg="white",
            activebackground="#3b7cc8",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
        ).pack(side="bottom", anchor="e", pady=(20, 0))

    # ------------------------------------------------------------------
    # Install action
    # ------------------------------------------------------------------
    def _on_install(self) -> None:
        install_dir = Path(self.install_dir_var.get()).expanduser()
        if not install_dir.is_absolute():
            messagebox.showerror(APP_NAME, "请输入完整的安装路径。")
            return

        # Refuse to install over system folders / drives root.
        bad_dirs = {
            Path("C:/"),
            Path(os.environ.get("WINDIR", "C:/Windows")),
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")),
        }
        if install_dir.resolve() in {p.resolve() for p in bad_dirs}:
            messagebox.showerror(APP_NAME, "请选择一个专属的安装文件夹。")
            return

        if install_dir.exists() and any(install_dir.iterdir()):
            ok = messagebox.askyesno(
                APP_NAME,
                f"目录已存在且非空：\n{install_dir}\n\n继续安装将覆盖里面的内容，是否继续？",
            )
            if not ok:
                return
            try:
                shutil.rmtree(install_dir)
            except Exception as e:
                messagebox.showerror(APP_NAME, f"清空目录失败：{e}")
                return

        self._build_progress_page()
        thread = threading.Thread(
            target=self._run_install_worker, args=(install_dir,), daemon=True
        )
        thread.start()

    def _run_install_worker(self, install_dir: Path) -> None:
        try:
            payload = _bundled_path(PAYLOAD_NAME)
            if not payload.exists():
                raise FileNotFoundError(
                    f"安装包损坏：缺少 {PAYLOAD_NAME}。请重新下载安装程序。"
                )

            # 1. Extract payload.
            self._set_status("正在解压程序文件…")

            def on_progress(frac: float) -> None:
                self.root.after(
                    0,
                    lambda f=frac: (
                        self.progress.configure(value=int(f * 900)),
                        self._set_status(f"正在解压程序文件… {int(f * 100)}%"),
                    ),
                )

            _extract_zip_with_progress(payload, install_dir, on_progress)

            # 2. Drop the uninstaller next to the main exe. The
            #    bundled BreezeMateUninstall.exe is itself a frozen
            #    Python script -- see uninstaller_main.py.
            self._set_status("正在写入卸载程序…")
            self.root.after(0, lambda: self.progress.configure(value=920))
            bundled_uninst = _bundled_path(UNINSTALLER_NAME)
            uninst_target = install_dir / UNINSTALLER_NAME
            if bundled_uninst.exists():
                shutil.copy2(bundled_uninst, uninst_target)

            # 3. Drop a small marker for the uninstaller's "where am I
            #    installed" lookup and for debug forensics.
            (install_dir / "install.marker").write_text(
                f"install_dir={install_dir}\nversion={APP_VERSION}\n",
                encoding="utf-8",
            )

            # 4. Shortcuts.
            exe_path = install_dir / EXE_NAME
            icon_path = install_dir / "_internal" / "assets" / "breezemate.ico"
            if not icon_path.exists():
                # Fallback: PyInstaller layouts vary by version. Use
                # the exe itself as the icon source -- Windows will
                # extract its embedded RT_ICON.
                icon_path = exe_path
            self._set_status("正在创建快捷方式…")
            self.root.after(0, lambda: self.progress.configure(value=950))

            if self.create_start_menu_shortcut.get():
                _make_shortcut(
                    _start_menu_dir() / f"{APP_NAME}.lnk",
                    exe_path,
                    install_dir,
                    icon_path,
                    f"{APP_NAME} {APP_NAME_CN}",
                )
            if self.create_desktop_shortcut.get():
                _make_shortcut(
                    _desktop_dir() / f"{APP_NAME}.lnk",
                    exe_path,
                    install_dir,
                    icon_path,
                    f"{APP_NAME} {APP_NAME_CN}",
                )

            # 5. Add/Remove Programs registration.
            self._set_status("正在注册卸载入口…")
            self.root.after(0, lambda: self.progress.configure(value=980))
            try:
                est_kb = (
                    sum(p.stat().st_size for p in install_dir.rglob("*") if p.is_file())
                    // 1024
                )
            except Exception:
                est_kb = 0
            try:
                _write_uninstall_registry(
                    install_dir,
                    uninst_target if bundled_uninst.exists() else exe_path,
                    icon_path,
                    est_kb,
                )
            except Exception as e:
                # Non-fatal: the app still works without an ARP entry.
                self._set_status(f"注册卸载入口失败（已忽略）：{e}")

            self.root.after(0, lambda: self.progress.configure(value=1000))
            self._set_status("安装完成。")
            self.root.after(
                250,
                lambda: self._build_done_page(install_dir, exe_path),
            )
        except Exception as e:
            tb = traceback.format_exc()
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    APP_NAME, f"安装失败：\n{e}\n\n详细信息：\n{tb}"
                ),
            )
            self.root.after(50, lambda: self._build_setup_page())

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))


def main() -> int:
    if sys.platform != "win32":
        print("This installer only runs on Windows.", file=sys.stderr)
        return 2
    root = tk.Tk()
    InstallerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
