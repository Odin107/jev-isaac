"""Read-only installation discovery and a narrowly scoped mod installer."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path


def find_game() -> Path | None:
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))/"Steam"]
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            roots.insert(0, Path(winreg.QueryValueEx(key, "SteamPath")[0]))
    except (ImportError, OSError):
        pass
    for root in list(roots):
        libraries = root/"steamapps"/"libraryfolders.vdf"
        if libraries.is_file():
            for path in re.findall(r'"path"\s+"([^"\n]+)"', libraries.read_text(encoding="utf-8", errors="replace")):
                roots.append(Path(path.replace("\\\\", "\\")))
    for root in roots:
        folder = root/"steamapps"/"common"/"The Binding of Isaac Rebirth"
        if (folder/"isaac-ng.exe").is_file():
            return folder
    return None


def install_mod(project: Path, game_dir: Path | None = None) -> Path:
    game = game_dir or find_game()
    if game is None or not (game/"isaac-ng.exe").is_file():
        raise ValueError("Install Isaac's Rebirth Complete Bundle in Steam first, then run this installer again.")
    game = game.resolve()
    source = project/"mod"/"jev_bridge"
    destination = game/"mods"/"jev_bridge"
    if destination.exists():
        expected = {p.name for p in source.iterdir() if p.is_file()}
        existing = {p.name for p in destination.iterdir()}
        if expected == existing and all((source/name).read_bytes() == (destination/name).read_bytes() for name in expected):
            return destination
        raise ValueError("A different jev_bridge mod already exists. Keep a backup and remove or rename it before updating.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination
