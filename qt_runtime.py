"""Prepare desktop identity, standard streams and Qt plugins for launches."""
from __future__ import annotations

import filecmp
import os
from pathlib import Path
import shutil
import sys


def prepare_app_identity() -> None:
    """Give Windows a FileSense taskbar group instead of the Python default."""
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('FileSense.Desktop')


def prepare_standard_streams() -> None:
    """Give console-oriented libraries valid output streams under pythonw."""
    for name in ('stdout', 'stderr'):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, 'w', encoding='utf-8'))


def prepare_qt_plugins() -> None:
    if sys.platform != 'darwin':
        return

    from PyQt6.QtCore import QCoreApplication, QFileInfo, QLibraryInfo

    installed = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    cocoa = installed / 'platforms' / 'libqcocoa.dylib'
    if not cocoa.is_file() or not QFileInfo(str(cocoa)).isHidden():
        return

    # Qt excludes hidden plugins from discovery. Copy only file contents: copying
    # metadata would also copy the macOS hidden flag that caused the crash.
    from app_storage import PATHS, private_directory, prepare_managed_file
    cache = private_directory(PATHS.cache / 'qt-plugins')
    for source in installed.rglob('*.dylib'):
        target = cache / source.relative_to(installed)
        prepare_managed_file(target)
        if (not target.exists() or QFileInfo(str(target)).isHidden()
                or not filecmp.cmp(source, target, shallow=False)):
            temporary = target.with_suffix('.dylib.tmp')
            prepare_managed_file(temporary)
            shutil.copyfile(source, temporary)
            temporary.replace(target)

    QCoreApplication.addLibraryPath(str(cache))
