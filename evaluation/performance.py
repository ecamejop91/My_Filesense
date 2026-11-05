"""Qt controller: bounded session history and isolated background evaluation jobs."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QSettings, QTimer, pyqtSignal

from .tracking import SearchHistory
from app_storage import PATHS, private_directory

EVENT_PREFIX = 'FILESENSE_EVAL_EVENT:'


def unique_folder(parent, prefix):
    return Path(parent) / f'{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}'


def suggested_corpus_root(database, source_base):
    """Suggest a root from indexed paths only; never scan a user's home folder."""
    source_base = Path(source_base).resolve()
    try:
        with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as connection:
            rows = connection.execute('SELECT pdf_path FROM documents').fetchall()
        paths = [(Path(row[0]) if Path(row[0]).is_absolute() else source_base / row[0]).resolve().parent
                 for row in rows]
        return Path(os.path.commonpath(paths)) if paths else source_base
    except (OSError, sqlite3.Error, ValueError):
        return source_base


class PerformanceController(QObject):
    history_changed = pyqtSignal()
    job_changed = pyqtSignal()

    def __init__(self, parent=None, *, source_base=None, database=None, state_dir=None, run_dir=None):
        super().__init__(parent)
        self.source_base = Path(source_base or Path(__file__).resolve().parents[1]).resolve()
        self.database = Path(database or PATHS.library / 'file_tags.sqlite3')
        state_dir = private_directory(state_dir or PATHS.settings)
        self.run_dir = private_directory(run_dir or PATHS.evaluations)
        self.settings = QSettings(str(state_dir / 'performance.ini'), QSettings.Format.IniFormat)
        self.history = SearchHistory(limit=100)
        self.history.set_enabled(self.settings.value('tracking_enabled', False, type=bool))
        self.process = None
        self.state = 'idle'
        self.message = 'Choose a sample test or your saved answer key.'
        self.result = None
        self.report_dir = None
        self._job_dir = None
        self._stdout = ''
        self._diagnostics = ''
        self._worker_error = ''
        self._completion_received = False

    @property
    def busy(self):
        return self.state in ('running', 'cancelling')

    def set_tracking(self, enabled):
        self.history.set_enabled(enabled)
        self.settings.setValue('tracking_enabled', bool(enabled))
        self.settings.sync()
        self.history_changed.emit()

    def record_search(self, query, kind, results, elapsed_ms):
        if self.history.record(query, kind, results, elapsed_ms) is not None:
            self.history_changed.emit()

    def clear_history(self):
        self.history.clear()
        self.history_changed.emit()

    def start(self, *, mode='demo', benchmark=None, corpus_root=None, split='test', depth=5, systems='both'):
        if self.busy:
            raise ValueError('An evaluation is already running.')
        if mode not in ('demo', 'benchmark'):
            raise ValueError('Choose a sample test or a saved benchmark.')
        if mode == 'benchmark':
            if not benchmark or not Path(benchmark).is_file():
                raise ValueError('Choose an existing JSON or CSV answer key.')
            if not corpus_root or not Path(corpus_root).is_dir():
                raise ValueError('Choose the folder used as the root of your answer key document paths.')
            if not self.database.is_file():
                raise ValueError('Index your PDFs in FileSense before evaluating your collection.')
        if type(depth) is not int or depth < 1 or depth > 100:
            raise ValueError('Choose a result count between 1 and 100.')
        if split not in ('dev', 'test') or systems not in ('both', 'filesense', 'tfidf'):
            raise ValueError('Invalid evaluation settings.')
        self._job_dir = unique_folder(self.run_dir, 'app-' + mode)
        args = ['-m', 'evaluation.gui_worker', '--mode', mode, '--job-dir', str(self._job_dir),
                '--split', split, '--depth', str(depth), '--systems', systems]
        if mode == 'benchmark':
            args += ['--benchmark', str(Path(benchmark).resolve()), '--database', str(self.database.resolve()),
                     '--corpus-root', str(Path(corpus_root).resolve())]
        self.state = 'running'
        self.message = 'Starting evaluation in the background. You can continue using FileSense.'
        self.result = None
        self.report_dir = None
        self._stdout = ''
        self._diagnostics = ''
        self._worker_error = ''
        self._completion_received = False
        process = QProcess(self)
        self.process = process
        process.setWorkingDirectory(str(self.source_base))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUNBUFFERED', '1')
        environment.insert('HF_HUB_OFFLINE', '1')
        environment.insert('TRANSFORMERS_OFFLINE', '1')
        environment.insert('HF_HUB_DISABLE_TELEMETRY', '1')
        # The desktop bootstrap supplies account-private caches and temporary files.
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self.job_changed.emit()
        process.start(sys.executable, args)

    def _read_stdout(self):
        self._stdout += bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        while '\n' in self._stdout:
            line, self._stdout = self._stdout.split('\n', 1)
            if line.startswith(EVENT_PREFIX):
                try:
                    event = json.loads(line[len(EVENT_PREFIX):])
                    if not isinstance(event, dict):
                        continue
                except ValueError:
                    continue
                if event.get('event') == 'status' and self.state == 'running':
                    self.message = str(event.get('message', 'Working…'))
                    self.job_changed.emit()
                elif event.get('event') == 'error':
                    self._worker_error = str(event.get('message', 'Evaluation failed.'))
                elif event.get('event') == 'complete':
                    # Use our own known destination, never a path printed by another process.
                    self._completion_received = True
            else:
                self._diagnostics = (self._diagnostics + line + '\n')[-6000:]
        self._stdout = self._stdout[-12000:]

    def _read_stderr(self):
        self._diagnostics = (self._diagnostics + bytes(self.process.readAllStandardError()).decode(
            'utf-8', errors='replace'))[-6000:]

    def _finished(self, exit_code, exit_status):
        self._read_stdout()
        self._read_stderr()
        if self.state == 'cancelling':
            self.state = 'idle'
            self.message = 'Evaluation cancelled. No completed report was added.'
        elif exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit and self._completion_received:
            try:
                report = self._job_dir / 'report'
                result = json.loads((report / 'results.json').read_text(encoding='utf-8'))
                if not isinstance(result, dict) or not all(
                        key in result for key in ('metadata', 'aggregate', 'per_query', 'rankings')):
                    raise ValueError('The generated report is incomplete.')
                self.result = result
                self.report_dir = report
                self.state = 'complete'
                self.message = 'Evaluation complete. Review the results below or export the report.'
            except (OSError, ValueError) as exc:
                self.state = 'failed'
                self.message = f'Could not open the completed report: {exc}'
        else:
            self.state = 'failed'
            self.message = self._worker_error or 'Evaluation did not finish. Check the answer key and local model files, then retry.'
        process = self.process
        self.process = None
        if process is not None:
            process.deleteLater()
        self.job_changed.emit()

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.state = 'failed'
            self.message = 'Could not start the evaluation process. Check the FileSense Python environment.'
            if self.process is not None:
                self.process.deleteLater()
            self.process = None
            self.job_changed.emit()

    def cancel(self):
        if not self.busy or self.process is None:
            return
        self.state = 'cancelling'
        self.message = 'Cancelling evaluation…'
        process = self.process
        process.terminate()
        QTimer.singleShot(1500, lambda: self._kill_if_running(process))
        self.job_changed.emit()

    def _kill_if_running(self, process):
        if self.process is process and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def shutdown(self):
        process = self.process
        if process is not None:
            self.cancel()
            if not process.waitForFinished(1500):
                process.kill()
                process.waitForFinished(1500)
