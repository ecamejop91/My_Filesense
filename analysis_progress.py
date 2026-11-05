"""Background analysis and a modeless, live progress window."""
from pathlib import Path
from time import perf_counter

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QHeaderView, QLabel, QProgressBar, QPushButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from document_database import DATABASE_NAME
from document_search_engine import update_vector_store
from document_tag_store import tag_files, prepare_tagging, load_tagging_vectors
from ocr import DEFAULT_OCR_MODEL, validate_ocr_model


class AnalysisCancelled(Exception):
    pass


class AnalysisWorker(QThread):
    progress = pyqtSignal(dict)
    completed = pyqtSignal(object, object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, file_paths, vector_store_dir, parent=None, *,
                 ocr_model=DEFAULT_OCR_MODEL):
        super().__init__(parent)
        self.file_paths = list(dict.fromkeys(Path(p).expanduser().resolve() for p in file_paths))
        self.vector_store_dir = Path(vector_store_dir)
        self.ocr_model = validate_ocr_model(ocr_model)

    def report(self, event):
        if self.isInterruptionRequested():
            raise AnalysisCancelled()
        self.progress.emit(event)

    def run(self):
        try:
            # Incremental: the worker never loads the whole index. The window
            # merges the returned delta into its loaded index, if it has one.
            store = update_vector_store(self.file_paths, self.vector_store_dir, progress=self.report,
                                        full_result=False, ocr_model=self.ocr_model)
            db_path = self.vector_store_dir / DATABASE_NAME
            plan = prepare_tagging(self.file_paths, db_path)
            # Categories come from the chunk embeddings just saved (or already
            # stored), read for only the files that need tags: no second model
            # pass over the document text.
            tags = tag_files(
                self.file_paths, db_path=db_path,
                vectors_by_path=load_tagging_vectors(plan, db_path),
                model=store['model'], progress=self.report,
                plan=plan,
            )
            # Once the operation has fully completed, present that outcome even
            # if a cancellation request arrived immediately after the last write.
            self.completed.emit(store, tags)
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


STAGES = {
    'loading': 'Preparing analysis', 'checking': 'Checking file',
    'extracting': 'Reading document text', 'ocr': 'Recognizing scanned pages (OCR)',
    'chunking': 'Creating text chunks', 'embedding': 'Embedding text chunks',
    'saving': 'Saving index', 'indexed': 'Index saved', 'tagging': 'Assigning categories',
    'done': 'Complete', 'skipped': 'Skipped',
    'reusing': 'Reusing identical content',
}


class AnalysisProgressDialog(QDialog):
    cancel_requested = pyqtSignal()

    def __init__(self, file_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle('FileSense — Analysis progress')
        self.setModal(False)
        self.resize(820, 480)
        self.setMinimumSize(680, 420)
        self.running = True
        self.stopping = False
        self.started = self.step_started = perf_counter()
        self._step = None
        self._active_item = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        self.stage_label = QLabel('Preparing analysis…')
        self.stage_label.setStyleSheet('font-size: 17px; font-weight: bold;')
        self.file_label = QLabel('Waiting for the worker to start')
        self.file_label.setWordWrap(True)
        self.detail_label = QLabel('')
        self.detail_label.setWordWrap(True)
        self.stage_bar = QProgressBar()
        self.stage_bar.setRange(0, 0)
        self.stage_bar.setFormat('%v / %m')
        self.elapsed_label = QLabel('Elapsed: 0s')
        self.files_label = QLabel('')
        for widget in (self.stage_label, self.file_label, self.detail_label,
                       self.stage_bar, self.elapsed_label, self.files_label):
            layout.addWidget(widget)
        self.files = QTreeWidget()
        self.files.setHeaderLabels(['File', 'Stage', 'Progress / details'])
        self.files.setRootIsDecorated(False)
        self.files.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.files.header().resizeSection(0, 220)
        self.files.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.files.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.items = {}
        for raw in file_paths:
            path = str(Path(raw).expanduser().resolve())
            if path not in self.items:
                item = QTreeWidgetItem([Path(path).name, 'Queued', 'Waiting'])
                item.setToolTip(0, path)
                self.files.addTopLevelItem(item)
                self.items[path] = item
        self.completed_paths = set()
        self._update_count()
        layout.addWidget(self.files, 1)
        actions = QHBoxLayout()
        actions.addStretch()
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self._button_clicked)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _tick(self):
        self.elapsed_label.setText(
            f'Elapsed: {perf_counter() - self.started:.0f}s · Current step: {perf_counter() - self.step_started:.0f}s'
        )

    def _update_count(self):
        self.files_label.setText(f'Files complete: {len(self.completed_paths)} / {len(self.items)}')

    def update_progress(self, event):
        stage = event.get('stage', '')
        path = event.get('path')
        path = str(Path(path).expanduser().resolve()) if path else None
        step = (stage, path)
        if step != self._step:
            self.step_started = perf_counter()
            self._step = step
        title = STAGES.get(stage, stage)
        self.stage_label.setText('Stopping after the current step…' if self.stopping else title)
        self.file_label.setText(Path(path).name if path else 'All queued files')
        self.file_label.setToolTip(path or '')
        detail = event.get('detail', '')
        self.detail_label.setText(detail)
        total = event.get('total', 0)
        current = event.get('current', 0)
        self.stage_bar.setRange(0, total if total > 0 else 0)
        self.stage_bar.setValue(current)
        if stage == 'embedding':
            self.stage_bar.setFormat('Chunks: %v / %m')
        else:
            self.stage_bar.setFormat(event.get('unit', 'Pages') + ': %v / %m')
        self._active_item = self.items.get(path)
        if self._active_item is not None:
            self._active_item.setText(1, title)
            self._active_item.setText(2, detail)
            self.files.scrollToItem(self._active_item)
            if stage == 'done' or (stage == 'skipped' and detail == 'Unsupported file type'):
                self.completed_paths.add(path)
        self._update_count()

    def _button_clicked(self):
        if self.running:
            self.stopping = True
            self.cancel_button.setEnabled(False)
            self.stage_label.setText('Stopping after the current step…')
            self.detail_label.setText('The current model call or database save must finish before stopping.')
            self.cancel_requested.emit()
        else:
            self.accept()

    def finish(self, outcome, message):
        self.running = False
        self.timer.stop()
        self._tick()
        self.stage_label.setText(outcome)
        self.detail_label.setText(message)
        self.stage_bar.setRange(0, 1)
        self.stage_bar.setValue(1 if outcome == 'Analysis complete' else 0)
        self.stage_bar.setFormat('Finished' if outcome == 'Analysis complete' else 'Stopped')
        if outcome != 'Analysis complete' and self._active_item is not None:
            self._active_item.setText(1, outcome)
        self.cancel_button.setText('Close')
        self.cancel_button.setEnabled(True)

    def reject(self):
        # Escape/close must not destroy a progress window while its worker runs.
        if not self.running:
            super().reject()

    def closeEvent(self, event):
        if self.running:
            event.ignore()
        else:
            super().closeEvent(event)
