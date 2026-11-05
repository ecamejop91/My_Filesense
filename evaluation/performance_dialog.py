"""Optional in-app performance view; all expensive work belongs to the worker."""
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .performance import suggested_corpus_root, unique_folder
from .excel_report import export_history_workbook, export_evaluation_workbook


def _label(text):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


def _table(headers):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().hide()
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.setAlternatingRowColors(True)
    return table


def _fill(table, rows):
    table.setRowCount(len(rows))
    for index, row in enumerate(rows):
        for column, value in enumerate(row):
            text = '—' if value is None else str(value)
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            table.setItem(index, column, item)


def _number(value, percent=False):
    if value is None:
        return 'N/A'
    return f'{value:.1%}' if percent else f'{value:.3f}'


def _system(name):
    return {'filesense': 'FileSense', 'tfidf': 'TF-IDF'}.get(name, name)


class PerformanceDialog(QDialog):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._rendered_result = None
        self._records = []
        self.setWindowTitle('FileSense · Performance')
        self.setModal(False)
        self.resize(1120, 790)
        self.setMinimumSize(920, 680)
        self.setStyleSheet('''
            QDialog { background: #f3f0e8; }
            QTableWidget { background: white; alternate-background-color: #f5f7f3;
                color: #202825; gridline-color: #e1ddd2; selection-background-color: #d5eadc;
                selection-color: #173f3a; border: 1px solid #d9d4c8; }
            QHeaderView::section { background: #e7eee7; color: #173f3a; padding: 8px; border: 0; }
            QTabWidget::pane { border: none; background: #fcfbf6; }
            QTabBar::tab { padding: 10px 16px; }
            QComboBox, QSpinBox { background: white; color: #202825; padding: 5px; }
            QLineEdit { min-height: 30px; }
        ''')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        title = _label('Performance')
        title.setObjectName('dialogTitle')
        layout.addWidget(title)
        layout.addWidget(_label('See how searches behave, then measure quality against known answers.'))
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_history()
        self._build_evaluation()
        footer = QHBoxLayout()
        footer.addWidget(_label('Closing this panel keeps a running evaluation active. Quitting FileSense cancels it.'))
        close = QPushButton('Close')
        close.setObjectName('quietButton')
        close.clicked.connect(self.close)
        footer.addWidget(close)
        layout.addLayout(footer)
        controller.history_changed.connect(self.refresh_history)
        controller.job_changed.connect(self.refresh_job)
        self.refresh_history()
        self.refresh_job()

    def _build_history(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.tracking_toggle = QCheckBox('Track search performance')
        self.tracking_toggle.toggled.connect(self.controller.set_tracking)
        row.addWidget(self.tracking_toggle)
        row.addStretch()
        self.clear_button = QPushButton('Clear history')
        self.clear_button.clicked.connect(self.controller.clear_history)
        self.export_history_button = QPushButton('Export Excel…')
        self.export_history_button.clicked.connect(self.export_history)
        row.addWidget(self.clear_button)
        row.addWidget(self.export_history_button)
        layout.addLayout(row)
        layout.addWidget(_label(
            'Off by default. When enabled, keeps the last 100 searches in this session: query, displayed '
            'results, scores and response time. No extra search or model run. History is saved only when '
            'you export one Excel workbook; it includes query text and file paths.'))
        self.history_status = _label('')
        layout.addWidget(self.history_status)
        self.history_table = _table(['Time', 'Query', 'Search type', 'Matches', 'Response (ms)'])
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.history_table.itemSelectionChanged.connect(self.show_search)
        layout.addWidget(self.history_table, 1)
        layout.addWidget(_label('Quality not evaluated — similarity scores do not tell us whether a result is correct. '
                                'Use the Evaluation tab for precision, recall, MRR and nDCG.'))
        self.search_table = _table(['Rank', 'PDF', 'Page', 'Score'])
        self.search_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.search_table, 1)
        layout.addWidget(_label('Results match what the search displayed, including repeated PDFs and tag filters. '
                                'Semantic scores are cosine similarity; exact tag matches use the tag score. '
                                'Timing includes loading the index when needed and preparing the display.'))
        self.tabs.addTab(page, 'Recent searches')

    def _build_evaluation(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        top = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem('Sample test · ready to run', 'demo')
        self.mode_combo.addItem('My collection · saved answer key', 'benchmark')
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        top.addWidget(self.mode_combo, 1)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 100)
        self.depth_spin.setValue(5)
        top.addWidget(_label('Top results:'))
        top.addWidget(self.depth_spin)
        self.system_combo = QComboBox()
        self.system_combo.addItem('FileSense + TF-IDF', 'both')
        self.system_combo.addItem('FileSense only', 'filesense')
        self.system_combo.addItem('TF-IDF only', 'tfidf')
        top.addWidget(self.system_combo)
        layout.addLayout(top)
        self.custom_widget = QWidget()
        form = QFormLayout(self.custom_widget)
        form.setContentsMargins(0, 0, 0, 0)
        self.benchmark_input = QLineEdit()
        self.benchmark_input.setPlaceholderText('Version 1 JSON or CSV benchmark with relevance grades')
        form.addRow('Answer key:', self._path_picker(self.benchmark_input, self._choose_benchmark))
        self.root_input = QLineEdit()
        form.addRow('Document root:', self._path_picker(self.root_input, self._choose_root))
        self.split_combo = QComboBox()
        self.split_combo.addItem('Test · final evaluation', 'test')
        self.split_combo.addItem('Development · tuning', 'dev')
        form.addRow('Query split:', self.split_combo)
        layout.addWidget(self.custom_widget)
        self.mode_note = _label('')
        layout.addWidget(self.mode_note)
        actions = QHBoxLayout()
        self.run_button = QPushButton('Run sample evaluation')
        self.run_button.setObjectName('primaryButton')
        self.run_button.clicked.connect(self.start_evaluation)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.controller.cancel)
        self.open_button = QPushButton('Open report folder')
        self.open_button.clicked.connect(self.open_report)
        self.export_report_button = QPushButton('Export Excel…')
        self.export_report_button.clicked.connect(self.export_report)
        for button in (self.run_button, self.cancel_button, self.open_button, self.export_report_button):
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        self.job_status = _label('')
        layout.addWidget(self.job_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(12)
        layout.addWidget(self.progress)
        self.result_note = _label('No evaluation has run in this session.')
        layout.addWidget(self.result_note)
        self.report_tabs = QTabWidget()
        self.summary_table = _table(['System', 'Precision@5', 'Recall@5', 'MRR@5', 'nDCG@5', 'Mean query (ms)'])
        self.report_tabs.addTab(self.summary_table, 'Summary')
        self.query_table = _table(['System', 'Query ID', 'Precision@5', 'Recall@5', 'RR@5', 'nDCG@5', 'Query (ms)'])
        self.report_tabs.addTab(self.query_table, 'Each query')
        self.rankings_table = _table(['System', 'Query ID', 'Rank', 'PDF', 'Score', 'Relevance (0–3)'])
        self.rankings_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.report_tabs.addTab(self.rankings_table, 'Ranked PDFs')
        layout.addWidget(self.report_tabs, 1)
        layout.addWidget(_label(
            'Precision@k: useful PDFs / k slots. Recall@k: useful PDFs found / all useful PDFs. '
            'MRR: how early the first useful PDF appears. nDCG: how well the strongest matches appear first. '
            'Higher is better (0–1). N/A means undefined, not zero. Summary quality values are macro averages '
            '(each eligible query has equal weight); pooled values are available in the reports.'))
        self.tabs.addTab(page, 'Evaluation')
        self._mode_changed()

    def _path_picker(self, field, callback):
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(field, 1)
        browse = QPushButton('Browse…')
        browse.clicked.connect(callback)
        row.addWidget(browse)
        return widget

    def _choose_benchmark(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose a labeled benchmark', '', 'Answer keys (*.json *.csv)')
        if path:
            self.benchmark_input.setText(path)

    def _choose_root(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose benchmark document root', self.root_input.text())
        if path:
            self.root_input.setText(path)

    def _mode_changed(self):
        custom = self.mode_combo.currentData() == 'benchmark'
        self.custom_widget.setVisible(custom)
        if custom and not self.root_input.text():
            self.root_input.setText(str(suggested_corpus_root(self.controller.database, self.controller.source_base)))
        self.run_button.setText('Evaluate my collection' if custom else 'Run sample evaluation')
        self.mode_note.setText(
            'Uses a frozen copy of your current index and your saved labels. Both systems search the same PDFs. '
            'Every PDF must have a grade for each selected query; unjudged PDFs are not silently marked irrelevant. '
            'The document root must match the relative paths in the answer key. '
            'The local run folder retains an index snapshot (including document text) and the answer key; '
            'Export Excel saves the results in one workbook without the private index snapshot.' if custom else
            'A ready-made test with 3 invented PDFs and 4 labeled queries. No manual setup. '
            'This checks the evaluation workflow; it does not measure your own collection. '
            'Equal FileSense and TF-IDF quality metrics are expected on some simple queries. '
            'The sample needs the locally cached FileSense model, even for a TF-IDF-only test.')

    def refresh_history(self):
        selected = self.history_table.currentRow()
        selected_id = self._records[selected]['search_id'] if 0 <= selected < len(self._records) else None
        self._records = self.controller.history.records
        self.tracking_toggle.blockSignals(True)
        self.tracking_toggle.setChecked(self.controller.history.enabled)
        self.tracking_toggle.blockSignals(False)
        self.history_table.blockSignals(True)
        rows = []
        for record in self._records:
            timestamp = datetime.fromisoformat(record['timestamp']).astimezone().strftime('%H:%M:%S')
            rows.append([timestamp, record['query'], record['kind'], record['result_count'], f"{record['elapsed_ms']:.1f}"])
        _fill(self.history_table, rows)
        self.history_table.blockSignals(False)
        if self._records:
            row = next((i for i, record in enumerate(self._records) if record['search_id'] == selected_id), 0)
            self.history_table.selectRow(row)
        self.show_search()
        self.history_status.setText(f"{'Tracking on' if self.controller.history.enabled else 'Tracking off'} · "
                                   f'{len(self._records)} searches in this session')
        self.clear_button.setEnabled(bool(self._records))
        self.export_history_button.setEnabled(bool(self._records))

    def show_search(self):
        row = self.history_table.currentRow()
        results = self._records[row]['results'] if 0 <= row < len(self._records) else []
        _fill(self.search_table, [[hit['rank'], hit['path'], hit['page'], _number(hit['score'])] for hit in results])

    def start_evaluation(self):
        mode = self.mode_combo.currentData()
        try:
            self.controller.start(mode=mode, benchmark=self.benchmark_input.text().strip(),
                                  corpus_root=self.root_input.text().strip(),
                                  split=self.split_combo.currentData() if mode == 'benchmark' else 'test',
                                  depth=self.depth_spin.value(), systems=self.system_combo.currentData())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Could not start evaluation', str(exc))

    def refresh_job(self):
        controller = self.controller
        self.job_status.setText(controller.message)
        self.progress.setVisible(controller.busy)
        self.run_button.setEnabled(not controller.busy)
        self.cancel_button.setEnabled(controller.state == 'running')
        for widget in (self.mode_combo, self.depth_spin, self.system_combo, self.custom_widget):
            widget.setEnabled(not controller.busy)
        self.open_button.setEnabled(controller.report_dir is not None)
        self.export_report_button.setEnabled(controller.report_dir is not None)
        if controller.result is self._rendered_result:
            return
        self._rendered_result = controller.result
        self._render_report()

    def _render_report(self):
        result = self.controller.result
        if not result:
            for table in (self.summary_table, self.query_table, self.rankings_table):
                table.setRowCount(0)
            self.result_note.setText('No completed evaluation to display.')
            return
        meta = result['metadata']
        k = max(meta['cutoffs'])
        depth = meta['depth']
        self.summary_table.setHorizontalHeaderLabels(['System', f'Precision@{k}', f'Recall@{k}',
                                                     f'MRR@{depth}', f'nDCG@{k}', 'Mean query (ms)'])
        self.query_table.setHorizontalHeaderLabels(['System', 'Query ID', f'Precision@{k}', f'Recall@{k}',
                                                   f'RR@{depth}', f'nDCG@{k}', 'Query (ms)'])
        _fill(self.summary_table, [[_system(row['system']), _number(row[f'macro_precision@{k}'], True),
             _number(row[f'macro_recall@{k}'], True), _number(row['mrr_at_depth']),
             _number(row[f'macro_ndcg@{k}']), f"{row['mean_latency_ms']:.1f}"] for row in result['aggregate']])
        _fill(self.query_table, [[_system(row['system']), row['query_id'], _number(row[f'precision@{k}'], True),
             _number(row[f'recall@{k}'], True), _number(row['reciprocal_rank']),
             _number(row[f'ndcg@{k}']), f"{row['latency_ms']:.1f}"] for row in result['per_query']])
        _fill(self.rankings_table, [[_system(row['system']), row['query_id'], row['rank'], row['document_id'],
             _number(row['score']), row['relevance']] for row in result['rankings']])
        summary = result['aggregate'][0]
        prefix = 'SAMPLE DATA · ' if meta.get('synthetic_demo') else ''
        self.result_note.setText(
            f"{prefix}{meta['documents']} PDFs · {summary['queries']} queries · "
            f"{summary['eligible_recall_rr_ndcg_queries']} eligible for recall, MRR and nDCG. "
            f"{summary['zero_relevant_queries']} queries have no relevant PDF; their recall/MRR/nDCG are undefined "
            'and excluded from those averages. Precision includes all queries and uses k slots even if fewer '
            'results exist. Query timing excludes model loading, indexing and TF-IDF fitting. '
            'FileSense and TF-IDF scores use different scales; compare quality metrics, not raw scores.')

    def export_history(self):
        path = self._choose_excel_path('Export search history as Excel', 'filesense-searches')
        if path is None:
            return
        try:
            output = export_history_workbook(self.controller.history.records, path)
            self._open_exported_file(output)
        except (ImportError, OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Export failed', str(exc))

    def open_report(self):
        if self.controller.report_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.controller.report_dir)))

    def export_report(self):
        if self.controller.result is None:
            return
        path = self._choose_excel_path('Export evaluation as Excel', 'filesense-evaluation')
        if path is None:
            return
        try:
            output = export_evaluation_workbook(self.controller.result, path)
            self._open_exported_file(output)
        except (ImportError, OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Export failed', str(exc))

    def _choose_excel_path(self, title, prefix):
        default = unique_folder(Path.home(), prefix).with_suffix('.xlsx')
        path, _ = QFileDialog.getSaveFileName(
            self, title, str(default), 'Excel workbook (*.xlsx)',
            options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return None
        output = Path(path)
        return output.with_suffix('.xlsx') if not output.suffix else output

    def _open_exported_file(self, output):
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(output))):
            QMessageBox.information(self, 'Excel report saved', f'The report was saved to:\n{output}')
