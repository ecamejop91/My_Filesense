#!/usr/bin/env python3
"""Desktop interface for indexing and searching a local PDF library."""

from __future__ import annotations

import html
from pathlib import Path
import sys

from qt_runtime import prepare_app_identity, prepare_qt_plugins, prepare_standard_streams

# pythonw has no console streams. Prepare them before importing ML libraries
# that inspect or write to stdout/stderr while loading their models.
prepare_standard_streams()

try:
    from app_storage import configure_runtime, PATHS
    configure_runtime()
except Exception as exc:
    # pythonw has no console, so storage failures must be visible before any
    # library is opened. Never fall back to a database beside the application.
    from PyQt6.QtWidgets import QApplication, QMessageBox
    _startup_app = QApplication.instance() or QApplication([])
    QMessageBox.critical(None, 'FileSense private storage unavailable', str(exc))
    raise SystemExit(1) from exc

from ocr import DEFAULT_OCR_MODEL
from search_snippets import PassageSelector
from word_page_lookup import WordPageLocator
from document_formats import FILE_FILTER, is_supported, location_label

from document_search_engine import (
    VECTOR_STORE_DIR,
    TOP_K,
    load_vector_store,
    search_index,
    get_embedding_model,
    merge_index_delta,
)
from document_tag_store import (
    delete_file_tags,
    exact_tag_matches,
    load_tag_definitions,
    load_tag_rows,
    save_tag_definition,
    tag_color,
    tag_names,
)

try:
    from PyQt6.QtCore import Qt, QUrl
    from PyQt6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter, QPen, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QDialog,
        QFileDialog,
        QFrame,
        QInputDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QCheckBox,
        QMainWindow,
        QMenu,
        QScrollArea,
        QSizePolicy,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "PyQt6 is required to run this GUI. Install it with 'pip install PyQt6'."
    ) from exc


ASSETS_DIR = Path(__file__).resolve().parent / 'assets'
APP_ICON_PATH = ASSETS_DIR / ('filesense.ico' if sys.platform == 'win32' else 'filesense.png')


class DocumentQueryVisuals:
    """Owns the widget tree and interactions for the main window."""

    def setup_ui(self) -> None:
        self._analysis_worker = None
        self._analysis_dialog = None
        self._analysis_outcome = None
        self._close_after_analysis = False
        self.index_data = None
        self.tag_filters = {}
        self._last_results = None
        self._last_results_kind = None
        self._last_query = None
        self._passage_selector = PassageSelector()
        self._word_page_locator = WordPageLocator()
        self.vector_store_dir = VECTOR_STORE_DIR
        self.setWindowTitle("FileSense")
        self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.ExpandedClientAreaHint
                            | Qt.WindowType.NoTitleBarBackgroundHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ContentsMarginsRespectsSafeArea, False)
        self.setMenuWidget(self._build_title_bar())

        self.resize(1416, 1014)
        self.setMinimumSize(1080, 806)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(30, 26, 30, 24)
        layout.setSpacing(20)
        layout.addWidget(self._build_file_section())
        layout.addWidget(self._build_tag_filter())

        lower_layout = QHBoxLayout()
        left_column = QVBoxLayout()
        left_column.addWidget(self._build_query_section())
        left_column.addWidget(self._build_results_section(), 1)
        lower_layout.setSpacing(20)
        lower_layout.addLayout(left_column, 3)
        lower_layout.addWidget(self._build_categories_section(), 2)

        layout.addLayout(lower_layout, 1)

        self.setCentralWidget(central)
        self._apply_theme()
        self.categories_view.setHtml(self._format_categories_html())
        self.statusBar().setToolTip('Private library for your current computer account\n' + str(PATHS.root.resolve()))
        self.statusBar().showMessage("Ready. Add documents, configure tags, or search the archive.")

    def _build_title_bar(self) -> QFrame:
        title_bar = QFrame()
        title_bar.setObjectName('titleBar')
        title_bar.setFixedHeight(36)
        layout = QHBoxLayout(title_bar)
        # Leave room for the system-drawn app title/icon and caption buttons.
        layout.setContentsMargins(128, 3, 144, 3)
        self.settings_button = QPushButton('Settings')
        self.settings_button.setObjectName('settingsButton')
        self.settings_button.setFixedHeight(28)
        self.settings_button.setShortcut('Alt+S')
        settings_menu = QMenu(self.settings_button)
        settings_menu.addAction('Clear Database', self.clear_database)
        self.settings_button.setMenu(settings_menu)
        layout.addWidget(self.settings_button)
        layout.addStretch()
        return title_bar

    def clear_database(self) -> None:
        answer = QMessageBox.question(
            self, "Clear Database",
            "Remove all indexed documents, extracted text, embeddings, and saved tag assignments "
            "from your current account's private library?\n\n"
            "Your files and custom category definitions will be kept. "
            "Loaded documents will stay in the queue so you can analyze them again.\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            from document_database import clear_database
            clear_database(self.vector_store_dir)
        except Exception as exc:
            self._show_warning(f"Could not clear the database: {exc}")
            return
        self.index_data = None
        self._clear_query_cache()
        for check in self.tag_filters.values():
            check.blockSignals(True)
            check.setChecked(False)
            self._style_tag_filter(check, False)
            check.blockSignals(False)
        self.tag_filter_scope.setText("All documents · select tags to narrow your search")
        self.query_input.clear()
        self.results_view.setHtml(self._empty_results_html())
        self.categories_view.setHtml(self._format_categories_html())
        self.analysis_progress.setValue(0)
        self.analysis_progress.setVisible(False)
        self.statusBar().showMessage("Database cleared. Load documents or analyze the queued files to start again.")

    def _build_file_section(self) -> QWidget:
        section = QWidget()
        section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(10)

        heading = QLabel('ADD DOCUMENTS')
        heading.setObjectName('sectionTitle')
        section_layout.addWidget(heading)

        panel = QFrame()
        panel.setObjectName('documentControls')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        hint = QLabel("Choose PDF, Word, PowerPoint, Excel, or RTF files, then analyze them to create searchable text and categories.")
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.file_list = QListWidget()
        self.file_list.setObjectName('documentQueue')
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.file_list.setUniformItemSizes(True)
        self.file_list.setSpacing(0)
        self.file_list.setWordWrap(False)
        self.file_list.model().rowsInserted.connect(self._update_file_queue)
        self.file_list.model().rowsRemoved.connect(self._update_file_queue)
        self.file_list.model().modelReset.connect(self._update_file_queue)
        self.file_list.setAlternatingRowColors(False)
        self.file_list.itemDoubleClicked.connect(self._open_file_item)
        self.file_list.setVisible(False)

        button_row = QHBoxLayout()
        self.add_btn = QPushButton("Load Documents")
        self.add_btn.setObjectName("secondaryButton")
        self.add_btn.clicked.connect(self.add_files)
        self.analyze_btn = QPushButton("Analyze Files")
        self.analyze_btn.setObjectName("primaryButton")
        self.analyze_btn.clicked.connect(self.analyze_files)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setObjectName("quietButton")
        self.remove_btn.clicked.connect(self.remove_selected_files)

        button_row.addWidget(self.add_btn)
        button_row.addWidget(self.analyze_btn)
        button_row.addWidget(self.remove_btn)
        button_row.addStretch()

        self.analysis_progress = QProgressBar()
        self.analysis_progress.setVisible(False)
        self.analysis_progress.setTextVisible(True)
        self.analysis_progress.setMinimum(0)
        self.analysis_progress.setMaximum(1)
        self.analysis_progress.setValue(0)

        layout.addWidget(self.file_list)
        layout.addLayout(button_row)
        layout.addWidget(self.analysis_progress)
        section_layout.addWidget(panel)
        return section

    def _build_tag_filter(self) -> QWidget:
        section = QWidget()
        section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        heading = QLabel('TAG FILTER')
        heading.setObjectName('sectionTitle')
        layout.addWidget(heading)

        self.tag_filter_scope = QLabel("All documents · select tags to narrow your search")
        self.tag_filter_scope.setObjectName("sectionHint")
        self.tag_filter_scope.setWordWrap(True)
        layout.addWidget(self.tag_filter_scope)

        panel = QFrame()
        panel.setObjectName('tagFilterPanel')
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 2, 8, 2)
        self.tag_filter_scroll = QScrollArea()
        self.tag_filter_scroll.setObjectName('tagFilterScroll')
        self.tag_filter_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.tag_filter_scroll.setWidgetResizable(True)
        self.tag_filter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tag_filter_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tag_filter_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tag_filter_scroll.setFixedHeight(60)
        content = QWidget()
        content.setObjectName('tagFilterContent')
        row = QHBoxLayout(content)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(8)
        for tag in tag_names():
            check = QCheckBox(tag)
            # Preserve the full label and font size; the viewport scrolls instead
            # of squeezing the checkbox row to the window width.
            check.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            check.setChecked(False)
            check.setProperty("tag_color", tag_color(tag))
            self._style_tag_filter(check, False)
            check.toggled.connect(self._tag_filter_changed)
            check.toggled.connect(lambda checked, widget=check: self._style_tag_filter(widget, checked))
            self.tag_filters[tag] = check
            row.addWidget(check)
        row.addStretch()
        self.tag_filter_scroll.setWidget(content)
        panel_layout.addWidget(self.tag_filter_scroll)
        layout.addWidget(panel)
        return section

    def _style_tag_filter(self, check: QCheckBox, checked: bool) -> None:
        background = check.property("tag_color") if checked else "#e2e0d9"
        foreground = "#ffffff" if checked else "#7b817c"
        check.setStyleSheet(
            "QCheckBox { "
            f"color: {foreground}; background-color: {background}; "
            "border-radius: 5px; padding: 5px 9px; font-weight: bold; }"
        )

    def _build_query_section(self) -> QWidget:
        section = QWidget()
        section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(10)

        heading = QLabel('SEARCH ARCHIVE')
        heading.setObjectName('sectionTitle')
        layout.addWidget(heading)

        self.query_input = QLineEdit()
        self.query_input.setObjectName('archiveSearch')
        self.query_input.setAccessibleName('Search archive')
        self.query_input.setPlaceholderText('Ask a question or search by tag...')
        self.query_input.setClearButtonEnabled(True)
        self.query_input.setMinimumHeight(50)
        heading.setBuddy(self.query_input)

        # Draw a crisp magnifier without platform-dependent icon fonts or assets.
        icon = QPixmap(48, 48)
        icon.setDevicePixelRatio(2)
        icon.fill(Qt.GlobalColor.transparent)
        painter = QPainter(icon)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor('#20604f'), 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawEllipse(4, 4, 12, 12)
        painter.drawLine(14, 14, 20, 20)
        painter.end()
        self.search_action = self.query_input.addAction(QIcon(icon), QLineEdit.ActionPosition.TrailingPosition)
        self.search_action.setText('Search')
        self.search_action.setToolTip('Search archive (Enter)')
        self.search_action.triggered.connect(self.search_documents)
        self.query_input.returnPressed.connect(self.search_documents)

        layout.addWidget(self.query_input)
        return section

    def _build_results_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        heading = QLabel('RESULTS')
        heading.setObjectName('sectionTitle')
        layout.addWidget(heading)

        self.results_view = QTextBrowser()
        self.results_view.setObjectName('archiveResults')
        self.results_view.setAccessibleName('Search results')
        self.results_view.setReadOnly(True)
        self.results_view.setOpenLinks(False)
        self.results_view.setOpenExternalLinks(False)
        self.results_view.anchorClicked.connect(self._open_link)
        self.results_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_view.customContextMenuRequested.connect(self._results_context_menu)
        self.results_view.setHtml(self._empty_results_html())

        layout.addWidget(self.results_view, 1)
        return section

    def _build_categories_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(10)

        heading = QLabel('DOCUMENT CATEGORIES')
        heading.setObjectName('sectionTitle')
        layout.addWidget(heading)

        self.configure_tags_btn = QPushButton("Create Custom Tags")
        self.configure_tags_btn.setObjectName("secondaryButton")
        self.configure_tags_btn.setToolTip(
            'Create your own document categories with a tag name and descriptive key phrases.\n'
            'Then run Analyze Files to apply them to your documents.'
        )
        self.configure_tags_btn.setAccessibleDescription(self.configure_tags_btn.toolTip())
        self.configure_tags_btn.clicked.connect(self.open_tag_configuration)
        layout.addWidget(self.configure_tags_btn)

        self.categories_view = QTextBrowser()
        self.categories_view.setObjectName('documentCategories')
        self.categories_view.setAccessibleName('Document categories')
        self.categories_view.setReadOnly(True)
        self.categories_view.setHtml(self._empty_categories_html())

        layout.addWidget(self.categories_view, 1)
        return section

    def _apply_theme(self) -> None:
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(
            """
            QMainWindow { background: #f3f0e8; }
            QWidget { color: #202825; }
            QFrame#titleBar { background: #f3f0e8; }
            QFrame#documentControls, QFrame#tagFilterPanel { background: #ffffff; border: 1px solid #57967b; border-radius: 12px; }
            QScrollArea#tagFilterScroll, QWidget#tagFilterContent { background: transparent; border: none; }
            QLabel#sectionHint { color: #6c746e; font-size: 12px; font-weight: 400; letter-spacing: 0; }
            QLabel#sectionTitle { color: #38655b; font-size: 11px; font-weight: 800; letter-spacing: 1.1px; }
            QLabel#inputLabel { color: #6c746e; font-size: 10px; font-weight: 800; letter-spacing: 1px; }
            QDialog { background: #f3f0e8; }
            QFrame#tagConfigFrame { background: #fcfbf6; border: none; border-radius: 8px; }
            QLabel#dialogTitle { color: #173f3a; font-size: 18px; font-weight: 700; }
            QListWidget, QTextBrowser { background: #ffffff; border: 1px solid #e1ddd2; border-radius: 8px; padding: 8px; selection-background-color: #d5eadc; selection-color: #173f3a; }
            QListWidget::item { padding: 7px 8px; border-radius: 4px; }
            QListWidget::item:selected { background: #d5eadc; }
            QListWidget#documentQueue { padding: 4px; }
            QListWidget#documentQueue::item { padding: 2px 8px; }
            QLineEdit { background: #ffffff; border: 1px solid #d1cdc2; border-radius: 8px; padding: 0 12px; font-size: 13px; }
            QLineEdit:focus { border: 2px solid #57967b; }
            QLineEdit#archiveSearch { border-radius: 12px; padding: 0 14px; font-size: 14px; }
            QTextBrowser#archiveResults, QTextBrowser#documentCategories { border: 1px solid #57967b; border-radius: 12px; padding: 14px; }
            QTextBrowser#archiveResults:focus, QTextBrowser#documentCategories:focus { border: 2px solid #57967b; }
            QPushButton { border: none; border-radius: 8px; font-weight: 700; padding: 9px 14px; }
            QPushButton#primaryButton { background: #20604f; color: #ffffff; }
            QPushButton#primaryButton:hover { background: #173f3a; }
            QPushButton#secondaryButton { background: #d5eadc; color: #173f3a; }
            QPushButton#secondaryButton:hover { background: #c4e0ce; }
            QPushButton#quietButton { background: transparent; border: 1px solid #d1cdc2; color: #4f5b55; }
            QPushButton#quietButton:hover { background: #f0ede5; }
            QPushButton#settingsButton { background: transparent; color: #4f5b55; font-weight: 500; border-radius: 4px; padding: 3px 10px; }
            QPushButton#settingsButton:hover { background: #e6e1d6; }
            QProgressBar { border: 1px solid #d1cdc2; border-radius: 6px; background: #ece8de; text-align: center; height: 16px; }
            QProgressBar::chunk { background: #57967b; border-radius: 5px; }
            QStatusBar { background: #e6e1d6; color: #53605a; font-size: 11px; }
            """
        )

    def _empty_results_html(self) -> str:
        return (
            "<div style='margin:32px 18px;color:#6c746e;'>"
            "<div style='font-size:18px;color:#173f3a;font-weight:600;'>Ready when you are.</div>"
            "<p>Search the existing archive, or add documents above to expand it.</p>"
            "</div>"
        )

    def _empty_categories_html(self) -> str:
        definitions = load_tag_definitions()
        tag_preview = self._tag_badges_html(list(definitions.keys())[:8])
        return (
            "<div style='margin:24px 10px;color:#6c746e;'>"
            "<div style='font-size:15px;color:#173f3a;font-weight:600;'>No analyzed documents yet.</div>"
            "<p>Add documents and click Analyze Files. Add custom tags first when you want new categories included.</p>"
            f"<div style='margin-top:10px;'>{tag_preview}</div>"
            "</div>"
        )

    def _tag_badges_html(self, tags: list[str], available_tags=None) -> str:
        if not tags:
            return ""

        if available_tags is None:
            available_tags = tag_names()
        cells = []
        for tag in tags:
            color = tag_color(tag, available_tags)
            display_tag = html.escape(tag).replace(" ", "&nbsp;")
            cells.append(
                "<td bgcolor='"
                f"{color}"
                "' style='color:#ffffff;padding:5px 10px;font-size:11px;"
                f"font-weight:700;border-radius:2px;white-space:nowrap;'>{display_tag}</td>"
            )
        spacer = "<td width='8'></td>"
        return (
            "<table align='right' cellspacing='0' cellpadding='0' border='0'>"
            f"<tr>{spacer.join(cells)}</tr>"
            "</table>"
        )

    def _format_categories_html(self) -> str:
        rows = []
        available_tags = tag_names()
        for row in load_tag_rows():
            tags = [row["tag_1"], row["tag_2"], row["tag_3"]]
            visible_tags = [tag for tag in tags if tag]
            tag_html = self._tag_badges_html(visible_tags, available_tags)
            if not visible_tags:
                tag_html = "<span style='color:#87908a;'>No tags</span>"
            rows.append(
                "<tr>"
                "<td style='padding:10px 8px;border-bottom:1px solid #ece8de;"
                f"color:#202825;font-weight:600;'>{html.escape(row['file_name'])}</td>"
                "<td style='padding:10px 8px;border-bottom:1px solid #ece8de;text-align:right;'>"
                f"{tag_html}"
                "</td>"
                "</tr>"
            )

        if not rows:
            return self._empty_categories_html()

        return (
            "<div style='margin:4px 2px;'>"
            "<table width='100%' cellspacing='0' cellpadding='0' style='border-collapse:collapse;'>"
            "<thead><tr>"
            "<th align='left' style='padding:0 8px 8px;color:#6c746e;font-size:11px;'>Document</th>"
            "<th align='right' style='padding:0 8px 8px;color:#6c746e;font-size:11px;'>Tags</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
            "</div>"
        )

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select documents",
            str(Path.home()),
            FILE_FILTER,
        )
        for raw_path in paths:
            path = Path(raw_path)
            if not is_supported(path):
                self._show_warning(f"{path.name}: unsupported file type.")
                continue
            if not path.exists():
                self._show_warning(f"{path} does not exist and was skipped.")
                continue
            if not self._list_contains_path(path):
                item = QListWidgetItem(path.name)
                item.setToolTip(str(path))
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.file_list.addItem(item)

        if paths:
            self.statusBar().showMessage(f"Added {len(paths)} document file(s) to the queue.")

    def _update_file_queue(self) -> None:
        count = self.file_list.count()
        self.file_list.setVisible(count > 0)
        if count:
            row_height = self.file_list.sizeHintForRow(0)
            # Keep small queues snug, and scroll larger queues after five rows.
            self.file_list.setFixedHeight(min(count, 5) * row_height
                                          + 2 * self.file_list.frameWidth())

    def _show_warning(self, message: str) -> None:
        QMessageBox.warning(self, "Notice", message)

    def _list_contains_path(self, path: Path) -> bool:
        for index in range(self.file_list.count()):
            if Path(self.file_list.item(index).data(Qt.ItemDataRole.UserRole)) == path:
                return True
        return False

    def _selected_file_paths(self) -> list[Path]:
        return [Path(self.file_list.item(index).data(Qt.ItemDataRole.UserRole)) for index in range(self.file_list.count())]

    def _open_file_item(self, item: QListWidgetItem) -> None:
        path = Path(item.data(Qt.ItemDataRole.UserRole)).expanduser().resolve()
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self._show_warning(f"Could not open file: {path}")

    def _active_tags(self) -> set[str]:
        return {tag.casefold() for tag, check in self.tag_filters.items() if check.isChecked()}

    def _filter_result_rows(self, results, path_key: str):
        active = self._active_tags()
        if not active:
            return results
        tag_rows = {str(Path(row["file_path"]).resolve()): row for row in load_tag_rows()}
        filtered = []
        for result in results:
            path = str(Path(result[path_key]).resolve())
            row = tag_rows.get(path)
            row_tags = {str(row[f"tag_{i}"]).casefold() for i in range(1, 4) if row and row[f"tag_{i}"]}
            if row_tags & active:
                filtered.append(result)
        return filtered

    def _tag_filter_changed(self) -> None:
        active_names = [tag for tag, check in self.tag_filters.items() if check.isChecked()]
        scope = "Matching any of: " + ", ".join(active_names) if active_names else "All documents"
        self.tag_filter_scope.setText(scope)
        if self._last_query is None:
            if not active_names:
                self.results_view.setHtml(self._empty_results_html())
                self.statusBar().showMessage("All documents. Enter a query to search the archive.")
                return
            combined = {}
            for tag in active_names:
                for result in exact_tag_matches(tag):
                    combined[str(Path(result["file_path"]).resolve())] = result
            results = list(combined.values())
            self.results_view.setHtml(self._format_tag_results_html(results))
            self.statusBar().showMessage(f"Displayed {len(results)} tag-filtered document(s).")
            return
        # Reuse the last submitted query, even if the input now contains an
        # unsent edit. Filtering is not a new search.
        self._display_query_results()

    def _clear_query_cache(self) -> None:
        self._last_results = None
        self._last_results_kind = None
        self._last_query = None

    def _display_query_results(self):
        key = "path" if self._last_results_kind == "search" else "file_path"
        filtered = self._filter_result_rows(self._last_results, key)
        if self._last_results_kind == "search":
            filtered = filtered[:TOP_K]
            if filtered:
                self._passage_selector.select(self._last_query, filtered, self.index_data['model'])
                self._word_page_locator.enrich(filtered)
            render = self._format_search_results_html
        else:
            render = self._format_tag_results_html
        if filtered:
            self.results_view.setHtml(render(filtered))
        else:
            self.results_view.setPlainText("No results match this query and the selected tag filters.")
        self.statusBar().showMessage(f'Displayed {len(filtered)} result(s) for "{self._last_query}". '
                                     + ("Matching selected tags." if self._active_tags() else "All documents."))
        return filtered

    def _file_uri(self, path: Path) -> str:
        return path.resolve().as_uri()

    def _open_link(self, url: QUrl) -> None:
        if not QDesktopServices.openUrl(url):
            self._show_warning(f"Could not open link: {url.toString()}")

    def _results_context_menu(self, position) -> None:
        url = self.results_view.anchorAt(position)
        if not url:
            return
        file_path = Path(QUrl(url).toLocalFile()).expanduser().resolve()
        menu = QMenu(self.results_view)
        open_location = menu.addAction("Open location")
        remove_file = menu.addAction("Remove file from database")
        rename_file = menu.addAction("Rename file")
        chosen = menu.exec(self.results_view.viewport().mapToGlobal(position))
        if chosen == open_location:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path.parent)))
        elif chosen == remove_file:
            self._remove_file_from_database(file_path)
        elif chosen == rename_file:
            self._rename_file(file_path)

    def _rename_file(self, file_path: Path) -> None:
        new_name, accepted = QInputDialog.getText(self, "Rename file", "New file name:", text=file_path.name)
        if not accepted or not new_name.strip():
            return
        new_name = new_name.strip()
        if not Path(new_name).suffix:
            new_name += file_path.suffix
        elif Path(new_name).suffix.lower() != file_path.suffix.lower():
            self._show_warning('Keep the original file extension when renaming a document.')
            return
        try:
            if '/' in new_name or '\\' in new_name:
                raise ValueError('Enter a file name without directory separators.')
            new_path = file_path.with_name(new_name)
            if new_path == file_path:
                return
            if new_path.exists():
                self._show_warning(f"A file named '{new_path.name}' already exists in that folder.")
                return
        except (ValueError, OSError) as exc:
            self._show_warning(f"Could not rename file: {exc}")
            return
        answer = QMessageBox.question(self, "Confirm rename", f"Rename '{file_path.name}' to '{new_path.name}' on the computer and update FileSense?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        renamed = False
        try:
            self._clear_query_cache()
            file_path.rename(new_path)
            renamed = True
            from document_search_engine import rename_document_in_vector_store
            if not rename_document_in_vector_store(file_path, new_path, self.vector_store_dir):
                raise ValueError("The document was not found in the vector index.")
        except Exception as exc:
            if renamed:
                try:
                    new_path.rename(file_path)
                except OSError as recovery_error:
                    self._show_warning(
                        f"Could not update the index: {exc}. The file remains at '{new_path}' "
                        f"because restoring its original name failed: {recovery_error}")
                    return
            self._show_warning(f"Could not rename file: {exc}")
            return

        # The database has committed both document and tag paths. A display
        # error must not undo just the physical filename after this point.
        try:
            old_target = str(file_path.resolve())
            new_target = str(new_path.resolve()).replace("\\", "/")
            if self.index_data is not None:
                for record in self.index_data["doc_records"]:
                    if str(Path(record["pdf_path"]).resolve()) == old_target:
                        record["pdf_path"] = new_target
                for chunk in self.index_data["chunk_records"]:
                    if str(Path(chunk["pdf_path"]).resolve()) == old_target:
                        chunk["pdf_path"] = new_target
            for index in range(self.file_list.count()):
                item = self.file_list.item(index)
                if str(Path(item.data(Qt.ItemDataRole.UserRole)).resolve()) == old_target:
                    item.setText(new_path.name)
                    item.setToolTip(str(new_path))
                    item.setData(Qt.ItemDataRole.UserRole, str(new_path))
            self.categories_view.setHtml(self._format_categories_html())
            self.results_view.setHtml(self._empty_results_html())
            self.statusBar().showMessage(f"Renamed file to {new_path.name}.")
        except Exception as exc:
            self.index_data = None
            self._show_warning(f"File renamed to '{new_path.name}', but the display could not refresh: {exc}")

    def _remove_file_from_database(self, file_path: Path) -> None:
        answer = QMessageBox.question(
            self, "Remove document",
            f"Remove '{file_path.name}' from the database, vector index, and saved tags?\n\nThe original file itself will not be deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._clear_query_cache()
            from document_search_engine import remove_document_from_vector_store
            if not remove_document_from_vector_store(file_path, self.vector_store_dir):
                raise ValueError("The document was not found in the vector index.")
            delete_file_tags(file_path)
            if self.index_data is not None:
                target = str(file_path.resolve())
                ids = {d["doc_id"] for d in self.index_data["doc_records"] if str(Path(d["pdf_path"]).resolve()) == target}
                self.index_data["doc_records"] = [d for d in self.index_data["doc_records"] if d["doc_id"] not in ids]
                kept = [i for i, c in enumerate(self.index_data["chunk_records"]) if c["doc_id"] not in ids]
                self.index_data["chunk_records"] = [self.index_data["chunk_records"][i] for i in kept]
                self.index_data["chunk_vectors"] = self.index_data["chunk_vectors"][kept] if kept else self.index_data["chunk_vectors"][:0]
                self.index_data["docs_by_id"] = {d["doc_id"]: d for d in self.index_data["doc_records"]}
            self.categories_view.setHtml(self._format_categories_html())
            self.results_view.setHtml(self._empty_results_html())
            self.statusBar().showMessage("Removed document from the database, vector index, and saved tags.")
        except Exception as exc:
            self._show_warning(f"Could not remove document: {exc}")

    def _tags_for_path(self, path: Path, rows=None) -> list[str]:
        target = str(path.resolve())
        for row in (load_tag_rows() if rows is None else rows):
            if str(Path(row["file_path"]).resolve()) == target:
                return [str(row[f"tag_{i}"]) for i in range(1, 4) if row[f"tag_{i}"]]
        return []

    def _format_search_results_html(self, results) -> str:
        blocks = []
        rows = load_tag_rows()
        for result in results:
            file_path = Path(result["path"]).resolve()
            score_100 = max(0.0, min(100.0, result["score"] * 100.0))
            tags = ", ".join(self._tags_for_path(file_path, rows)) or "No tags"
            passage_label = ''
            if file_path.suffix.lower() == '.docx':
                pages = result.get('match_pages')
                if pages:
                    start, end = pages
                    location = f'Page {start}' if start == end else f'Pages {start}–{end}'
                    label = f'Match found in {location}'
                else:
                    label = 'Match found · Page unavailable'
                passage_label = (
                    "<div style='color:#38655b;font-size:11px;margin-top:8px;'>"
                    f"{html.escape(label)}</div>"
                )
            elif 'snippet_score' in result:
                passage_score = max(0.0, min(100.0, result['snippet_score'] * 100.0))
                passage_label = (
                    "<div style='color:#38655b;font-size:11px;margin-top:8px;'>"
                    f"Closest short passage · {passage_score:.1f}% similarity</div>"
                )

            blocks.append(
                "<div style='margin:8px 3px 14px;padding:13px 14px;"
                "border:1px solid #e1ddd2;border-radius:8px;background:#fff;'>"
                f"<a title=\"{html.escape(str(file_path), quote=True)}\" style='color:#20604f;font-size:15px;font-weight:600;' href=\"{html.escape(self._file_uri(file_path), quote=True)}\">"
                f"{html.escape(result['document'])}</a>"
                f"<span style='color:#a8791a;'> &nbsp; {score_100:.1f}% section similarity</span><br>"
                f"<span style='color:#7b817c;'>{html.escape(location_label(file_path, result['page']))} · Tags: {html.escape(tags)}</span>"
                f"{passage_label}<p style='color:#37413c;margin:8px 0 0;'>{html.escape(result['text'])}</p>"
                "</div>"
            )

        return "".join(blocks)

    def _format_tag_results_html(self, results) -> str:
        blocks = []
        rows = load_tag_rows()
        for result in results:
            file_path = Path(result["file_path"]).resolve()
            score_100 = max(0, min(100, result["score"] * 100))
            tags = ", ".join(self._tags_for_path(file_path, rows)) or "No tags"

            blocks.append(
                "<div style='margin:8px 3px 14px;padding:13px 14px;"
                "border:1px solid #e1ddd2;border-radius:8px;background:#fff;'>"
                f"<a title=\"{html.escape(str(file_path), quote=True)}\" style='color:#20604f;font-size:15px;font-weight:600;' href=\"{html.escape(self._file_uri(file_path), quote=True)}\">"
                f"{html.escape(result['file_name'])}</a>"
                f"<span style='color:#a8791a;'> &nbsp; {score_100:.1f}% tag match</span><br>"
                f"<span style='color:#7b817c;'>Tags: {html.escape(tags)}</span>"
                "</div>"
            )

        return "".join(blocks)

    def remove_selected_files(self) -> None:
        selected = self.file_list.selectedItems()
        if not selected:
            self.statusBar().showMessage("No files selected.")
            return

        for item in selected:
            self.file_list.takeItem(self.file_list.row(item))

        self.statusBar().showMessage("Removed selected files.")

    def open_tag_configuration(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Custom Tags")
        dialog.setMinimumWidth(620)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)

        title = QLabel("Create Custom Tags")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)

        guidance = QLabel(
            "Create categories by entering a tag name and key phrases. "
            "Use short 2- or 3-word phrases instead of single keywords so the app has more context."
        )
        guidance.setObjectName("sectionHint")
        guidance.setWordWrap(True)
        layout.addWidget(guidance)

        example = QLabel(
            "Example: Travel = airline ticket, boarding pass, hotel reservation, "
            "rental car, trip confirmation"
        )
        example.setObjectName("sectionHint")
        example.setWordWrap(True)
        layout.addWidget(example)

        form_frame = QFrame()
        form_frame.setObjectName("tagConfigFrame")
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(14, 14, 14, 14)
        form_layout.setSpacing(10)

        tag_name_label = QLabel("TAG NAME")
        tag_name_label.setObjectName("inputLabel")
        tag_name_input = QLineEdit()
        tag_name_input.setPlaceholderText("Travel")
        tag_name_input.setMinimumHeight(40)

        phrases_label = QLabel("KEY PHRASES")
        phrases_label.setObjectName("inputLabel")
        phrases_input = QLineEdit()
        phrases_input.setPlaceholderText(
            "airline ticket, boarding pass, hotel reservation, rental car"
        )
        phrases_input.setMinimumHeight(40)

        form_layout.addWidget(tag_name_label)
        form_layout.addWidget(tag_name_input)
        form_layout.addWidget(phrases_label)
        form_layout.addWidget(phrases_input)
        layout.addWidget(form_frame)

        button_row = QHBoxLayout()
        save_btn = QPushButton("Save Tag")
        save_btn.setObjectName("primaryButton")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("quietButton")
        save_btn.clicked.connect(
            lambda: self.save_tag_configuration(
                tag_name_input.text(),
                phrases_input.text(),
                dialog,
            )
        )
        cancel_btn.clicked.connect(dialog.reject)
        button_row.addStretch()
        button_row.addWidget(cancel_btn)
        button_row.addWidget(save_btn)
        layout.addLayout(button_row)

        dialog.exec()

    def save_tag_configuration(
        self,
        tag_name_text: str,
        key_phrases_text: str,
        dialog: QDialog,
    ) -> None:
        try:
            tag_name = save_tag_definition(
                tag_name_text,
                key_phrases_text,
            )
        except ValueError as exc:
            self._show_warning(str(exc))
            return

        self.categories_view.setHtml(self._format_categories_html())
        self.statusBar().showMessage(
            f"Saved tag '{tag_name}'. Analyze documents again to apply updated categories."
        )
        dialog.accept()

    def analyze_files(self) -> None:
        if self._analysis_worker is not None:
            self._analysis_dialog.show()
            self._analysis_dialog.raise_()
            return
        file_paths = self._selected_file_paths()
        count = len(file_paths)
        if count == 0:
            self._show_warning("Add at least one file before running analysis.")
            return

        self.analysis_progress.setVisible(True)
        self.analysis_progress.setMaximum(count)
        self.analysis_progress.setValue(0)
        self._clear_query_cache()

        from analysis_progress import AnalysisProgressDialog, AnalysisWorker
        if self._analysis_dialog is not None:
            self._analysis_dialog.close()
            self._analysis_dialog.deleteLater()
        self._analysis_dialog = AnalysisProgressDialog(file_paths, self)
        worker = AnalysisWorker(file_paths, self.vector_store_dir, self,
                                ocr_model=DEFAULT_OCR_MODEL)
        self._analysis_worker = worker
        self._analysis_outcome = None
        worker.progress.connect(self._analysis_progress_changed)
        worker.completed.connect(self._analysis_completed)
        worker.failed.connect(self._analysis_failed)
        worker.cancelled.connect(self._analysis_cancelled)
        worker.finished.connect(self._analysis_finished)
        self._analysis_dialog.cancel_requested.connect(worker.requestInterruption)
        # Prevent edits/searches against a collection being replaced. The progress
        # window remains enabled and the UI event loop keeps painting/responding.
        self.centralWidget().setEnabled(False)
        self.settings_button.setEnabled(False)
        self.analysis_progress.setRange(0, 0)
        self.statusBar().showMessage("Analysis running. Follow the floating progress window.")
        self._analysis_dialog.show()
        worker.start()

    def _analysis_progress_changed(self, event) -> None:
        self._analysis_dialog.update_progress(event)
        detail = event.get('detail', '')
        path = event.get('path')
        self.statusBar().showMessage(f"{Path(path).name}: {detail}" if path else detail)

    def _analysis_completed(self, store, tags) -> None:
        if store.get('doc_records') is not None:
            self.index_data = store
        elif self.index_data is not None:
            # Incremental result: patch only the changed documents into the
            # index already in memory instead of reloading the whole library.
            self.index_data = merge_index_delta(self.index_data, store)
        # Otherwise the index loads lazily at the next search.
        self._analysis_outcome = ('Analysis complete',
            f"Indexed {len(store['added_doc_records'])} file(s); skipped {len(store['skipped_paths'])}. "
            "The collection is ready to search.")
        result_lines = []
        added = store["added_doc_records"]

        if added:
            result_lines.append("Indexed files:")
            for record in added:
                result_lines.append(
                    f"{Path(record['pdf_path']).name} [{record['method']}]"
                )
            result_lines.append("")

        if not result_lines:
            self.results_view.setHtml(self._empty_results_html())
        else:
            self.results_view.setPlainText("\n".join(result_lines))

    def _analysis_failed(self, message) -> None:
        self._analysis_outcome = ('Analysis failed', message + ' Some completed work may already be saved.')

    def _analysis_cancelled(self) -> None:
        self._analysis_outcome = ('Analysis cancelled',
            'Stopped safely. Some completed work may already be saved. Analyze again to continue.')

    def _analysis_finished(self) -> None:
        worker = self._analysis_worker
        self._analysis_worker = None
        if worker is not None:
            worker.deleteLater()
        outcome, message = self._analysis_outcome or ('Analysis failed', 'Worker ended unexpectedly.')
        if outcome != 'Analysis complete':
            # Index saving may finish before cancellation/tagging failure. Discard
            # stale in-memory data so the next search reloads the committed index.
            self.index_data = None
            self._clear_query_cache()
            self.results_view.setPlainText(message)
        self.centralWidget().setEnabled(True)
        self.settings_button.setEnabled(True)
        self.analysis_progress.setRange(0, 1)
        self.analysis_progress.setValue(1 if outcome == 'Analysis complete' else 0)
        self.analysis_progress.setVisible(False)
        try:
            self.categories_view.setHtml(self._format_categories_html())
        except Exception:
            self.categories_view.setPlainText('Categories could not be refreshed. Retry analysis.')
        self.statusBar().showMessage(f'{outcome}. {message}')
        self._analysis_dialog.finish(outcome, message)
        if self._close_after_analysis:
            self.close()

    def search_documents(self) -> None:
        query = self.query_input.text().strip()
        if not query:
            self._show_warning("Enter a query before searching.")
            return

        try:
            self._search_documents(query)
        except Exception as exc:
            self.statusBar().showMessage('Search failed. The index is unavailable.')
            self._show_warning(f"Search failed: {exc}")
            return

    def _search_documents(self, query: str) -> None:
        exact_query_tag = next((tag for tag in tag_names() if tag.casefold() == query.casefold()), None)
        if exact_query_tag is not None:
            tag_results = exact_tag_matches(exact_query_tag)
            self._last_query = query
            if not tag_results:
                self.results_view.setPlainText("No files were found for that tag.")
                self.statusBar().showMessage("No tag matches found.")
                self._last_results, self._last_results_kind = [], 'tag'
                return

            self._last_results = tag_results
            self._last_results_kind = "tag"
            self._display_query_results()
            return

        if self.index_data is None:
            self.statusBar().showMessage('Loading the search index…')
            self.index_data = load_vector_store(vector_store_dir=self.vector_store_dir)

        if not self.index_data["doc_records"]:
            self._show_warning("No indexed files were found. Run Analyze & Index first.")
            return

        if self.index_data['model'] is None:
            self.index_data['model'] = get_embedding_model()
        results = search_index(
            query=query,
            model=self.index_data["model"],
            chunk_records=self.index_data["chunk_records"],
            chunk_vectors=self.index_data["chunk_vectors"],
            docs_by_id=self.index_data["docs_by_id"],
            # The engine already scores and sorts the full corpus. Retain its
            # full ranking for this query so tag changes can filter BEFORE the
            # five-result cutoff without another model call. This adds one hit
            # row per chunk in memory until the next query or index change.
            top_k=len(self.index_data["chunk_records"]),
        )
        self._last_query = query

        if not results:
            self.results_view.setPlainText("No indexed text chunks are available to search.")
            self.statusBar().showMessage("No searchable text chunks found.")
            self._last_results, self._last_results_kind = [], 'search'
            return

        self._last_results = results
        self._last_results_kind = "search"
        self._display_query_results()


class DocumentQueryWindow(QMainWindow, DocumentQueryVisuals):
    def __init__(self) -> None:
        super().__init__()
        self.setup_ui()

    def closeEvent(self, event) -> None:
        if self._analysis_worker is not None:
            self._close_after_analysis = True
            if not self._analysis_dialog.stopping:
                self._analysis_dialog._button_clicked()
            self.statusBar().showMessage('Stopping analysis safely before closing…')
            event.ignore()
            return
        super().closeEvent(event)


def main() -> int:
    prepare_app_identity()
    prepare_qt_plugins()
    app = QApplication([])
    app.setApplicationName('FileSense')
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = DocumentQueryWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
