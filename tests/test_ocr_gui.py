"""Verify GUI wiring without opening a window or downloading models."""
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import analysis_progress as analysis


class OCRGUIIntegrationTests(unittest.TestCase):
    def test_analyze_tags_from_stored_ocr_chunk_vectors(self):
        pdf = Path('/tmp/image-only.pdf')
        pdf = pdf.resolve()
        store = dict(doc_records=None, chunk_records=None, chunk_vectors=None,
                     model=object(), added_doc_records=[dict(pdf_path=str(pdf), method='ocr')], skipped_paths=[])
        plan = {'needed': {str(pdf)}}
        vectors = {str(pdf): dict(vectors=[[1.0, 0.0, 0.0]], char_count=12)}
        worker = analysis.AnalysisWorker([pdf], Path('/tmp/store'))
        completed, failed = Mock(), Mock()
        worker.completed.connect(completed)
        worker.failed.connect(failed)
        with patch.object(analysis, 'update_vector_store', return_value=store) as update, \
             patch.object(analysis, 'prepare_tagging', return_value=plan), \
             patch.object(analysis, 'load_tagging_vectors', return_value=vectors) as load, \
             patch.object(analysis, 'tag_files', return_value={}) as tag:
            worker.run()
        self.assertFalse(update.call_args.kwargs['full_result'])
        load.assert_called_once_with(plan, Path('/tmp/store/file_tags.sqlite3'))
        tag.assert_called_once_with([pdf], db_path=Path('/tmp/store/file_tags.sqlite3'),
            vectors_by_path=vectors, model=store['model'], progress=worker.report, plan=plan)
        completed.assert_called_once_with(store, {})
        failed.assert_not_called()
