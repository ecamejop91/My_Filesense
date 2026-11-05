"""Locate existing Word passages using Word's actual, read-only pagination."""
from array import array
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
import re
import sys


def _clean(text):
    # Word uses a cell-end character in table text in addition to whitespace.
    return re.sub(r'[\s\x07]+', ' ', text).strip()


class PageMap:
    def __init__(self, pages):
        chars = []
        self.numbers = array('I')
        for number, text in enumerate(pages, 1):
            for char in text:
                if char.isspace() or char == '\x07':
                    if not chars or chars[-1] == ' ':
                        continue
                    char = ' '
                chars.append(char)
                self.numbers.append(number)
        self.text = ''.join(chars).rstrip()

    def locate(self, passage, source=''):
        passage, source = _clean(passage), _clean(source)
        if not passage:
            return None
        # The full indexed chunk disambiguates repeated sentences when possible.
        needle = source if source and passage in source and source in self.text else passage
        offset = needle.find(passage)
        matches = set()
        start = self.text.find(needle)
        while start >= 0:
            first = start + offset
            last = first + len(passage) - 1
            matches.add((self.numbers[first], self.numbers[last]))
            start = self.text.find(needle, start + 1)
        # Do not guess when identical text appears on different pages.
        return matches.pop() if len(matches) == 1 else None


def _read_word_pages(paths):
    """Use a separate Word instance, without saving or touching open documents."""
    results = {path: None for path in paths}
    if sys.platform != 'win32':
        return results
    try:
        import pythoncom
        from win32com.client import DispatchEx
    except ImportError:
        return results

    word = None
    update_links = None
    pythoncom.CoInitialize()
    try:
        word = DispatchEx('Word.Application')
        word.Visible = False
        word.DisplayAlerts = 0
        word.AutomationSecurity = 3  # Disable document macros.
        update_links = word.Options.UpdateLinksAtOpen
        word.Options.UpdateLinksAtOpen = False
        for path in paths:
            document = None
            try:
                document = word.Documents.Open(
                    FileName=str(path), ReadOnly=True, AddToRecentFiles=False,
                    ConfirmConversions=False, Visible=False,
                    PasswordDocument='', WritePasswordDocument='',
                )
                document.Repaginate()
                count = document.ComputeStatistics(2)  # wdStatisticPages
                starts = [document.GoTo(What=1, Which=1, Count=number).Start
                          for number in range(1, count + 1)]
                ends = starts[1:] + [document.Content.End]
                results[path] = [document.Range(start, end).Text
                                 for start, end in zip(starts, ends)]
            except Exception:
                # Searching remains usable without Word or for protected files.
                pass
            finally:
                if document is not None:
                    with suppress(Exception):
                        document.Close(SaveChanges=0)
    except Exception:
        pass
    finally:
        try:
            if word is not None:
                with suppress(Exception):
                    if update_links is not None:
                        word.Options.UpdateLinksAtOpen = update_links
                with suppress(Exception):
                    word.Quit(SaveChanges=0)
        finally:
            pythoncom.CoUninitialize()
    return results


class WordPageLocator:
    def __init__(self, cache_size=8, reader=None):
        self.cache_size = cache_size
        self._reader = reader or _read_word_pages
        self._cache = OrderedDict()

    def enrich(self, results):
        pending = {}
        for row in results:
            if Path(row['path']).suffix.lower() != '.docx':
                continue
            row.pop('match_pages', None)
            path = Path(row['path']).resolve()
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (path, stat.st_size, stat.st_mtime_ns)
            pending.setdefault(key, []).append(row)
        missing = [key[0] for key in pending if key not in self._cache]
        if missing:
            pages = self._reader(missing)
            for key in pending:
                if key not in self._cache:
                    content = pages.get(key[0])
                    self._cache[key] = PageMap(content) if content is not None else None
        for key, rows in pending.items():
            mapping = self._cache[key]
            self._cache.move_to_end(key)
            if mapping is not None:
                for row in rows:
                    row['match_pages'] = mapping.locate(row['text'], row.get('_source_text', ''))
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return results
