import os
import tempfile
import unittest

from rowtune.data import read_text_file, windows_from_texts
from tests.tiny import ByteTokenizer

TOK = ByteTokenizer()
LONG = ["abcdefghij" * 30 + str(i) for i in range(40)]  # 301-302 characters each, above the 200-character floor


class WindowsFromTexts(unittest.TestCase):
    def test_count_and_length(self):
        w = windows_from_texts(TOK, LONG, n=10, length=64, seed=1)
        self.assertEqual(len(w), 10)
        self.assertTrue(all(len(x) == 64 for x in w))

    def test_short_documents_are_dropped(self):
        w = windows_from_texts(TOK, ["short"] * 50, n=3, length=8, seed=1)
        self.assertEqual(w, [])

    def test_same_seed_same_windows_other_seed_other_order(self):
        a = windows_from_texts(TOK, LONG, n=5, length=64, seed=1)
        b = windows_from_texts(TOK, LONG, n=5, length=64, seed=1)
        c = windows_from_texts(TOK, LONG, n=5, length=64, seed=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_stops_at_n_and_returns_fewer_when_text_runs_out(self):
        self.assertEqual(len(windows_from_texts(TOK, LONG[:2], n=100, length=64, seed=1)), 602 // 64)


class ReadTextFile(unittest.TestCase):
    def _write(self, text):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "t.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p

    def test_splits_on_doc_marker_when_present(self):
        p = self._write("first doc\nsecond line of first\n<<<DOC>>>\nsecond doc")
        self.assertEqual(read_text_file(p), ["first doc\nsecond line of first", "second doc"])

    def test_splits_on_lines_otherwise(self):
        p = self._write("line one\nline two\n")
        self.assertEqual(read_text_file(p), ["line one", "line two", ""])


if __name__ == "__main__":
    unittest.main()
