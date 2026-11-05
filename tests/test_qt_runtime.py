"""Desktop startup must support libraries that expect console output streams."""
import io
import sys
import unittest
from unittest.mock import patch

from qt_runtime import prepare_standard_streams


class StandardStreamTests(unittest.TestCase):
    def test_windowed_launch_provides_writable_output_streams(self):
        with patch.object(sys, 'stdout', None), patch.object(sys, 'stderr', None):
            prepare_standard_streams()
            stdout, stderr = sys.stdout, sys.stderr
            try:
                for stream in (stdout, stderr):
                    # Windows treats NUL as a character device; callers need a
                    # valid isatty() method regardless of its platform result.
                    self.assertIsInstance(stream.isatty(), bool)
                    stream.write('Model loading progress\n')
                    stream.flush()
                prepare_standard_streams()
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
            finally:
                stdout.close()
                stderr.close()

    def test_existing_output_streams_are_preserved(self):
        with io.StringIO() as output, patch.object(sys, 'stdout', output), \
                patch.object(sys, 'stderr', output):
            prepare_standard_streams()
            self.assertIs(sys.stdout, output)
            self.assertIs(sys.stderr, output)
            print('console output remains available')
            self.assertEqual(output.getvalue(), 'console output remains available\n')
