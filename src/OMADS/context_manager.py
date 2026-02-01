import sys
import re


class WarningSuppressor:
  """
  A file‑like object that forwards everything to the original stderr,
  except lines that look like warnings.
  """

  def __init__(self, real_stderr):
    self._real = real_stderr
    # Simple heuristic: a warning line usually contains "Warning:" or ends with "warning"
    self._warn_re = re.compile(r"(?i)\bwarning\b")

  def write(self, text):
    if not self._warn_re.search(text):
      self._real.write(text)

  def flush(self):
    self._real.flush()


# # Install the wrapper
# _original_stderr = sys.stderr          # keep a reference
# sys.stderr = WarningSuppressor(sys.stderr)
