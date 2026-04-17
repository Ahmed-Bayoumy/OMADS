# ... (existing imports)

import sys
from .._setup._options import Options
from .._barriers._barriers import AdaptiveBarrier
from .._include import DESIGN_STATUS
import numpy as np
from .._points._candidate_point import CandidatePoint
from typing import List

# _original_stderr = sys.stderr          # keep a reference
# sys.stderr = open(os.devnull, 'w')     # silence everything on stderr


class ProgressBar:
  COLORS = {
      "pending": "\033[0m",
      "feasible": "\033[32m",
      "infeasible": "\033[33m",
      "error": "\033[31m",
      "unevaluated": "\033[2;35m"  # "\033[35m",
  }

  def __init__(
          self, options: Options, active_barrier: AdaptiveBarrier,
          p_name="untitled_prob"):
    self.options = options
    self.active_barrier = active_barrier
    self.length = 50
    self.p_name: str = p_name

  def _update_counts(self, status_list=None):
    """Count statuses and update `status_list` in place."""
    if status_list is None:
      status_list = ["pending"] * self.options.budget

    nf = ninf = nu = npen = nerr = 0
    elements: List[CandidatePoint] = [self.active_barrier.elements[i]
                                      for i in range(self.options.budget)]
    for idx, elem in enumerate(elements):
      if elem is None or elem.status == DESIGN_STATUS.UNEVALUATED:
        status_list[idx] = "pending"
        npen += 1
      elif elem is None or (isinstance(elem.status, float) and np.isnan(elem.status)):
        status_list[idx] = "unevaluated"
        nu += 1
      elif elem.is_feasible():
        status_list[idx] = "feasible"
        nf += 1
      elif elem.status == DESIGN_STATUS.INFEASIBLE:
        status_list[idx] = "infeasible"
        ninf += 1
      elif elem.status == DESIGN_STATUS.ERROR:
        status_list[idx] = "error"
        nerr += 1

    return nf, ninf, nu, npen, nerr, status_list

  def _build_bar(self, counts):
    """Return a string containing the coloured bar."""
    filler = "█"
    prog_bar = ""
    for count, color_key in zip(counts,
                                ["infeasible", "feasible",
                                 "unevaluated", "pending", "error"]):
      filled_length = int(self.length * count // self.options.budget)
      color = self.COLORS.get(color_key, self.COLORS["pending"])
      prog_bar += f"{color}{filler}{self.COLORS['pending']}" * filled_length
    return prog_bar

  def display(self, peval: int, prefix="", status_list=None):
    nf, ninf, nu, npen, nerr, status_list = self._update_counts(status_list)
    # os.system('cls' if os.name == 'nt' else 'clear')
    prog_bar = self._build_bar((ninf, nf, nu, npen, nerr))

    legend = (
        f"Pending: {self.COLORS['pending']} █ {self.COLORS['pending']} "
        f", Feasible: {self.COLORS['feasible']} █ {self.COLORS['pending']} "
        f", Infeasible: {self.COLORS['infeasible']} █ {self.COLORS['pending']}"
    )
    # `end=''` keeps the cursor on the same line; `flush=True` forces an update
    sys.stdout.write(
        "\r\033[K" + f"\r{self.p_name}: |{legend} |{prog_bar}| {peval}/{self.options.budget} #Evaluated\n")
