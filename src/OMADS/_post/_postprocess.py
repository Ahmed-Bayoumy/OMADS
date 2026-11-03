"""
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <https://www.gnu.org/licenses/>.                    #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/OMADS                                             #
#  Copyright (C) 2026  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#
"""
from dataclasses import field
import os
from typing import List, Dict, Any, Optional, Tuple
import json
import csv

import numpy as np
from .._include import CandidatePoint


class Output:
  """ Results output file decorator
  """
  file_path: str
  vnames: List[str]
  fnames: List[str]
  file_writer: Any = field(init=False)
  field_names: List[str] = field(default_factory=list)
  pname: str = "MADS0"
  runfolder: str = "undefined"
  replace: bool = True
  step_name: str = "Poll"
  suffix: str = "all"
  h_max: float = np.inf

  def __init__(self, file_path: str = None,
               vnames: List[str] = None,
               fnames: List[str] = None,
               file_writer: Any = None,
               field_names: List[str] = [],
               pname: str = "MADS0",
               runfolder: str = "undefined",
               replace: bool = True,
               step_name: str = "Poll",
               suffix: str = "all",
               h_max: float = np.inf):

    self.file_path: str = file_path
    self.vnames: List[str] = vnames
    self.fnames: List[str] = fnames
    self.file_writer: Any = file_writer
    self.field_names: List[str] = field_names
    self.pname: str = pname
    self.runfolder: str = runfolder
    self.replace: bool = replace
    self.step_name: str = step_name
    self.suffix: str = suffix
    self.h_max: float = h_max

    if not os.path.exists(self.file_path):
      os.mkdir(self.file_path)
    self.field_names = [
        f'{"Runtime (Sec)".rjust(25)}', f'{"Iteration".rjust(25)}',
        f'{"Evaluation #".rjust(25)}', f'{"Step".rjust(25)}',
        f'{"Source".rjust(25)}', f'{"Model_name".rjust(25)}',
        f'{"Delta".rjust(25)}', f'{"Status".rjust(25)}', f'{"phi".rjust(25)}']
    for k in self.fnames:
      self.field_names.append(f'{f"{k}".rjust(25)}')
    self.field_names += [f'{"max(c_in)".rjust(25)}',
                         f'{"Penalty_parameter".rjust(25)}',
                         f'{"Multipliers".rjust(25)}', f'{"hmax".rjust(25)}']
    for k in self.vnames:
      self.field_names.append(f'{f"{k}".rjust(25)}')
    sp = os.path.join(self.file_path, self.runfolder)
    if not os.path.exists(sp):
      os.makedirs(sp)
    if self.replace:
      with open(os.path.abspath(sp + f'/{self.pname}_{self.suffix}.out'),
                'w', newline='', encoding='utf-8') as f:
        self.file_writer = csv.DictWriter(f, fieldnames=self.field_names)
        self.file_writer.writeheader()

  def clear_csv_content(self):
    header_line = None
    rows = []

    # Read the header and rows
    sp = os.path.join(self.file_path, self.runfolder)
    if not os.path.exists(sp):
      os.makedirs(sp)
    with open(os.path.abspath(sp + f'/{self.pname}_{self.suffix}.out'),
              'r', newline='', encoding='utf-8') as csvfile:
      reader = csv.reader(csvfile)
      header_line = next(reader)  # Read the header line
      for row in reader:
        rows.append(row)

    # Truncate the file to remove existing content
    with open(os.path.abspath(sp + f'/{self.pname}_{self.suffix}.out'),
              'w', newline='', encoding='utf-8') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(header_line)  # Write back the header line

    csvfile.close()

  def add_row(
          self, eval_time: int, iterno: int, evalno: int, source: str,
          m_name: str, poll_size: float, status: str, fobj: Any, h: float,
          f: float, rho: float, lambdas: List[float],
          hmax: float, x: List[float],
          step_name: str, fnames: List[str]):

    row = {
        f'{"Runtime (Sec)".rjust(25)}': f'{f"{eval_time}".rjust(25)}',
        f'{"Iteration".rjust(25)}': f'{f"{iterno}".rjust(25)}',
        f'{"Evaluation #".rjust(25)}': f'{f"{evalno}".rjust(25)}',
        f'{"Step".rjust(25)}': f'{f"{step_name}".rjust(25)}',
        f'{"Source".rjust(25)}': f'{f"{source}".rjust(25)}',
        f'{"Model_name".rjust(25)}': f'{f"{m_name}".rjust(25)}',
        f'{"Delta".rjust(25)}': f'{f"{min(poll_size)}".rjust(25)}',
        f'{"Status".rjust(25)}': f'{f"{status}".rjust(25)}',
        f'{"phi".rjust(25)}': f'{f"{max(f)}".rjust(25)}'}
    for i, _ in enumerate((fobj)):
      row.update({f'{f"{fnames[i]}".rjust(25)}': f'{f"{fobj[i]}".rjust(25)}'})

    row.update(
        {f'{"max(c_in)".rjust(25)}': f'{f"{h}".rjust(25)}',
         f'{"Penalty_parameter".rjust(25)}': f'{f"{rho}".rjust(25)}',
         f'{"Multipliers".rjust(25)}':
         f'{f"{max(lambdas) if len(lambdas) > 0 else None}".rjust(25)}',
         f'{"hmax".rjust(25)}': f'{f"{hmax}".rjust(25)}'})

    ss = 0
    for k in range(13 + len(fnames), len(self.field_names)):
      row[self.field_names[k]] = f'{f"{x[ss]}".rjust(25)}'
      ss += 1
    with open(os.path.abspath(os.path.join(os.path.join(self.file_path, self.runfolder),
                                           f'{self.pname}_{self.suffix}.out')),
              'a', newline='', encoding='utf-8') as file:
      self.file_writer = csv.DictWriter(file, fieldnames=self.field_names)
      self.file_writer.writerow(row)


class PostMADS:
  """ Results postprocessor
  """
  x_incumbent: List[CandidatePoint]
  xmin: CandidatePoint
  coords: List[List[CandidatePoint]] = field(default_factory=list)
  poll_dirs: List[CandidatePoint] = field(default_factory=list)
  iter: List[int] = field(default_factory=list)
  bb_eval: List[int] = field(default_factory=list)
  psize: List[float] = field(default_factory=list)
  step_name: Optional[List[str]] = None
  nd_points: List[CandidatePoint] = field(default_factory=list)
  counter: int = 0
  # evaluation number -> (iteration, frame size, step name) of every row
  # written to the results file, so the Pareto file can reuse them.
  rows_info: Dict[int, Tuple[Any, Any, str]] = field(default_factory=dict)

  def __init__(self, x_incumbent: Optional[List[CandidatePoint]] = None,
               xmin: CandidatePoint = None,
               coords: Optional[List[List[CandidatePoint]]] = None,
               poll_dirs: Optional[List[CandidatePoint]] = None,
               iter: Optional[List[int]] = None,
               bb_eval: Optional[List[int]] = None,
               psize: Optional[List[float]] = None,
               step_name: Optional[List[str]] = None,
               nd_points: Optional[List[CandidatePoint]] = None,
               counter: int = 0):
    # Fresh lists per instance: shared mutable defaults made every run in a
    # process append to (and misalign) the previous runs' lists.
    self.x_incumbent = [] if x_incumbent is None else x_incumbent
    self.xmin = xmin
    self.coords = [] if coords is None else coords
    self.poll_dirs = [] if poll_dirs is None else poll_dirs
    self.iter = [] if iter is None else iter
    self.bb_eval = [] if bb_eval is None else bb_eval
    self.psize = [] if psize is None else psize
    self.step_name = [] if step_name is None else step_name
    self.nd_points = [] if nd_points is None else nd_points
    self.counter = counter
    self.rows_info = {}
    # Frame center of each batch in `coords` (None for an empty batch).
    self.coords_centers: List[Optional[CandidatePoint]] = []

  def _row_info(self, i: int, n: int):
    """(iteration, frame size, step name) of the i-th of the n pending rows.

    The evaluators append one entry per evaluated point to `iter`, `psize`
    and `step_name`, so the pending points are the last n entries; after a
    flush, `iter` and `psize` keep their last entry (the frame size of the
    next step's first row may be unknown otherwise)."""
    def tail(values, default):
      j = len(values) - n + i
      return values[j] if values and 0 <= j < len(values) else default
    psize = tail(self.psize, self.psize[-1] if self.psize else [np.nan])
    return (tail(self.iter, self.iter[-1] if self.iter else 0), psize,
            tail(self.step_name, "Poll-2n"))

  def output_results(self, out: Output, all_res: bool = True):
    """ Create a results file from the saved cache"""
    if all_res:
      self.counter = 0
    self.h_max = out.h_max
    n = len(self.poll_dirs)
    for i, p in enumerate(self.poll_dirs):
      if p.evaluated:
        iterno, psize, step = self._row_info(i, n)
        self.rows_info[p.eval_no] = (iterno, psize, step)
        out.add_row(
            eval_time=p.eval_time, iterno=iterno, evalno=p.eval_no,
            poll_size=psize,
            source=p.source, m_name=p.model, f=p.f, status=p.status.name,
            h=self.h_max,
            fobj=p.fobj, rho=p.rho, lambdas=p.lambda_multipliers,
            x=p.coordinates, hmax=self.h_max, step_name=step,
            fnames=out.fnames)
        self.counter += 1

    # del self.poll_dirs[-2]
    del self.poll_dirs
    self.poll_dirs = []
    del self.step_name
    self.step_name = []

    self.bb_eval = self.bb_eval[-1:]
    self.psize = self.psize[-1:]
    self.iter = self.iter[-1:]

  def output_nd_results(self, out: Output):
    """Rewrite the Pareto file with the current non-dominated points
    (`nd_points`), each with the iteration, frame size and step it was
    evaluated with (see `output_results`)."""
    out.clear_csv_content()
    self.h_max = out.h_max
    fallback = (self.iter[-1] if self.iter else 0,
                self.psize[-1] if self.psize else [np.nan], "Poll-2n")
    for p in sorted(self.nd_points, key=lambda q: q.eval_no):
      if p.evaluated:
        iterno, psize, step = self.rows_info.get(p.eval_no, fallback)
        out.add_row(
            eval_time=p.eval_time, iterno=iterno, evalno=p.eval_no,
            poll_size=psize,
            source=p.source, m_name=p.model, f=p.f, status=p.status.name,
            h=self.h_max,
            fobj=p.fobj, rho=p.rho, lambdas=p.lambda_multipliers,
            x=p.coordinates, hmax=self.h_max, step_name=step,
            fnames=out.fnames)

  def output_coordinates(self, out: Output):
    """ Save spinners in a json file """
    with open(out.file_path + f"/{out.runfolder}/coords.json", "w", encoding='utf-8') as json_file:
      dict_out = {}
      # Each batch with its own frame center: `x_incumbent` has one entry per
      # evaluated point, not per batch, and empty batches made it run out.
      center = self.x_incumbent[0] if self.x_incumbent else None
      for ii, ob in enumerate(self.coords, start=1):
        batch_center = (self.coords_centers[ii - 1]
                        if ii - 1 < len(self.coords_centers) else None)
        if batch_center is not None:
          center = batch_center
        entry: Dict[Any, Any] = {}
        p = [ib.coordinates for ib in ob]
        entry['iter'] = ii
        entry['coord'] = p
        entry['x_incumbent'] = None if center is None else center.coordinates
        dict_out[ii] = entry
        del entry
      json.dump(dict_out, json_file, indent=4, sort_keys=True)

  def __str__(self):
    """ Initialize the log file """
    return (
        f'iteration {self.iter[-1]}, '
        f'bbeval= {self.bb_eval[-1]}, '
        f'min(psize)= {min(self.psize[-1])}, '
        f'hmax = {self.h_max if self.h_max else None}, '
        f'hmin = {self.xmin.h if self.xmin else None}, '
        f'status: {self.xmin.status.name if self.xmin else None}, '
        f'fmin = {self.xmin.fobj if self.xmin else None}'
    )

  def __add_to_cache__(self, x: CandidatePoint):
    self.x_incumbent.append(x)
