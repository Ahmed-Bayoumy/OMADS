"""
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the BSD 3-Clause License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the BSD 3-Clause License for more details.   #
#                                                                                     #
#  You should have received a copy of the BSD 3-Clause License along     #
#  with this program. If not, see <https://opensource.org/license/bsd-3-clause/>.     #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/OMADS                                             #
#  Copyright (C) 2026  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#
"""
from dataclasses import field
import os
from typing import List, Dict, Any, Optional
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

  def __init__(self, x_incumbent: List[CandidatePoint] = [],
               xmin: CandidatePoint = None,
               coords: List[List[CandidatePoint]] = [],
               poll_dirs: List[CandidatePoint] = [],
               iter: List[int] = [],
               bb_eval: List[int] = [],
               psize: List[float] = [],
               step_name: Optional[List[str]] = [],
               nd_points: List[CandidatePoint] = [],
               counter: int = 0):
    self.x_incumbent = x_incumbent
    self.xmin = xmin
    self.coords = coords
    self.poll_dirs = poll_dirs
    self.iter = iter
    self.bb_eval = bb_eval
    self.psize = psize
    self.step_name = step_name
    self.nd_points = nd_points
    self.counter = counter

  def output_results(self, out: Output, all_res: bool = True):
    """ Create a results file from the saved cache"""
    if all_res:
      self.counter = 0
    self.h_max = out.h_max
    for i, p in enumerate(self.poll_dirs):
      if p.evaluated:
        out.add_row(
            eval_time=p.eval_time, iterno=self.iter[i],
            evalno=self.bb_eval[i],
            poll_size=self.psize[i],
            source=p.source, m_name=p.model, f=p.f, status=p.status.name,
            h=self.h_max,
            fobj=p.fobj, rho=p.rho, lambdas=p.lambda_multipliers,
            x=p.coordinates, hmax=self.h_max, step_name="Poll-2n"
            if self.step_name is None else self.step_name[i],
            fnames=out.fnames)
        self.counter += 1

    # del self.poll_dirs[-2]
    del self.poll_dirs
    self.poll_dirs = []
    del self.step_name
    self.step_name = []

    self.bb_eval = [self.bb_eval[-1]]
    self.psize = [self.psize[-1]]
    self.iter = [self.iter[-1]]

  def output_nd_results(self, out: Output):
    """ Create a results file from the saved cache"""
    counter = 0
    out.clear_csv_content()
    self.h_max = out.h_max
    for p in self.nd_points:
      if p.evaluated and counter < len(self.iter):
        out.add_row(
            eval_time=p.eval_time, iterno=self.iter[counter],
            evalno=p.eval_no, poll_size=self.psize[counter],
            source=p.source, m_name=p.model, f=p.f, status=p.status.name,
            h=self.h_max,
            fobj=p.fobj, rho=p.rho, lambdas=p.lambda_multipliers,
            x=p.coordinates, hmax=self.h_max, step_name="Poll-2n"
            if self.step_name is None else self.step_name[counter],
            fnames=out.fnames)
        counter += 1

  def output_coordinates(self, out: Output):
    """ Save spinners in a json file """
    with open(out.file_path + f"/{out.runfolder}/coords.json", "w", encoding='utf-8') as json_file:
      dict_out = {}
      for ii, ob in enumerate(self.coords, start=1):
        entry: Dict[Any, Any] = {}
        p = [ib.coordinates for ib in ob]
        entry['iter'] = ii
        entry['coord'] = p
        entry['x_incumbent'] = self.x_incumbent[ii - 1].coordinates
        dict_out[ii] = entry
        del entry
      json.dump(dict_out, json_file, indent=4, sort_keys=True)

  def __str__(self):
    """ Initialize the log file """
    return f'iteration {
        self.iter[-1]}, bbeval=  'f'{
        self.bb_eval[-1]}, min(psize)=  {
        min(self.psize[-1])}, hmax =  {
        self.h_max if self.h_max else None}, hmin =  {
        self.xmin.h if self.xmin else None}, status: {
        self.xmin.status.name if self.xmin else None}  , fmin =  {
        self.xmin.fobj if self.xmin else None} '

  def __add_to_cache__(self, x: CandidatePoint):
    self.x_incumbent.append(x)
