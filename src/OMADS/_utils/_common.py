"""
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the BSD 3-Clause License as published by the Free Software                #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the BSD 3-Clause License for more details.                #
#                                                                                     #
#  You should have received a copy of the BSD 3-Clause License along                  #
#  with this program. If not, see <https://opensource.org/license/bsd-3-clause/>.     #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/OMADS                                             #
#  Copyright (C) 2026  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#
"""


import logging
import sys
import time
import os
import numpy as np
import json
import OMADS._utils._globals as _globals

np.set_printoptions(legacy='1.21')


class validator:

  def check_input_file(self, args) -> dict:
    if type(args[0]) is dict:
      data = args[0]
    elif isinstance(args[0], str):
      if os.path.exists(os.path.abspath(args[0])):
        _, file_extension = os.path.splitext(args[0])
        if file_extension == ".json":
          try:
            with open(args[0]) as file:
              data = json.load(file)
          except ValueError:
            raise IOError('invalid json file: ' + args[0])
        else:
          raise IOError(f"The input file {args[0]} is not a JSON dictionary. "
                        f"Currently, OMADS supports JSON files solely!")
      else:
        raise IOError(f"Couldn't find {args[0]} file!")
    else:
      raise IOError(
          "The first input argument couldn't be recognized. "
          "It should be either a dictionary object or a JSON file that holds "
          "the required input parameters.")
    return data


class logger:
  log: None = None
  is_verbose: bool = False

  def __init__(self, log: None = None, is_verbose: bool = False):
    self.log = log
    self.is_verbose = is_verbose

  def initialize(self, file: str, w_time=False, is_verbose=False):
    # Create and configure logger
    self.is_verbose = is_verbose
    logging.basicConfig(filename=file,
                        format='%(message)s',
                        filemode='w')

    # Let us Create an object
    self.log = logging.getLogger()

    # Now we are going to Set the threshold of logger to DEBUG
    self.log.setLevel(logging.DEBUG)
    cur_time = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    self.log_msg(
        msg="###################################################### \n",
        msg_type=_globals.MSG_TYPE.INFO)
    self.log_msg(
        msg=f"################# OMADS release no {2604.01} #################### \n",
        msg_type=_globals.MSG_TYPE.INFO)
    self.log_msg(
        msg=f"############### {cur_time} ################# \n",
        msg_type=_globals.MSG_TYPE.INFO)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
      logging.root.removeHandler(handler)

    # Create and configure logger
    if w_time:
      logging.basicConfig(filename=file,
                          format='%(asctime)s %(message)s',
                          filemode='a')
    else:
      logging.basicConfig(filename=file,
                          format='%(message)s',
                          filemode='a')

    # Let us Create an object
    self.log = None
    self.log = logging.getLogger()

    # Now we are going to Set the threshold of logger to DEBUG
    self.log.setLevel(logging.DEBUG)

  def log_msg(self, msg: str, msg_type: _globals.MSG_TYPE):
    if msg_type == _globals.MSG_TYPE.DEBUG:
      self.log.debug(msg)
    elif msg_type == _globals.MSG_TYPE.INFO:
      self.log.info(msg)
    elif msg_type == _globals.MSG_TYPE.WARNING:
      self.log.warning(msg)
    elif msg_type == _globals.MSG_TYPE.ERROR:
      self.log.error(msg)
    elif msg_type == _globals.MSG_TYPE.CRITICAL:
      self.log.critical(msg)


def total_size(obj, seen=None):
  """Recursively finds the total size of an object including attributes."""
  if seen is None:
    seen = set()

  obj_id = id(obj)
  if obj_id in seen:
    return 0

  seen.add(obj_id)
  size = sys.getsizeof(obj)

  if isinstance(obj, dict):
    size += sum(total_size(k, seen) + total_size(v, seen)
                for k, v in obj.items())
  elif hasattr(obj, '__dict__'):
    size += total_size(vars(obj), seen)
  elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
    size += sum(total_size(i, seen) for i in obj)

  return size
