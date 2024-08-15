
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - ORTHO-MADS (MADS)                                    #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU Lesser General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   #
#                                                                                     #
#  You should have received a copy of the GNU Lesser General Public License along     #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/OMADS                                             #
#  Copyright (C) 2022  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#

import copy
from dataclasses import dataclass, field
import importlib
import logging
import operator
import time
import shutil
import os
from typing import List, Dict, Any
import numpy as np
from .CandidatePoint import CandidatePoint
import json
from ._globals import *

np.set_printoptions(legacy='1.21')

@dataclass
class validator:

  def checkInputFile(self, args) -> dict:
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
      raise IOError("The first input argument couldn't be recognized. "
              "It should be either a dictionary object or a JSON file that holds "
              "the required input parameters.")
    return data

@dataclass
class logger:
  log: None = None
  isVerbose: bool = False

  def initialize(self, file: str, wTime = False, isVerbose = False):
    # Create and configure logger 
    self.isVerbose = isVerbose
    logging.basicConfig(filename=file, 
              format='%(message)s', 
              filemode='w') 

    #Let us Create an object 
    self.log = logging.getLogger() 

    #Now we are going to Set the threshold of logger to DEBUG 
    self.log.setLevel(logging.DEBUG) 
    cur_time = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    self.log_msg(msg=f"###################################################### \n", msg_type=MSG_TYPE.INFO)
    self.log_msg(msg=f"################# OMADS ver. 2401 #################### \n", msg_type=MSG_TYPE.INFO)
    self.log_msg(msg=f"############### {cur_time} ################# \n", msg_type=MSG_TYPE.INFO)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create and configure logger 
    if wTime:
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
  
  def log_msg(self, msg: str, msg_type: MSG_TYPE):
    if msg_type == MSG_TYPE.DEBUG:
      self.log.debug(msg) 
    elif msg_type == MSG_TYPE.INFO:
      self.log.info(msg) 
    elif msg_type == MSG_TYPE.WARNING:
      self.log.warning(msg) 
    elif msg_type == MSG_TYPE.ERROR:
      self.log.error(msg) 
    elif msg_type == MSG_TYPE.CRITICAL:
      self.log.critical(msg) 
  
  def relocate_logger(self, source_file: str = None, Dest_file: str = None):
    if Dest_file is not None and source_file is not None and os.path.exists(source_file):
      shutil.copy(source_file, Dest_file)
      if os.path.exists("DSMToDMDO.yaml"):
        shutil.copy("DSMToDMDO.yaml", Dest_file)
      # Remove all handlers associated with the root logger object.
      for handler in logging.root.handlers[:]:
          logging.root.removeHandler(handler)
      # Create and configure logger 
      logging.basicConfig(filename=os.path.join(Dest_file, "DMDO.log"), 
                format='%(asctime)s %(message)s', 
                filemode='a')
      #Let us Create an object 
      self.log = logging.getLogger() 

      #Now we are going to Set the threshold of logger to DEBUG 
      self.log.setLevel(logging.DEBUG) 
