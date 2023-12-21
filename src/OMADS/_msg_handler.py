from dataclasses import dataclass, field
import logging
from enum import Enum, auto
import time
import shutil
import os

class MSG_TYPE(Enum):
  DEBUG = auto()
  WARNING = auto()
  ERROR = auto()
  INFO = auto()
  CRITICAL = auto()

@dataclass
class logger:
  log: None = None

  def initialize(self, file: str):
    # Create and configure logger 
    logging.basicConfig(filename=file, 
              format='%(message)s', 
              filemode='w') 

    #Let us Create an object 
    self.log = logging.getLogger() 

    #Now we are going to Set the threshold of logger to DEBUG 
    self.log.setLevel(logging.DEBUG) 
    cur_time = time.strftime("%H:%M:%S", time.localtime())
    self.log_msg(msg=f"###################################################### \n", msg_type=MSG_TYPE.INFO)
    self.log_msg(msg=f"####################### OMADS ver. 2312 ######################### \n", msg_type=MSG_TYPE.INFO)
    self.log_msg(msg=f"###################### {cur_time} ###################### \n", msg_type=MSG_TYPE.INFO)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create and configure logger 
    logging.basicConfig(filename=file, 
              format='%(asctime)s %(message)s', 
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
