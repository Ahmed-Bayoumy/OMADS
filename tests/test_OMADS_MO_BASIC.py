import time
from OMADS import POLL, SEARCH, MADS
import copy
import os
import numpy as np
from typing import Dict, List
from multiprocessing import freeze_support
import platform
import logging

# Configure the logging
# Create a custom logger
logger = logging.getLogger('OMADS_MO_BBO_unit_test')
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only log INFO and above to console

# Create a file handler
file_handler = logging.FileHandler(filename='tests/OMADS_BBO_unit_test.log', mode = 'a')
file_handler.setLevel(logging.DEBUG)  # Log all messages to file

# Create a formatter and set it for handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Example filter to exclude messages from the root logger
class NoRootMessagesFilter(logging.Filter):
    def filter(self, record):
        return record.name != 'root'

# Add the filter to handlers
console_handler.addFilter(NoRootMessagesFilter())
file_handler.addFilter(NoRootMessagesFilter())

# logging.basicConfig(level=logging.DEBUG, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='tests/unit_tests_moo.log', filemode='w')



def common_dict():
  outDict: dict = {
    "evaluator":
      {
        "blackbox": None},

    "param":
      {
        "baseline": None,
        "lb": None,
        "ub": None,
        "var_names": ["x", "y"],
        "fun_names": ["f1", "f2"],
        # "constraints_type": ["PB", "PB"],
        "nobj": 2,
        "isPareto": True,
        "scaling": None,
        "LAMBDA": [1E5, 1E5],
        "RHO": 1.0,
        "h_max": np.inf,
        "meshType": "GMESH",
        "post_dir": None
      },

    "options":
      {
        "seed": 0,
        "budget": 2000,
        "tol": 1e-12,
        "psize_init": 1,
        "display": False,
        "opportunistic": False,
        "check_cache": True,
        "store_cache": True,
        "collect_y": False,
        "rich_direction": True,
        "precision": "medium",
        "save_results": True,
        "save_coordinates": False,
        "save_all_best": False,
        "parallel_mode": False
      },

    "search": {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": 10,
      "visualize": False
    },
  }
  return outDict

def MO_Binh_and_Korn(x):
  f1 = 4 * x[0]**2 + 4 * x[1]**2
  f2 = (x[0] - 5)**2 + (x[1] - 5)**2
  g1 = (x[0]-5)**2 + x[1]**2 -25
  g2 = 7.7 - (x[0]-8)**2 - (x[1]+3)**2 
  
  return [[f1, f2], [g1, g2]]

def MO_Chankong_and_Haimes(x):
  f1 = 2 + (x[0]-2)**2 + (x[1]-1)**2
  f2 = 9*x[0]-(x[1]-1)**2
  g1 = x[0]**2 + x[1]**2-225
  g2 = x[0] -3*x[1]+10
  
  return [[f1, f2], [g1, g2]]

def MO_Test_function_4(x):
  f1 = x[0]**2-x[1]
  f2 = -0.5*x[0]-x[1]-1
  g1 = -(6.5 - (x[0]/6) - x[1])
  g2 = -(7.5 - 0.5 *x[0] -x[1])
  g3 = -(30 - 5*x[0] -x[1])
  
  return [[f1, f2], [g1, g2, g3]]

def MO_Kursawe(x):
  f1 = sum([-10*np.exp(-0.2*np.sqrt(x[i]**2 + x[i+1]**2)) for i in range(2)])
  f2 = sum([abs(x[i])**0.8 + 5*np.sin(x[i]**3) for i in range(3)])

  return [[f1, f2], [0]]

def MO_Fonseca_Fleming(x):
  n = len(x)
  f1 = 1 - np.exp(-sum([(x[i]-(1/np.sqrt(n)))**2 for i in range(n)]))
  f2 = 1 - np.exp(-sum([(x[i]+(1/np.sqrt(n)))**2 for i in range(n)]))
  
  return [[f1, f2], [0]]

def MO_Osyczka_Kundu(x):
  f1 = -25*(x[0]-2)**2 - (x[1]-2)**2 - (x[2]-1)**2  - (x[3]-4)**2 - (x[4]-1)**2
  f2 = sum([x[i]**2 for i in range(6)]) 

  g1 = x[0] + x[1] -2
  g2 = 6 - x[0] - x[1] 
  g3 = 2 - x[1] + x[0]
  g4 = 2 - x[0] + 3*x[1]
  g5 = 4-(x[2]-3)**2 -x[3]
  g6 = (x[4]-3)**2 + x[5] -4

  return [[f1, f2], [-g1, -g2, -g3, -g4, -g5, -g6]]

def MO_CTP1(x):
  f1 = x[0]
  f2 = (1+x[1])*np.exp(-(x[0])/(1+x[1]))
  g1 = 1-((f2)/(0.858*np.exp(-0.541*f1)))
  g2 = 1-(f2/(0.728*np.exp(-0.295*f1)))

  return [[f1, f2], [g1, g2]]

def MO_Ex(x):
  f1 = x[0]
  f2 = (1+x[1])/x[0]

  g1 = 6-(x[1]+9*x[0])
  g2 = 1+x[1] - 9*x[0]

  return [[f1,f2],[g1,g2]]

def MO_ZDT1(x):
  f1 = x[0]  # objective 1
  g = 1 + 9 * np.sum(np.divide(x[1:len(x)], (len(x) - 1)))
  h = 1 - np.sqrt(f1 / g)
  f2 = g * h  # objective 2

  return [[f1, f2], [0]]

def MO_ZDT3(x):
  f1 = x[0]  # objective 1
  g = 1 + (9/(len(x) - 1)) * np.sum(x[1:len(x)])
  h = 1 - np.sqrt(f1 / g) - (f1/g)*np.sin(10*np.pi*f1)
  f2 = g * h  # objective 2

  return [[f1, f2], [0]]

def MO_ZDT4(x):
  f1 = x[0]  # objective 1
  g = 1 + 10*(len(x)-1) + np.sum([x[i]**2 - 10*np.cos(4*np.pi*x[i]) for i in range(1, len(x))])
  h = 1 - np.sqrt(f1 / g)
  f2 = g * h  # objective 2

  return [[f1, f2], [0]]

def MO_ZDT6(x):
  f1 = 1 - np.exp(-4*x[0]) * np.sin(6*np.pi*x[0])**6
  g = 1+9*(sum(x[1:len(x)])/9)**.25
  h = 1 - (f1/g)**2
  f2 = g * h  # objective 2

  return [[f1, f2], [0]]

def test_MO_Binh_and_Korn():
  logger.info('\nStarted running MO_Binh_and_Korn test... \n')
  tic = time.perf_counter()
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_Binh_and_Korn
  data["param"]["name"] = "Binh_and_Korn"
  data["param"]["baseline"] = [0, 0]
  data["param"]["lb"] = [0, 0]
  data["param"]["ub"] = [5, 3]
  data["param"]["constraints_type"] = ["PB", "PB"]
  data["param"]["scaling"] = [5, 3]
  data["param"]["post_dir"] = "./tests/bm/MOO/constrained/Binh_and_Korn/post"
  data["options"]["budget"] = 500
  data["param"]["ref_point"] = [140, 50]
  
  p_out, _ = POLL.main(data)
  s_out, _ = SEARCH.main(data)
  m_out, _ = MADS.main(data)
  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]
  
  toc = time.perf_counter()
  logger.info(f'Completed MO_Binh_and_Korn run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {0.8}\n poll step HV_obtained = {PHV}\n search step HV_expected = {0.79}\n search step HV_obtained = {SHV}\n MADS HV_expected = {0.8}\n MADS HV_obtained = {MHV}\n")
  if PHV < 0.8 or SHV < 0.79 or MHV < 0.8:
    logger.error("The MO_Binh_and_Korn QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_Binh_and_Korn QA test completed but failed.")
  else:
    logger.info("The MO_Binh_and_Korn QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")

def test_MO_Chankong_and_Haimes():
  logger.info('\nStarted running MO_Chankong_and_Haimes test... \n')
  tic = time.perf_counter()
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_Chankong_and_Haimes
  data["param"]["name"] = "Chankong_and_Haimes"
  data["param"]["baseline"] = [0, 0]
  data["param"]["lb"] = [-20, -20]
  data["param"]["ub"] = [20, 20]
  data["param"]["constraints_type"] = ["PB", "PB"]
  data["param"]["scaling"] = [40, 40]
  data["param"]["post_dir"] = "./tests/bm/MOO/constrained/Chankong_and_Haimes/post"
  data["options"]["budget"] = 500
  data["param"]["ref_point"] = [275, 0.1]
  
  # POLL.main(data)
  # SEARCH.main(data)
  p_out, _ = POLL.main(data)
  s_out, _ = SEARCH.main(data)
  m_out, _ = MADS.main(data)
  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]
  
  toc = time.perf_counter()
  logger.info(f'Completed MO_Chankong_and_Haimes run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {0.8}\n poll step HV_obtained = {PHV}\n search step HV_expected = {0.6}\n search step HV_obtained = {SHV}\n MADS HV_expected = {0.8}\n MADS HV_obtained = {MHV}\n")
  if PHV < 0.8 or SHV < 0.6 or MHV < 0.8:
    logger.error("The MO_Chankong_and_Haimes QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_Chankong_and_Haimes QA test completed but failed.")
  else:
    logger.info("The MO_Chankong_and_Haimes QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")

def test_MO_Fonseca_Fleming():
  logger.info('\nStarted running MO_Fonseca_Fleming test... \n')
  tic = time.perf_counter()
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_Fonseca_Fleming
  data["param"]["name"] = "Fonseca_Fleming"
  data["param"]["baseline"] = [0, 0]
  data["param"]["lb"] = [-4, -4]
  data["param"]["ub"] = [4, 4]
  data["meshType"] = "GMESH"
  # data["param"]["constraints_type"] = ["EB"]
  data["param"]["scaling"] = [8, 8]
  data["param"]["post_dir"] = "./tests/bm/MOO/unconstrained/Fonseca_Fleming/post"
  data["options"]["budget"] = 500
  data["param"]["ref_point"] = [1, 1]
  
  # POLL.main(data)
  # SEARCH.main(data)
  p_out, _ = POLL.main(data)
  s_out, _ = SEARCH.main(data)
  m_out, _ = MADS.main(data)
  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]
  
  toc = time.perf_counter()
  logger.info(f'Completed MO_Fonseca_Fleming run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {0.34}\n poll step HV_obtained = {PHV}\n search step HV_expected = {0.53}\n search step HV_obtained = {SHV}\n MADS HV_expected = {0.35}\n MADS HV_obtained = {MHV}\n")
  if PHV < 0.34 or SHV < 0.34 or MHV < 0.35:
    logger.error("The MO_Fonseca_Fleming QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_Fonseca_Fleming QA test completed but failed.")
  else:
    logger.info("The MO_Fonseca_Fleming QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")

def test_MO_Test_function_4():
  logger.info('\nStarted running MO_Test_function test... \n')
  tic = time.perf_counter()
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_Test_function_4
  data["param"]["name"] = "Test_function_4"
  data["param"]["baseline"] = [0, 0]#[3, 3]
  data["param"]["lb"] = [-7, -7]
  data["param"]["ub"] = [4, 4]
  data["meshType"] = "GMESH"
  data["param"]["constraints_type"] = ["PB", "PB"]
  data["param"]["scaling"] = [10, 10]
  data["param"]["post_dir"] = "./tests/bm/MOO/constrained/Test_function_4/post"
  data["options"]["budget"] = 500
  data["param"]["ref_point"] = [12, -5]
  
  

  p_out, _ = POLL.main(data)
  m_out, _ = MADS.main(data)
  s_out, _ = SEARCH.main(data)

  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]
  
  toc = time.perf_counter()
  logger.info(f'Completed MO_Test_function_4 run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {0.6}\n poll step HV_obtained = {PHV}\n search step HV_expected = {0.4}\n search step HV_obtained = {SHV}\n MADS HV_expected = {0.6}\n MADS HV_obtained = {MHV}\n")
  if PHV < 0.6 or SHV < 0.4 or MHV < 0.6:
    logger.error("The MO_Test_function_4 QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_Test_function_4 QA test completed but failed.")
  else:
    logger.info("The MO_Test_function_4 QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")
  
def test_MO_Kursawe():
  # TODO: uncon logic needs review
  logger.info('\nStarted running MO_Kursawe test... \n')
  tic = time.perf_counter()
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_Kursawe
  data["param"]["name"] = "Kursawe"
  # data["param"]["baseline"] = [-2.0, 0.5, -4.5]
  data["param"]["baseline"] = [-2.0, -0.5, -5]
  data["param"]["var_names"] = ['x1', 'x2', 'x3']
  data["param"]["lb"] = [-5, -5, -5]
  data["param"]["ub"] = [5, 5, 5]
  # data["param"]["LAMBDA"]= None
  # data["param"]["RHO"] = 1
  # data["param"]["h_max"] = 0
  data["meshType"] = "GMESH"
  # data["param"]["constraints_type"] = ["PB"]
  data["param"]["scaling"] = [10, 10, 10]
  data["param"]["post_dir"] = "./tests/bm/MOO/unconstrained/Kursawe/post"
  data["options"]["budget"] = 1000
  data["param"]["ref_point"] = [-14, 1]

  # POLL.main(data)
  # SEARCH.main(data)
  p_out, _ = POLL.main(data)
  s_out, _ = SEARCH.main(data)
  m_out, _ = MADS.main(data)
  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]
  
  toc = time.perf_counter()
  logger.info(f'Completed MO_Kursawe run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {0.64}\n poll step HV_obtained = {PHV}\n search step HV_expected = {0.5}\n search step HV_obtained = {SHV}\n MADS HV_expected = {0.45}\n MADS HV_obtained = {MHV}\n")
  if PHV < 0.64 or SHV < 0.5 or MHV < 0.45:
    logger.error("The MO_Kursawe QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_Kursawe QA test completed but failed.")
  else:
    logger.info("The MO_Kursawe QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")
  
def test_MO_Osyczka_Kundu():
  # COMPLETED: Investigate why starting from infeasible point does not work in MOO
  logger.info('\nStarted running MO_Osyczka_Kundu test... \n')
  tic = time.perf_counter()
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_Osyczka_Kundu
  data["param"]["name"] = "Osyczka_Kundu"
  data["param"]["baseline"] = [3, 2, 2, 0, 5, 10]
  # data["param"]["baseline"] = [5, 1, 5, 0, 5, 8]
  data["param"]["var_names"] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
  data["param"]["lb"] = [0,   0, 1, 0, 1,  0]
  data["param"]["ub"] = [10, 10, 5, 6, 5, 10]
  data["param"]["meshType"] = "GMESH"
  data["param"]["constraints_type"] = ["PB"]*6
  data["param"]["scaling"] = [10, 10, 4, 6, 4, 10]
  data["param"]["post_dir"] = "./tests/bm/MOO/constrained/Osyczka_Kundu/post"
  data["options"]["budget"] = 10000
  data["options"]["seed"] = 1234
  data["param"]["ref_point"] = [-50, 80]
  is_win = platform.platform().split('-')[0] == 'Windows'
  data["param"]["lhs_search_initialization"] = False if is_win else True

  data["search"]["ns"] = 50
  p_out, _ = POLL.main(data)
  s_out, _ = SEARCH.main(data)
  data["search"]["ns"] = 150
  m_out, _ = MADS.main(data)
  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]
  toc = time.perf_counter()
  logger.info(f'Completed MO_Osyczka_Kundu run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {1.3}\n poll step HV_obtained = {PHV}\n search step HV_expected = {1.0}\n search step HV_obtained = {SHV}\n MADS HV_expected = {1.8}\n MADS HV_obtained = {MHV}\n")
  if PHV < 1.3 or SHV < 1.0 or MHV < 1.15:
    logger.error("The MO_Osyczka_Kundu QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_Osyczka_Kundu QA test completed but failed.")
  else:
    logger.info("The MO_Osyczka_Kundu QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")
  
def test_MO_CTP1():
  logger.info('\nStarted running MO_CTP1 test... \n')
  tic = time.perf_counter()
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_CTP1
  data["param"]["name"] = "MO_CTP1"
  data["param"]["baseline"] = [0.5, 0.5]
  data["param"]["var_names"] = ['x1', 'x2']
  data["param"]["lb"] = [0, 0]
  data["param"]["ub"] = [1, 1]
  data["param"]["meshType"] = "GMESH"
  data["param"]["constraints_type"] = ["PB"]*2
  data["param"]["scaling"] = [1, 1]
  data["param"]["post_dir"] = "./tests/bm/MOO/constrained/MO_CTP1/post"
  data["options"]["budget"] = 500
  data["search"]["ns"] = 50
  data["param"]["ref_point"] = [1, 1]
  p_out, _ = POLL.main(data)
  s_out, _ = SEARCH.main(data)
  m_out, _ = MADS.main(data)
  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]
  
  toc = time.perf_counter()
  logger.info(f'Completed MO_CTP1 run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {0.6}\n poll step HV_obtained = {PHV}\n search step HV_expected = {0.6}\n search step HV_obtained = {SHV}\n MADS HV_expected = {0.65}\n MADS HV_obtained = {MHV}\n")
  if PHV < 0.6 or SHV < 0.6 or MHV < 0.65:
    logger.error("The MO_CTP1 QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_CTP1 QA test completed but failed.")
  else:
    logger.info("The MO_CTP1 QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")
  
def test_MO_Ex():
  logger.info('\nStarted running MO_Ex test... \n')
  tic = time.perf_counter()
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_Ex
  data["param"]["name"] = "Ex"
  data["param"]["baseline"] = [0.6, 2.5]
  data["param"]["var_names"] = ['x1', 'x2']
  data["param"]["lb"] = [0.1, 0]
  data["param"]["ub"] = [1, 5]
  data["param"]["meshType"] = "GMESH"
  data["param"]["constraints_type"] = ["PB"]*2
  data["param"]["scaling"] = [0.9, 5]
  data["param"]["post_dir"] = "./tests/bm/MOO/constrained/Ex/post"
  data["options"]["budget"] = 1000
  data["search"]["ns"] = 15
  data["param"]["ref_point"] = [1, 9]

  p_out, _ = POLL.main(data)
  s_out, _ = SEARCH.main(data)
  m_out, _ = MADS.main(data)
  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]
  
  toc = time.perf_counter()
  logger.info(f'Completed MO_Ex run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {1.2}\n poll step HV_obtained = {PHV}\n search step HV_expected = {0.8}\n search step HV_obtained = {SHV}\n MADS HV_expected = {0.8}\n MADS HV_obtained = {MHV}\n")
  if PHV < 1.2 or SHV < 0.8 or MHV < 0.8:
    logger.error("The MO_Ex QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_Ex QA test completed but failed.")
  else:
    logger.info("The MO_Ex QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")

def test_MO_ZDT1():
  logger.info('\nStarted running MO_ZDT1 test... \n')
  tic = time.perf_counter()
  d = 30
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_ZDT1
  data["param"]["name"] = "MO_ZDT1"
  np.random.seed(seed= 12345)
  data["param"]["baseline"] = np.random.rand(d)
  data["param"]["var_names"] = [f'x{i}' for i in range(d)]
  data["param"]["lb"] = [0]*d
  data["param"]["ub"] = [1]*d
  data["param"]["meshType"] = "GMESH"
  data["param"]["constraints_type"] = ["PB"]
  data["param"]["scaling"] = [1]*d
  data["param"]["post_dir"] = "./tests/bm/MOO/unconstrained/MO_ZDT1/post"
  data["options"]["budget"] = 10000
  data["param"]["ref_point"] = [1, 1]

  p_out, _ = POLL.main(data)
  data["options"]["budget"] = 500
  s_out, _ = SEARCH.main(data)
  data["options"]["budget"] = 10000
  m_out, _ = MADS.main(data)
  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]
  
  toc = time.perf_counter()
  logger.info(f'Completed MO_ZDT1 run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {0.62}\n poll step HV_obtained = {PHV}\n search step HV_expected = {0.7}\n search step HV_obtained = {SHV}\n MADS HV_expected = {0.62}\n MADS HV_obtained = {MHV}\n")
  if PHV < 0.62 or SHV < 0.7 or MHV < 0.62:
    logger.error("The MO_ZDT1 QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_ZDT1 QA test completed but failed.")
  else:
    logger.info("The MO_ZDT1 QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")

def test_MO_ZDT3():
  logger.info('\nStarted running MO_ZDT3 test... \n')
  tic = time.perf_counter()
  d = 30
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_ZDT3
  data["param"]["name"] = "MO_ZDT3"
  np.random.seed(seed= 12345)
  data["param"]["baseline"] = np.random.rand(d)
  data["param"]["var_names"] = [f'x{i}' for i in range(d)]
  data["param"]["lb"] = [0]*d
  data["param"]["ub"] = [1]*d
  data["param"]["meshType"] = "GMESH"
  data["param"]["constraints_type"] = ["PB"]
  data["param"]["scaling"] = [1]*d
  data["param"]["post_dir"] = "./tests/bm/MOO/unconstrained/MO_ZDT3/post"
  data["options"]["budget"] = 10000
  data["search"]["ns"] = 50
  data["param"]["ref_point"] = [1, 1]

  p_out, _ = POLL.main(data)
  s_out, _ = SEARCH.main(data)
  m_out, _ = MADS.main(data)
  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]
  
  toc = time.perf_counter()
  logger.info(f'Completed MO_ZDT3 run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {0.6}\n poll step HV_obtained = {PHV}\n search step HV_expected = {0.7}\n search step HV_obtained = {SHV}\n MADS HV_expected = {0.6}\n MADS HV_obtained = {MHV}\n")
  if PHV < 0.6 or SHV < 0.7 or MHV < 0.6:
    logger.error("The MO_ZDT3 QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_ZDT3 QA test completed but failed.")
  else:
    logger.info("The MO_ZDT3 QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")

def test_MO_ZDT4():
  logger.info('\nStarted running MO_ZDT4 test... \n')
  tic = time.perf_counter()
  d = 10
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_ZDT4
  data["param"]["name"] = "MO_ZDT4"
  np.random.seed(seed= 12345)
  data["param"]["baseline"] = np.random.rand(1).tolist() + np.random.uniform(low=-10, high=10, size=(d-1,)).tolist()
  data["param"]["var_names"] = [f'x{i}' for i in range(d)]
  data["param"]["lb"] = [0] + [-10]*(d-1)
  data["param"]["ub"] = [1] + [10]*(d-1)
  data["param"]["meshType"] = "GMESH"
  data["param"]["constraints_type"] = ["PB"]
  data["param"]["scaling"] = [1] + [20]*(d-1)
  data["param"]["post_dir"] = "./tests/bm/MOO/unconstrained/MO_ZDT4/post"
  data["options"]["budget"] = 2000 #40000
  data["search"]["ns"] = 55
  data["param"]["ref_point"] = [1, 1.2]
  data["param"]["lhs_search_initialization"] = True

  m_out, _ = MADS.main(data)
  MHV = m_out["HV"]
  
  toc = time.perf_counter()
  logger.info(f'Completed MO_ZDT4 run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n MADS step HV_expected = {0.8}\n MADS HV_obtained = {MHV}\n")
  if MHV < 0.8:
    logger.error("The MO_ZDT4 QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_ZDT4 QA test completed but failed.")
  else:
    logger.info("The MO_ZDT4 QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")

def test_MO_ZDT6():
  logger.info('\nStarted running MO_ZDT6 test... \n')
  tic = time.perf_counter()
  d = 10
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_ZDT6
  data["param"]["name"] = "MO_ZDT6"
  np.random.seed(seed= 12345)
  data["param"]["baseline"] = np.random.rand(d)
  data["param"]["var_names"] = [f'x{i}' for i in range(d)]
  data["param"]["lb"] = [0]*d
  data["param"]["ub"] = [1]*d
  
  data["param"]["constraints_type"] = ["PB"]
  data["param"]["scaling"] = [1]*d
  data["param"]["post_dir"] = "./tests/bm/MOO/unconstrained/MO_ZDT6/post"
  
  data["search"]["ns"] = 100
  data["param"]["ref_point"] = [1, 1.2]
  data["options"]["budget"] = 10000 #10000
  data["param"]["meshType"] = "GMESH"
  p_out, _ = POLL.main(data)
  data["options"]["budget"] = 500 #10000
  data["param"]["meshType"] = "OMESH"
  s_out, _ = SEARCH.main(data)
  m_out, _ = MADS.main(data)
  PHV = p_out["HV"]
  SHV = s_out["HV"]
  MHV = m_out["HV"]

  toc = time.perf_counter()
  logger.info(f'Completed MO_ZDT6 run in {toc - tic:.4f} seconds.\n')
  logger.info(f"Hypervolume indicators:\n poll step HV_expected = {0.4}\n poll step HV_obtained = {PHV}\n search step HV_expected = {0.1}\n search step HV_obtained = {SHV}\n MADS HV_expected = {0.9}\n MADS HV_obtained = {MHV}\n")
  if PHV < 0.4 or SHV < 0.1 or MHV < 0.9:
    logger.error("The MO_ZDT6 QA test failed: \n hypervolume indicators obtained does not pass the success criteria \n")
    raise IOError("The MO_ZDT6 QA test completed but failed.")
  else:
    logger.info("The MO_ZDT6 QA test successfully passed: \n hypervolume indicators obtained pass the success criteria \n")


if __name__ == "__main__":
  test_MO_Binh_and_Korn()
