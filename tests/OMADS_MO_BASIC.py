from OMADS import POLL, SEARCH, MADS
import copy
import os
import numpy as np

from typing import Dict, List
from multiprocessing import freeze_support
import platform

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
        "precision": "high",
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
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_Binh_and_Korn
  data["param"]["name"] = "Binh_and_Korn"
  data["param"]["baseline"] = [0, 0]
  data["param"]["lb"] = [0, 0]
  data["param"]["ub"] = [5, 3]
  data["param"]["constraints_type"] = ["PB", "PB"]
  data["param"]["scaling"] = [5, 3]
  data["param"]["post_dir"] = "./tests/bm/MOO/constrained/Binh_and_Korn/post"
  
  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_Chankong_and_Haimes():
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_Chankong_and_Haimes
  data["param"]["name"] = "Chankong_and_Haimes"
  data["param"]["baseline"] = [0, 0]
  data["param"]["lb"] = [-20, -20]
  data["param"]["ub"] = [20, 20]
  data["param"]["constraints_type"] = ["PB", "PB"]
  data["param"]["scaling"] = [40, 40]
  data["param"]["post_dir"] = "./tests/bm/MOO/constrained/Chankong_and_Haimes/post"
  
  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_Fonseca_Fleming():
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
  data["options"]["budget"] = 1000
  
  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_Test_function_4():
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
  data["options"]["budget"] = 1000
  
  # data["search"]["type"] = "VNS"
  # data["search"]["s_method"] = "RANDOM"
  # data["search"]["ns"] = 10

  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_Kursawe():
  # TODO: uncon logic needs review
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
  data["options"]["budget"] = 10000

  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_Osyczka_Kundu():
  # COMPLETED: Investigate why starting from infeasible point does not work in MOO
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
  data["options"]["budget"] = 30000
  data["options"]["seed"] = 1234

  # POLL.main(data)
  data["search"]["ns"] = 22
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_CTP1():
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
  data["options"]["budget"] = 3000
  data["search"]["ns"] = 50
  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_Ex():
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
  data["options"]["budget"] = 5000
  data["search"]["ns"] = 15
  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_ZDT1():
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
  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_ZDT3():
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
  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_ZDT4():
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
  data["options"]["budget"] = 5000 #40000
  data["search"]["ns"] = 55
  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

def test_MO_ZDT6():
  d = 10
  data = common_dict()
  data["evaluator"]["blackbox"] = MO_ZDT6
  data["param"]["name"] = "MO_ZDT6"
  np.random.seed(seed= 12345)
  data["param"]["baseline"] = np.random.rand(d)
  data["param"]["var_names"] = [f'x{i}' for i in range(d)]
  data["param"]["lb"] = [0]*d
  data["param"]["ub"] = [1]*d
  data["param"]["meshType"] = "OMESH"
  data["param"]["constraints_type"] = ["PB"]
  data["param"]["scaling"] = [1]*d
  data["param"]["post_dir"] = "./tests/bm/MOO/unconstrained/MO_ZDT6/post"
  data["options"]["budget"] = 10000
  data["search"]["ns"] = 100
  # POLL.main(data)
  # SEARCH.main(data)
  MADS.main(data)

if __name__ == "__main__":
  freeze_support()
