import copy
import json
from multiprocessing import freeze_support
import os
import sys
import OMADS.POLL as PS
import OMADS.SEARCH as SS
from typing import List, Dict, Any, Callable, Protocol, Optional
import numpy as np

def load_file(*args) -> dict:
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

def main(*args) -> Dict[str, Any]:
    datas = load_file(*args)
    datap = datas
    stop = False
    xsmin: SS = SS
    xpmin: PS = PS
    iter = 0
    total_budget = datas["options"]["budget"]
    tol = datas["options"]["tol"]
    nevals = 0
    poll = None
    while True:
      if datas["options"]["display"] == True:
        print(f'iter# = {iter}. Search step')
      if iter > 0:
        search.mesh.msize = pout["msize"]
        poll.mesh.psize = sout["psize"]
        datas["options"]["extend"] = search
        datap["options"]["extend"] = poll
        ps = search.hashtable._cache_dict
        pd = poll.hashtable._cache_dict

        [search.hashtable.hash_id.append(pp) for pp in pd]
        [poll.hashtable.hash_id.append(pp) for pp in ps]
        search.hashtable._cache_dict.update(pd)
        poll.hashtable._cache_dict.update(ps)
        
      if iter == 0 or not poll.success:
        sout, search = xsmin.main(datas)
      datap["options"]["psize_init"] = datas["options"]["psize_init"] = sout["psize"]
      datap["param"]["baseline"] = datas["param"]["baseline"] = sout["xmin"]
      if datas["options"]["display"] == True:
        print(f'Finished search step. Best found at xmin={sout["xmin"]} and msize={sout["msize"]}')
        print(f"iter# = {iter}. Poll Step")
      datap["options"]["budget"] -=  sout["nbb_evals"]
      # Update the avialable budget
      # datap["options"]["budget"] -= sout["nbb_evals"]
      # datas["options"]["budget"] -= sout["nbb_evals"]
      c = 0
      if "var_sets" in datap["param"].keys() and datap["param"]["var_sets"] != None:
        for t in datap["param"]["var_type"]:
          if t[0].lower() == "c" or t[0].lower() == "d":
            datap["param"]["baseline"][c] = datap["param"]["var_sets"][t.split("_")[1]].index(sout["xmin"][c])
          c += 1
      pout, poll = xpmin.main(datap)
      # Update the avialable budget
      # datas["options"]["budget"] -= pout["nbb_evals"]
      # datap["options"]["budget"] -= pout["nbb_evals"]
      datas["options"]["psize_init"] = datap["options"]["psize_init"] = pout["psize"]
      datas["param"]["baseline"] = datap["param"]["baseline"] = pout["xmin"]
      if datas["options"]["display"] == True:
        print(f'Finished poll step. Best found at xmin={pout["xmin"]} and msize={pout["msize"]}')

      nevals += sout["nbb_evals"] + pout["nbb_evals"]
      stop = abs(pout["msize"]) < tol or abs(pout["psize"]) < tol or nevals >= total_budget
      iter += 1
      if stop == True:
        if datas["options"]["display"] == True:
          print(f'Best solution found at xmin={pout["xmin"]} and msize={pout["msize"]}  using {nevals} evaluations')
        break
    
    return pout, sout
    


def rosen(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
        axis=0), [0]]
  return y


def test_omads_callable_quick():
  eval = {"blackbox": rosen}
  param = {"baseline": [-2.0, -2.0],
       "lb": [-5, -5],
       "ub": [10, 10],
       "var_names": ["x1", "x2"],
       "scaling": 15.0,
       "post_dir": "./post",
       "Failure_stop": True}
  sampling = {
    "method": 2,
    "ns": 5,
    "visualize": False
  }
  options = {"seed": 0, "budget": 100000, "tol": 1e-12, "display": True, "check_cache": True, "store_cache": True, "rich_direction": True, "psize_init": 1., "precision": "high"}

  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling}

  out: Dict = main(data)
  print(out)


if __name__ == "__main__":
    freeze_support()
    p_file: str = os.path.abspath("")

    """ Check if an input argument is provided"""
    if len(sys.argv) > 1:
      p_file = os.path.abspath(sys.argv[1])
      main(p_file)

    if (p_file != "" and os.path.exists(p_file)):
      main(p_file)

    if p_file == "":
      raise IOError("Undefined input args."
              " Please specify an appropriate input (parameters) jason file")