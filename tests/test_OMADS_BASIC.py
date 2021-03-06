import pytest
from OMADS import DType, DefaultOptions, Parameters, Evaluator, Point, \
    OrthoMesh, Cache, Directions2n, PreMADS, Output, PostMADS, main

import copy
import os
import csv
from BMDFO import toy

import pandas as pd
import numpy as np


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
             "scaling": 10.0,
             "post_dir": "./post"}
    options = {"seed": 0, "budget": 100000, "tol": 1e-12, "display": True}

    data = {"evaluator": eval, "param": param, "options": options}

    out: Dict = main(data)
    print(out)


def test_omads_toy_quick():
    assert DType
    assert DefaultOptions
    assert Parameters
    assert Evaluator
    assert Point
    assert OrthoMesh
    assert Cache
    assert Directions2n
    assert PreMADS
    assert Output
    assert PostMADS
    assert main

    p_file_1 = os.path.abspath("./tests/bm/unconstrained/rosenbrock.json")
    main(p_file_1)

    p_file_2 = os.path.abspath("./tests/bm/constrained/geom_prog.json")
    main(p_file_2)

    p_file_3 = os.path.abspath("./tests/Rosen/param.json")
    main(p_file_3)

    data = {
        "evaluator":
            {
                "blackbox": "rosenbrock",
                "internal": "uncon",
                "path": os.path.abspath(".\\bm"),
                "input": "input.inp",
                "output": "output.out"},

        "param":
            {
                "baseline": [-2.0, -2.0],
                "lb": [-5, -5],
                "ub": [10, 10],
                "var_names": ["x1", "x2"],
                "scaling": 10.0,
                "post_dir": "./tests/bm/unconstrained/post"
            },

        "options":
            {
                "seed": 0,
                "budget": 100000,
                "tol": 1e-12,
                "psize_init": 1,
                "display": False,
                "opportunistic": False,
                "check_cache": True,
                "store_cache": True,
                "collect_y": False,
                "rich_direction": True,
                "precision": "high",
                "save_results": False,
                "save_coordinates": False,
                "save_all_best": False,
                "parallel_mode": False
            }
    }
    main(data)


def test_omads_toy_extended():
    uncon_test_names = ["ackley", "beale", "dixonprice", "griewank", "levy", "michalewicz", "perm", "powell",
                        "powersum", "rastrigin", "rosenbrock", "schwefel", "sphere", "trid", "zakharov"]

    con_test_names = ["g1", "g2", "g3", "geom_prog", "himmelblau", "pressure_vessel", "tc_spring",
                      "speed_reducer", "wbeam"]

    for name in uncon_test_names:
        main(os.path.abspath(os.path.join("./tests/bm/unconstrained", name + ".json")))

    for name in con_test_names:
        main(os.path.abspath(os.path.join("./tests/bm/constrained", name + ".json")))


def test_omads_toy_uncon_bm():
    p_files = []
    runs: int = 2
    # Remove existing BM log files (if any)
    file = os.path.abspath(os.path.join('./tests/bm/unconstrained/post', 'BM_report.csv'))
    if os.path.exists(file) and os.path.isfile(file):
        os.remove(file)

    df = pd.DataFrame(list())
    df.to_csv(file)

    bm: toy.Run = toy.Run(os.path.abspath('./tests/bm/unconstrained/post'))
    bm.test_suite = "uncon"
    bm_root = os.path.abspath('./tests/bm/unconstrained')

    # get BM parameters file names
    for p, _, filename in os.walk(bm_root):
        if p == bm_root:
            p_files = copy.deepcopy(filename)
    ms: bool
    sl: List[int] = []
    if runs > 1:
        sl = list(range(runs))
        ms = True
    else:
        ms = False
    for run in range(runs):
        for i in range(0, len(p_files)):
            try:
                _, file_exe = os.path.splitext(p_files[i])
                print(f"Solving {p_files[i]}: run# {run:.0f}: seed is {sl[run]:.0f}")
                if file_exe == '.json':
                    if ms:
                        main(os.path.join(bm_root, p_files[i]), bm, run, sl[run])
                    else:
                        main(os.path.join(bm_root, p_files[i]), bm, run)
            except RuntimeError:
                print("An error occured while running" + p_files[i])

    # Show box plot for the BM stats as an indicator
    # for measuring various algorithmic performance
    # bm.BM_statistics()


def test_omads_toy_con_bm():
    p_files = []
    runs: int = 2
    # Remove existing BM log files (if any)
    file = os.path.abspath(os.path.join('./tests/bm/constrained/post', 'BM_report.csv'))
    if os.path.exists(file) and os.path.isfile(file):
        os.remove(file)

    df = pd.DataFrame(list())
    df.to_csv(file)

    bm: toy.Run = toy.Run(os.path.abspath('./tests/bm/constrained/post'))
    bm.test_suite = "con"
    bm_root = os.path.abspath('./tests/bm/constrained')
    # get BM parameters file names
    for p, _, filename in os.walk(bm_root):
        if p == bm_root:
            p_files = copy.deepcopy(filename)

    ms: bool
    sl: List[int] = []
    if runs > 1:
        sl = list(range(runs))
        ms = True
    else:
        ms = False
    for run in range(runs):
        for i in range(0, len(p_files)):
            try:
                _, file_exe = os.path.splitext(p_files[i])
                if file_exe == '.json':
                    if ms:
                        main(os.path.join(bm_root, p_files[i]), bm, run, sl[run])
                    else:
                        main(os.path.join(bm_root, p_files[i]), bm, run)
            except RuntimeError:
                print("An error occured while running" + p_files[i])
    # Show box plot for the BM stats as an indicator
    # for measuring various algorithmic performance
    # bm.BM_statistics()
