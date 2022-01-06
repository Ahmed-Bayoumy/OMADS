# OMADS
A python implementation for the mesh adaptive direct search (MADS) method; ORTHO-MADS algorithm

---

**Version 1.0.0**

MADS-poll step

---

## Contributors

---

## License & copyright

© Ahmed H. Bayoumy 
---
## How to use

After installing the libraries listed in the `requirements.txt`, `OMADS.py` can be called directly from a 
terminal window under the src directory. The path of the JSON template, which contains the problem input parameters, 
should be entered as an input argument to the `OMADS.py` call. 

```commandline
python OMADS.py ../examples/rosenbrock.json
```

Input parameters are provided in the `JASON` template using predefined attributes (keywords) under three dictionaries; 
evaluator, param, and options. Here is a brief description of each dictionary and its attributes.

* `evaluator`: in this dictionary we define the blackbox location and the name of input and output files (if exist)
  * `blackbox`: blackbox executable file name, or the function name if this is an internal function defined within the BM_suite
  * `internal`: the name of the testing category that holds your internal/external test function or blackbox evaluator
    * `con`: internal constrained single-objective function
    * `uncon`: internal unconctrained single-objective function
    * `exe`: external executable blackbox evaluator
  * `input`: the name of the input file (considered if external executable was defined)
  * `output`: the name of the output file (considered if external executable was defined)
---
* `param`: problem setup
  * `baseline`: this is the initial starting point (initial design vector)
  * `lb`: lower bounds vector
  * `ub`: uber bounds vector
  * `var_names`: list of design variables name
  * `scaling`: scaling factor
  * `post_dir`: the location of the post directory where results file shall be saved if requested
---
* `options`: algorithmic options
  * `seed`: the random generator seed that ensures results reproducibility. This should be an integer value
  * `budget`: the evaluation budget; maximum number of evaluations for the blackbox defined
  * `tol`: the minimum poll size tolerance; the algorithm terminates once the poll size falls below this value
  * `psize_init`: initial poll size
  * `display`: a boolean for displaying verbose outputs per iteration in the terminal window
  * `opportunistic`: a boolean for enabling opportunistic search
  * `check_cache`: a boolean for checking if the current point is a duplicate by checking its hashed address (integer signature)
  * `store_cache`: a boolean for saving evaluated designs in the cache memory
  * `collect_y`: currently inactive (to be used when the code is integrated with the PyADMM MDO module)
  * `rich_direction`: a boolean that enables capturing a rich set of directions in a generalized pattern
  * `precision`: a string character input that controls the `dtype` decimal resolution used by the numerical library `numpy`
    * `high`: `float128` 1e-18
    * `medium`: `float64` 1e-15
    * `low`: `float32` 1e-8
  * `save_results`: a boolean for generating a `MADS.csv` file for the output results under the post directory
  * `save_coordinates`: saving poll coordinates (spinners) of each iteration in a JASON dictionary template that can be used for visualization
  * `save_all_best`: a boolean for saving only incumbent solutions
  * `parallel_mode`: a boolean for parallel computation of the poll set
---
  
## Benchmarking

Two benchmarking (BM) suits are provided in the `BM_suite.py` code. The BM suits have different constrained and 
unconstrained optimization problems with various characteristics. You can run the BM by calling the following commands 
in the terminal window. 
```commandline
python OMADS.py bm uncon
python OMADS.py bm con
```

After the BM is finished, a `BM_report.csv` file will be generated in the post directory under 
the `examples` folder.
