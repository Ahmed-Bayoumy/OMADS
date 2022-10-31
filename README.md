# OMADS
MADS-poll step: A python implementation for the mesh adaptive direct search (MADS) method; ORTHO-MADS algorithm

---

**Version 1.5.0**

---

## License & copyright

© Ahmed H. Bayoumy 
---

## Citation

If you use this code, please cite it as below.

```pycon
   @software{OMADS_AB,
   author       = {Bayoumy, A.},
   title        = {OMADS},
   year         = 2022,
   publisher    = {Github},
   version      = {1.5.0},
   url          = {https://github.com/Ahmed-Bayoumy/OMADS}
   }
```

## How to use OMADS package

After installing the `OMADS` package from [PYPI](https://pypi.org/) website, the functions and classes of `OMADS` basic 
module can be imported directly to the python script as follows:

```pycon
from OMADS import *
```

## How to run OMADS from terminal
After installing the libraries listed in the `requirements.txt`, `OMADS/BASIC.py` can be called directly from a 
terminal window under the src directory. The path of the JSON template, which contains the problem input parameters, 
should be entered as an input argument to the `BASIC.py` call. 

```commandline
python ./OMADS/BASIC.py ../../tests/unconstrained/rosenbrock.json
```

## Input parameters
Input parameters are serialized in a `JSON` template using predefined attributes (keywords) under three dictionaries; 
`evaluator`, `param`, and `options`. Here is a brief description of each dictionary and its key attributes.

* `evaluator`: in this dictionary, we define the blackbox location and the name of input and output files (if exist)
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
  * `post_dir`: the location of the post directory where the results file shall be saved if requested
---
* `options`: algorithmic options
  * `seed`: the random generator seed that ensures results reproducibility. This should be an integer value
  * `budget`: the evaluation budget; the maximum number of evaluations for the blackbox defined
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

To benchmark `OMADS`, per se, you need to install the non-linear optimization benchmarking project `NOBM` from 
[PYPI](https://pypi.org/).  Two benchmarking suits are provided under the `BMDFO` package -- `BMDFO` stands for 
benchmarking derivative-free optimization algorithms.  The benchmarking suits have different constrained and 
unconstrained optimization problems with various characteristics.  The `BMDFO` modules can be imported directly 
to the python script as shown below: 
```pycon
from BMDFO import toy
```
For more details about the `NOBM` project and its use, check this [link](https://github.com/Ahmed-Bayoumy/NOBM). 
After running the benchmarking suite using various seed values, which are used to initialize the random number generator, 
a `BM_report.csv` file will be created in the post directory under the `examples` folder.

## Example

```pycon
import OMADS
import numpy as np

def rosen(x, *argv):
    x = np.asarray(x)
    y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
                axis=0), [0]]
    return y

eval = {"blackbox": rosen}
param = {"baseline": [-2.0, -2.0],
            "lb": [-5, -5],
            "ub": [10, 10],
            "var_names": ["x1", "x2"],
            "scaling": 10.0,
            "post_dir": "./post"}
options = {"seed": 0, "budget": 100000, "tol": 1e-12, "display": True}

data = {"evaluator": eval, "param": param, "options":options}

out = {}
# out is a dictionary that will hold output data of the final solution. The out dictionary has three keys: "xmin", "fmin" and "hmin"

out = OMADS.main(data)



```

### Results
```text
 --- Run Summary ---
 Run completed in 0.0303 seconds
 Random numbers generator's seed 0
 xmin = [1.0, 1.0]
 hmin = 1e-30
 fmin = 0.0
 #bb_eval = 185
 #iteration = 46
 nb_success = 4
 psize = 9.094947017729282e-13
 psize_success = 1.0
 psize_max = 2.0
```

