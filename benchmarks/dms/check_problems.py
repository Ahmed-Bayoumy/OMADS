"""Check the NumPy problems of ``problems.py`` against the original MATLAB files.

``problems.py`` was ported from the Julia versions of the collection. This
script checks it against an independent source: the MATLAB files of the same
collection written by Custodio et al. (DMultiMadsEB, problems/matlab). It does
not need MATLAB: the files use a small subset of the language (assignments,
indexing, ranges, matrix literals, for/if/while and about thirty elementwise
functions), which ``MatlabSubset`` below parses and evaluates with NumPy.

For each problem it compares
  - the objectives at the starting points and at random points of the box
    (relative difference, see --tol) with the MATLAB file,
  - the number of variables and objectives and the bounds with those of the
    DMulti-MADS experiments (scripts/generate_dmultimads_caches.jl; the
    MATLAB file headers disagree with them on a few problems).

Usage, from the repository root:

  python benchmarks/dms/check_problems.py --repo ../third_party/DMultiMadsEB
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from problems import PROBLEMS, start_points  # noqa: E402

DEFAULT_REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "third_party", "DMultiMadsEB"))


# --- Tokenizer ---------------------------------------------------------------

_TOKEN = re.compile(r"""
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?) |
    (?P<name>[A-Za-z_]\w*) |
    (?P<op>\.\*|\./|\.\^|\.'|==|~=|<=|>=|&&|\|\||[-+*/^'<>&|~=:;,()\[\]]) |
    (?P<nl>\n)
""", re.VERBOSE)
_KEYWORDS = {"for", "if", "elseif", "else", "while", "end", "function"}


class Tok:
  __slots__ = ("kind", "text", "space")

  def __init__(self, kind, text, space):
    self.kind, self.text, self.space = kind, text, space

  def __repr__(self):
    return f"{self.kind}:{self.text!r}"


def tokenize(src):
  src = re.sub(r"\.\.\.[^\n]*\n", " ", src)          # continuation lines
  toks, i, space = [], 0, True
  while i < len(src):
    c = src[i]
    if c in " \t\r":
      i, space = i + 1, True
      continue
    if c == "%":
      while i < len(src) and src[i] != "\n":
        i += 1
      continue
    m = _TOKEN.match(src, i)
    if not m:
      raise SyntaxError(f"unexpected {src[i:i + 20]!r}")
    kind = m.lastgroup
    text = m.group()
    if kind == "name" and text in _KEYWORDS:
      kind = "kw"
    toks.append(Tok(kind, text, space))
    i, space = m.end(), False
  toks.append(Tok("eof", "", True))
  return toks


# --- Parser (to nested tuples) -----------------------------------------------

_BINARY = {"||": 1, "&&": 2, "|": 3, "&": 4, "==": 5, "~=": 5, "<": 5, "<=": 5,
           ">": 5, ">=": 5, ":": 6, "+": 7, "-": 7, "*": 8, "/": 8, ".*": 8,
           "./": 8, "^": 10, ".^": 10}
_UNARY_PREC = 9


class Parser:
  def __init__(self, toks):
    self.toks, self.i = toks, 0
    self.brackets = 0          # inside [ ]: whitespace separates elements

  def peek(self, k=0):
    return self.toks[self.i + k]

  def take(self, text=None):
    t = self.toks[self.i]
    if text is not None and t.text != text:
      raise SyntaxError(f"expected {text!r}, got {t!r}")
    self.i += 1
    return t

  # statements
  def block(self, stop):
    stmts = []
    while True:
      t = self.peek()
      if t.kind == "eof" or (t.kind == "kw" and t.text in stop):
        return stmts
      if t.text in (";", ",") or t.kind == "nl":
        self.take()
        continue
      stmts.append(self.statement())

  def statement(self):
    t = self.peek()
    if t.kind == "kw" and t.text == "function":
      while self.peek().kind not in ("nl", "eof"):
        self.take()
      return ("nop",)
    if t.kind == "kw" and t.text == "for":
      self.take()
      var = self.take().text
      self.take("=")
      rng = self.expr()
      body = self.block({"end"})
      self.take("end")
      return ("for", var, rng, body)
    if t.kind == "kw" and t.text == "while":
      self.take()
      cond = self.expr()
      body = self.block({"end"})
      self.take("end")
      return ("while", cond, body)
    if t.kind == "kw" and t.text == "if":
      self.take()
      branches = [(self.expr(), self.block({"elseif", "else", "end"}))]
      other = []
      while True:
        k = self.take().text
        if k == "elseif":
          branches.append((self.expr(), self.block({"elseif", "else", "end"})))
        elif k == "else":
          other = self.block({"end"})
        else:
          return ("if", branches, other)
    lhs = self.expr()
    if self.peek().text == "=":
      self.take()
      return ("assign", lhs, self.expr())
    return ("expr", lhs)

  # expressions
  def expr(self, prec=0):
    left = self.unary()
    while True:
      t = self.peek()
      p = _BINARY.get(t.text) if t.kind == "op" else None
      if p is None or p <= prec:
        return left
      if self.brackets and t.text in "+-" and t.space and not self.peek(1).space:
        return left           # "[1 -2]": a new element
      self.take()
      if t.text == ":":
        right = self.expr(p)
        if self.peek().text == ":":           # a:step:b
          self.take()
          left = ("range", left, right, self.expr(p))
        else:
          left = ("range", left, None, right)
        continue
      # ^ is left-associative in MATLAB; its right operand may be unary.
      right = self.unary() if p == 10 else self.expr(p)
      left = ("bin", t.text, left, right)

  def unary(self):
    t = self.peek()
    if t.kind == "op" and t.text in ("-", "+", "~"):
      self.take()
      operand = self.expr(_UNARY_PREC)
      return ("un", t.text, operand)
    return self.postfix()

  def postfix(self):
    node = self.primary()
    while True:
      t = self.peek()
      if t.text == "(" and not (self.brackets and t.space):
        self.take()
        args = self.arguments()
        node = ("call", node, args)
      elif t.text in ("'", ".'") and not t.space:
        self.take()
        node = ("tr", node)
      else:
        return node

  def arguments(self):
    saved, self.brackets = self.brackets, 0
    args = []
    while self.peek().text != ")":
      if self.peek().text == ":" and self.peek(1).text in (",", ")"):
        self.take()
        args.append(("colon",))
      else:
        args.append(self.expr())
      if self.peek().text == ",":
        self.take()
    self.take(")")
    self.brackets = saved
    return args

  def primary(self):
    t = self.take()
    if t.kind == "num":
      return ("num", float(t.text))
    if t.kind == "name":
      return ("name", t.text)
    if t.kind == "kw" and t.text == "end":
      return ("end",)
    if t.text == "(":
      saved, self.brackets = self.brackets, 0
      e = self.expr()
      self.take(")")
      self.brackets = saved
      return ("paren", e)
    if t.text == "[":
      return self.matrix()
    raise SyntaxError(f"unexpected {t!r}")

  def matrix(self):
    self.brackets += 1
    rows, row = [], []
    while True:
      t = self.peek()
      if t.text == "]":
        self.take()
        break
      if t.text == ";" or t.kind == "nl":
        self.take()
        if row:
          rows.append(row)
        row = []
        continue
      if t.text == ",":
        self.take()
        continue
      row.append(self.expr())
    if row:
      rows.append(row)
    self.brackets -= 1
    return ("matrix", rows)


# --- Evaluator ---------------------------------------------------------------

def _m(v):
  """Every value is a 2-D float array, as in MATLAB."""
  a = np.asarray(v, dtype=float)
  if a.ndim == 0:
    return a.reshape(1, 1)
  if a.ndim == 1:
    return a.reshape(1, -1)
  return a


def _isvec(a):
  return a.shape[0] == 1 or a.shape[1] == 1


def _reduce(fn):
  def f(a, *rest):
    if rest:
      raise NotImplementedError("dimension argument")
    return _m(fn(a.ravel())) if _isvec(a) else _m(fn(a, axis=0))
  return f


def _minmax(red, elem):
  def f(a, b=None, *rest):
    if rest:
      raise NotImplementedError("dimension argument")
    if b is None:
      return _reduce(red)(a)
    return _m(elem(a, b))
  return f


def _setdiff(a, b):
  r = np.setdiff1d(a.ravel(), b.ravel())
  return _m(r).T if a.shape[1] == 1 and a.shape[0] > 1 else _m(r)


def _ones_like(fill):
  def f(*dims):
    d = [int(x.item()) for x in dims]
    return _m(np.full((d[0], d[0]) if len(d) == 1 else tuple(d[:2]), fill))
  return f


BUILTINS = {
    "ones": _ones_like(1.0), "zeros": _ones_like(0.0),
    "sum": _reduce(np.sum), "prod": _reduce(np.prod),
    "max": _minmax(np.max, np.maximum), "min": _minmax(np.min, np.minimum),
    "abs": lambda a: _m(np.abs(a)), "sin": lambda a: _m(np.sin(a)),
    "cos": lambda a: _m(np.cos(a)), "tan": lambda a: _m(np.tan(a)),
    "exp": lambda a: _m(np.exp(a)), "log": lambda a: _m(np.log(a)),
    "sqrt": lambda a: _m(np.sqrt(a)), "floor": lambda a: _m(np.floor(a)),
    "ceil": lambda a: _m(np.ceil(a)), "round": lambda a: _m(np.round(a)),
    "sign": lambda a: _m(np.sign(a)), "atan": lambda a: _m(np.arctan(a)),
    "mod": lambda a, b: _m(np.mod(a, b)),
    "numel": lambda a: _m(a.size), "length": lambda a: _m(max(a.shape)),
    "setdiff": _setdiff,
}
CONSTANTS = {"pi": np.pi, "Inf": np.inf, "inf": np.inf, "NaN": np.nan, "eps": np.finfo(float).eps}


def _index_vector(arg, extent):
  if isinstance(arg, str):                     # ':'
    return np.arange(extent)
  a = arg
  if a.size and np.all(np.isin(a, (0.0, 1.0))) and getattr(a, "_logical", False):
    return np.flatnonzero(a.ravel(order="F"))
  return a.ravel(order="F").astype(int) - 1


class Logical(np.ndarray):
  _logical = True


def _logical(a):
  return _m(a).astype(float).view(Logical)


class MatlabSubset:
  def __init__(self, src):
    p = Parser(tokenize(src))
    self.prog = p.block(set())
    if p.peek().kind != "eof":
      raise SyntaxError(f"trailing {p.peek()!r}")

  def __call__(self, x, out="F", env=None, raw=False):
    env = dict(env or {})
    env["x"] = _m(np.asarray(x, dtype=float)).reshape(-1, 1)
    self.run(self.prog, env)
    return env[out] if raw else env[out].ravel(order="F")

  # statements
  def run(self, stmts, env):  # noqa: C901
    for s in stmts:
      kind = s[0]
      if kind == "assign":
        self.assign(s[1], self.ev(s[2], env), env)
      elif kind == "for":
        rng = self.ev(s[2], env)
        for j in range(rng.shape[1]):
          env[s[1]] = _m(rng[:, j:j + 1])
          self.run(s[3], env)
      elif kind == "while":
        while self.truth(self.ev(s[1], env)):
          self.run(s[2], env)
      elif kind == "if":
        for cond, body in s[1]:
          if self.truth(self.ev(cond, env)):
            self.run(body, env)
            break
        else:
          self.run(s[2], env)
      elif kind == "expr":
        self.ev(s[1], env)

  @staticmethod
  def truth(a):
    return a.size > 0 and bool(np.all(a != 0))

  def assign(self, lhs, val, env):
    if lhs[0] == "name":
      env[lhs[1]] = _m(val)
      return
    if lhs[0] != "call" or lhs[1][0] != "name":
      raise SyntaxError(f"cannot assign to {lhs}")
    name = lhs[1][1]
    A = env.get(name, np.zeros((0, 0)))
    args = [self.ev_index(a, env, A, k, len(lhs[2])) for k, a in enumerate(lhs[2])]
    val = np.asarray(val, dtype=float)
    if len(args) == 1:
      idx = _index_vector(args[0], A.size)
      need = int(idx.max()) + 1 if idx.size else 0
      if need > A.size:                      # grow along the vector's direction
        if A.size == 0:
          B = np.zeros((1, need))
        elif A.shape[0] == 1:
          B = np.zeros((1, need))
        elif A.shape[1] == 1:
          B = np.zeros((need, 1))
        else:
          raise IndexError("linear growth of a matrix")
        B.ravel(order="F")[:A.size] = A.ravel(order="F")
        A = B
      A = np.array(A, dtype=float)
      flat = A.ravel(order="F").copy()
      flat[idx] = val.ravel(order="F") if val.size > 1 else val.item()
      env[name] = flat.reshape(A.shape, order="F")
    else:
      r = _index_vector(args[0], A.shape[0])
      c = _index_vector(args[1], A.shape[1])
      shape = (max(A.shape[0], int(r.max()) + 1 if r.size else 0),
               max(A.shape[1], int(c.max()) + 1 if c.size else 0))
      B = np.zeros(shape)
      B[:A.shape[0], :A.shape[1]] = A
      B[np.ix_(r, c)] = val.reshape(len(r), len(c), order="F") if val.size > 1 else val.item()
      env[name] = B

  def ev_index(self, node, env, A, k, nargs):
    if node[0] == "colon":
      return ":"
    if nargs == 1:
      end = A.size
    else:
      end = A.shape[k] if k < 2 else 1
    self._end.append(end)
    try:
      return self.ev(node, env)
    finally:
      self._end.pop()

  _end = []

  # expressions
  def ev(self, n, env):  # noqa: C901
    kind = n[0]
    if kind == "num":
      return _m(n[1])
    if kind == "name":
      if n[1] in env:
        return env[n[1]]
      if n[1] in CONSTANTS:
        return _m(CONSTANTS[n[1]])
      if n[1] in BUILTINS:
        return BUILTINS[n[1]]()
      raise NameError(n[1])
    if kind == "end":
      return _m(self._end[-1])
    if kind == "paren":
      return self.ev(n[1], env)
    if kind == "tr":
      return self.ev(n[1], env).T
    if kind == "un":
      a = self.ev(n[2], env)
      return {"-": lambda: -a, "+": lambda: a, "~": lambda: _logical(a == 0)}[n[1]]()
    if kind == "range":
      lo = self.ev(n[1], env).item()
      step = self.ev(n[2], env).item() if n[2] is not None else 1.0
      hi = self.ev(n[3], env).item()
      count = int(np.floor((hi - lo) / step + 1e-10)) + 1
      return _m(lo + step * np.arange(max(count, 0)))
    if kind == "matrix":
      rows = [np.hstack([self.ev(e, env) for e in row]) for row in n[1]]
      return _m(np.vstack(rows)) if rows else np.zeros((0, 0))
    if kind == "call":
      f = n[1]
      if f[0] == "name" and f[1] in env:
        A = env[f[1]]
        args = [self.ev_index(a, env, A, k, len(n[2])) for k, a in enumerate(n[2])]
        if len(args) == 1:
          idx = _index_vector(args[0], A.size)
          r = A.ravel(order="F")[idx]
          if isinstance(args[0], str):
            return _m(r).T
          if _isvec(A) and _isvec(args[0]):
            return _m(r) if A.shape[0] == 1 else _m(r).T
          return r.reshape(args[0].shape, order="F")
        r = _index_vector(args[0], A.shape[0])
        c = _index_vector(args[1], A.shape[1])
        return _m(A[np.ix_(r, c)])
      if f[0] == "name" and f[1] in BUILTINS:
        return BUILTINS[f[1]](*[self.ev(a, env) for a in n[2]])
      raise NameError(f"{f}")
    if kind == "bin":
      op, a, b = n[1], self.ev(n[2], env), self.ev(n[3], env)
      if op == "+":
        return a + b
      if op == "-":
        return a - b
      if op == ".*":
        return a * b
      if op == "./":
        return a / b
      if op == ".^":
        return _m(np.power(a, b))
      if op == "*":
        return a * b if a.size == 1 or b.size == 1 else a @ b
      if op == "/":
        if b.size == 1:
          return a / b
        raise NotImplementedError("matrix right division")
      if op == "^":
        if a.size == 1 and b.size == 1:
          return _m(np.power(a, b))
        raise NotImplementedError("matrix power")
      cmp = {"==": np.equal, "~=": np.not_equal, "<": np.less, "<=": np.less_equal,
             ">": np.greater, ">=": np.greater_equal}
      if op in cmp:
        return _logical(cmp[op](a, b))
      if op in ("&", "&&"):
        return _logical((a != 0) & (b != 0))
      if op in ("|", "||"):
        return _logical((a != 0) | (b != 0))
    raise NotImplementedError(kind)


# --- Header and comparison ---------------------------------------------------

def _split_top(text):
  """Split at the commas outside brackets."""
  parts, depth, cur = [], 0, ""
  for c in text:
    depth += c in "([" and 1 or (c in ")]" and -1 or 0)
    if c == "," and depth == 0:
      parts.append(cur)
      cur = ""
    else:
      cur += c
  return parts + [cur]


def _julia_to_matlab(e):
  e = re.sub(r"\bcopy\(", "(", e)
  e = re.sub(r"\b(zeros|ones)\(([^(),]+)\)", r"\1(\2, 1)", e)
  e = re.sub(r"\[\s*([^\[\]]+?)\s*\*\s*i\s+for\s+i\s*=\s*([^\]]+)\]", r"(\1 * (\2))'", e)
  return e


def experiment_problems(path):
  """{name: (n, m, lb, ub)} of the DMulti-MADS experiment script, whose
  definitions are simple Julia assignments evaluated here as MATLAB."""
  env, out = {}, {}
  for line in open(path, encoding="utf-8"):
    line = line.split("#")[0].strip()
    m = re.match(r'dict_problems\["(\w+)"\]\s*=\s*\((.*)\)\s*$', line)
    if m:
      n, nobj, lb, ub = (MatlabSubset("v = " + _julia_to_matlab(e) + ";")(
          np.zeros(1), out="v", env=env) for e in _split_top(m.group(2)))
      out[m.group(1)] = (int(n.item()), int(nobj.item()), lb, ub)
      continue
    m = re.match(r"(\w+)\s*=\s*(.+)$", line)
    if m and "dict_problems" not in line and "Dict" not in line:
      try:
        env[m.group(1)] = MatlabSubset("v = " + _julia_to_matlab(m.group(2)) + ";")(
            np.zeros(1), out="v", env=env, raw=True)
      except Exception:  # noqa: BLE001 - other Julia statements of the script
        pass
  return out


def check(name, path, setup, npts, tol, rng):
  src = open(path, encoding="latin-1").read()
  f, n, m, lb, ub = PROBLEMS[name]
  notes = []
  if setup is None:
    notes.append("not in the experiment script")
  else:
    en, em, elb, eub = setup
    if (en, em) != (n, m):
      notes.append(f"n, m = {n}, {m} vs {en}, {em} in the experiments")
    for label, ours, theirs in (("lb", lb, elb), ("ub", ub, eub)):
      if theirs.size != n or not np.allclose(theirs.ravel(), ours, rtol=1e-12, atol=0):
        notes.append(f"{label} differs from the experiments")
  g = MatlabSubset(src)
  X = np.vstack([np.atleast_2d(start_points(name)).reshape(-1, n),
                 lb + rng.random((npts, n)) * (np.asarray(ub) - np.asarray(lb))])
  worst = 0.0
  for x in X:
    a = np.asarray(f(x), dtype=float)
    b = g(x)
    if a.shape != b.shape:
      return f"FAIL {len(a)} objectives vs {len(b)}", notes
    d = np.abs(a - b) / np.maximum(1.0, np.abs(b))
    d[np.isnan(a) & np.isnan(b)] = 0.0
    worst = max(worst, float(np.nanmax(d)) if not np.isnan(d).all() else np.inf)
  status = "ok" if worst <= tol and not notes else "FAIL"
  return f"{status} max rel diff {worst:.1e} over {len(X)} points", notes


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--repo", default=DEFAULT_REPO,
                      help="clone of github.com/bbopt/DMultiMadsEB (default: %(default)s)")
  parser.add_argument("--points", type=int, default=200, help="random points per problem")
  parser.add_argument("--tol", type=float, default=1e-9, help="relative tolerance")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  rng = np.random.default_rng(args.seed)
  setups = experiment_problems(os.path.join(args.repo, "scripts", "generate_dmultimads_caches.jl"))
  failed = 0
  for name in PROBLEMS:
    path = os.path.join(args.repo, "problems", "matlab", name + ".m")
    if not os.path.exists(path):
      print(f"{name:9s} no MATLAB file")
      failed += 1
      continue
    try:
      result, notes = check(name, path, setups.get(name), args.points, args.tol, rng)
    except Exception as exc:  # noqa: BLE001 - report and continue
      result, notes = f"ERROR {exc!r}", []
    failed += not result.startswith("ok")
    print(f"{name:9s} {result}" + (f"  [{'; '.join(notes)}]" if notes else ""))
  print(f"{len(PROBLEMS) - failed}/{len(PROBLEMS)} problems match the MATLAB files and the experiment setup")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
