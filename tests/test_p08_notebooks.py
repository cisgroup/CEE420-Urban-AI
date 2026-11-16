"""The P08 notebooks keep the shape of the P07 kit, with two guards for the deep learning libraries.

Students run and read these notebooks; they do not write code. So every code cell is short and calls the toolbox
`graph_tools.py` instead of defining anything, every text cell is short and plain, each notebook has exactly one
"change me" value and no blanks to fill, and nothing reaches outside the kit's folder. A cell that trains a model sits
under `if torch_ok:`, a cell that uses PyTorch Geometric under `if torch_ok and pyg_ok:`, each with the one printed line
in its `else`, so the kit runs to the end without either library; no notebook imports torch, torch_geometric or
networkx itself. The toolbox stays readable and self-contained, and no token or key reaches the kit.
"""

import ast
import json
import re
from importlib.util import module_from_spec, spec_from_file_location

import pytest

from conftest import P08_KIT

NOTEBOOKS = sorted(p for p in P08_KIT.glob("08_*.ipynb") if "-mywork" not in p.name)
IDS = [p.stem for p in NOTEBOOKS]
EXPECTED = ["08_01_message_passing", "08_02_a_gcn_on_streets", "08_03_does_the_graph_help", "08_04_traffic_on_a_sensor_graph"]
MODULE = P08_KIT / "graph_tools.py"
ALLOWED_IMPORTS = {"json", "math", "warnings", "numpy", "matplotlib", "geopandas", "torch", "torch_geometric"}
MAX_CODE_LINES = 12
MAX_GUARDED_LINES = 15           # a guard and its else around a twelve-line body, or the your-turn cell
MAX_LINE_CHARS = 110
MAX_TOOLBOX_LINE_CHARS = 150
MAX_MD_WORDS = 120
MARKER = "# change me:"
TORCH_GUARD = re.compile(r"^if torch_ok:\s*$", re.M)
PYG_GUARD = re.compile(r"^if torch_ok and pyg_ok:\s*$", re.M)
TORCH_ELSE = re.compile(r"^else:\n\s+print\(gt\.NO_TORCH\)\s*$", re.M)
PYG_ELSE = re.compile(r"^else:\n\s+print\(gt\.NO_TORCH if not torch_ok else gt\.NO_PYG\)\s*$", re.M)
TORCH_CALLS = ("gt.train(", "gt.fit(", "gt.run_draws(", "gt.GCN(", "gt.MLP(", "gt.count_parameters(", "gt.GraphConvolution(")
PYG_CALLS = ("gt.pyg_gcn(", "gt.copy_weights(")
SECRETS = re.compile(r"hf_[A-Za-z0-9]{20,}|HF_TOKEN=\S|MLY\|[0-9]+\||CENSUS_API_KEY=\S", re.I)


def cells(notebook, kind):
    for index, cell in enumerate(json.loads(notebook.read_text())["cells"]):
        if cell["cell_type"] == kind:
            yield index, "".join(cell["source"])


def code_lines(text):
    return [line for line in text.splitlines() if line.strip()]


def guarded(text):
    return bool(TORCH_GUARD.search(text) or PYG_GUARD.search(text))


def test_the_four_notebooks_carry_plain_names():
    assert IDS == EXPECTED, f"the kit holds {IDS}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_no_notebook_defines_code_or_chains_statements(notebook):
    """Students call the toolbox; a notebook never defines a function or a class, and one line holds one statement."""
    for index, text in cells(notebook, "code"):
        tree = ast.parse(text)
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)), (
                f"{notebook.name} cell {index} defines a function, a class or a lambda")
        starts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.stmt)]
        assert len(starts) == len(set(starts)), f"{notebook.name} cell {index} puts two statements on one line"
        for line in code_lines(text):
            assert len(line) <= MAX_LINE_CHARS, f"{notebook.name} cell {index}: a line of {len(line)} characters"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_every_cell_is_short(notebook):
    """At most twelve lines of code per cell (the setup cell excepted, a guarded cell and the your-turn cell get three
    more), at most 120 words per text cell (the Further resources cell excepted)."""
    code = list(cells(notebook, "code"))
    for index, text in code[1:]:
        limit = MAX_GUARDED_LINES if guarded(text) or MARKER in text else MAX_CODE_LINES
        n = len(code_lines(text))
        assert n <= limit, f"{notebook.name} cell {index} has {n} code lines"
    for index, text in cells(notebook, "markdown"):
        if text.lstrip().startswith("## Further resources"):
            continue
        words = len(re.sub(r"http\S+", "", text).split())
        assert words <= MAX_MD_WORDS, f"{notebook.name} cell {index} has {words} words"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_kit_runs_without_the_libraries(notebook):
    """A cell that trains sits under `if torch_ok:`, a cell that uses PyTorch Geometric under `if torch_ok and pyg_ok:`,
    and each guard is followed by the else that prints the one line saying the cell was skipped."""
    for index, text in cells(notebook, "code"):
        assert not re.search(r"^\s*(import|from)\s+(torch|torch_geometric|networkx)\b", text, re.M), (
            f"{notebook.name} cell {index} imports a library the toolbox wraps")
        if TORCH_GUARD.search(text):
            assert TORCH_ELSE.search(text), f"{notebook.name} cell {index}: `if torch_ok:` lacks its else line"
        if PYG_GUARD.search(text):
            assert PYG_ELSE.search(text), f"{notebook.name} cell {index}: `if torch_ok and pyg_ok:` lacks its else line"
        if any(call in text for call in PYG_CALLS):
            assert PYG_GUARD.search(text), f"{notebook.name} cell {index} uses PyTorch Geometric without its guard"
        elif any(call in text for call in TORCH_CALLS):
            assert guarded(text), f"{notebook.name} cell {index} trains a model without the guard"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_notebook_uses_the_toolbox_and_one_change_me(notebook):
    code = [text for _, text in cells(notebook, "code")]
    assert any("import graph_tools as gt" in text for text in code), f"{notebook.name} does not import the toolbox"
    marked = [text for text in code if MARKER in text]
    assert len(marked) == 1, f"{notebook.name} has {len(marked)} cells with a change-me value, one is required"
    assert sum(line.count(MARKER) for line in marked[0].splitlines()) == 1
    assert re.search(r"^\s*[A-Z_][A-Z0-9_]*\s*=.*" + re.escape(MARKER), marked[0], re.M), f"{notebook.name}: the marked line does not set a named constant"
    for text in code:
        assert "___" not in text, f"{notebook.name} still has a blank to fill"
        assert not SECRETS.search(text), f"{notebook.name} carries something that looks like a token"
    tags = [tag for cell in json.loads(notebook.read_text())["cells"] for tag in cell.get("metadata", {}).get("tags", [])]
    assert "break-glass" not in tags, f"{notebook.name} still carries an appendix"
    headings = [text.splitlines()[0] for _, text in cells(notebook, "markdown") if text.startswith("## ")]
    for required in ("## Your turn", "## Check your understanding", "## Where we are", "## Further resources"):
        assert required in headings, f"{notebook.name} lacks {required}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_every_data_path_in_code_exists(notebook):
    """The loaders hide the open() call from the style test, so every data path literal is checked here."""
    for index, text in cells(notebook, "code"):
        for quoted in re.findall(r"""["'](data/[^"'\n{]+)["']""", text):
            assert (notebook.parent / quoted).exists(), f"{notebook.name} cell {index} names {quoted}, which is not there"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_credits_are_in_the_prose(notebook):
    """OpenStreetMap's credit appears wherever the street segments are read, AlphaEarth's where their embedding is,
    METR-LA's and Caltrans PeMS's where the speeds or the sensors are, Fey and Lenssen's where PyTorch Geometric runs."""
    code = "\n".join(text for _, text in cells(notebook, "code"))
    prose = "\n".join(text for _, text in cells(notebook, "markdown"))
    assert "CC BY 4.0" in prose, f"{notebook.name} lacks the kit's own licence"
    if "streets." in code:
        assert "OpenStreetMap" in prose and "ODbL" in prose, f"{notebook.name} uses OpenStreetMap data without crediting it"
    if "streets.npz" in code:
        assert "AlphaEarth" in prose and "Google" in prose, f"{notebook.name} uses the embedding without crediting it"
    if "metr_" in code:
        assert "METR-LA" in prose and "Caltrans PeMS" in prose and "Li" in prose, f"{notebook.name} uses METR-LA without its source"
    if "gt.pyg_gcn(" in code or PYG_GUARD.search(code):                 # pyg_ok alone (the setup cell) uses nothing of the library
        assert "Fey and Lenssen" in prose, f"{notebook.name} uses PyTorch Geometric without crediting it"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_outputs_are_clean(notebook):
    """An executed notebook carries no error, nothing on stderr, no warning text and no @ anywhere."""
    text = notebook.read_text()
    assert "@" not in text, f"{notebook.name} carries an @"
    for cell in json.loads(text)["cells"]:
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", f"{notebook.name} was saved with an error"
            assert out.get("name") != "stderr", f"{notebook.name} prints to stderr: {''.join(out.get('text', ''))[:120]}"
            body = "".join(out.get("text", "")) + "".join(out.get("data", {}).get("text/plain", ""))
            assert "Warning" not in body, f"{notebook.name} prints a warning: {body[:120]}"


def test_the_toolbox_is_readable_and_self_contained():
    assert MODULE.exists(), "the kit ships graph_tools.py"
    source = MODULE.read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= ALLOWED_IMPORTS, f"graph_tools imports {sorted(imported - ALLOWED_IMPORTS)}"
    assert "@" not in source, "graph_tools.py carries an @ (use np.dot or torch.matmul)"
    for forbidden in ("/Users", "census.env", "MLY|", "hf_"):
        assert forbidden not in source, f"graph_tools.py mentions {forbidden}"
    for number, line in enumerate(source.splitlines(), 1):
        assert len(line) <= MAX_TOOLBOX_LINE_CHARS, f"graph_tools.py line {number} is {len(line)} characters"
    spec = spec_from_file_location("graph_tools_under_test", MODULE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__doc__ and len(module.__all__) >= 60
    for name in module.__all__:
        member = getattr(module, name)
        if name.isupper() or name == "torch_ok":
            continue                                   # a documented constant, such as the line printed without PyTorch
        assert callable(member), f"{name} is not callable"
        assert member.__doc__, f"{name} has no docstring"
    assert isinstance(module.NO_TORCH, str) and "PyTorch" in module.NO_TORCH and "PyTorch Geometric" in module.NO_PYG
    assert module.CRS == "EPSG:26918" and module.SEED == 20261116
