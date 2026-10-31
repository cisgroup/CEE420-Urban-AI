"""The P07 notebooks keep the shape of the P06 kit, without a model to guard.

Students run and read these notebooks; they do not write code. So every code cell is short and calls the toolbox
`mob_tools.py` instead of defining anything, every text cell is short and plain, each notebook has exactly one
"change me" value and no blanks to fill, and nothing reaches outside the kit's folder. The lanes for your own trace
and for the class's counts run to their end whether or not the files are there, and no notebook imports osmnx or
networkx itself. The toolbox stays readable and self-contained, and no token or key reaches the kit.
"""

import ast
import json
import re
from importlib.util import module_from_spec, spec_from_file_location

import pytest

from conftest import P07_KIT

NOTEBOOKS = sorted(p for p in P07_KIT.glob("07_*.ipynb") if "-mywork" not in p.name)
IDS = [p.stem for p in NOTEBOOKS]
MODULE = P07_KIT / "mob_tools.py"
ALLOWED_IMPORTS = {"collections", "geopandas", "hashlib", "heapq", "itertools", "json", "math", "matplotlib", "networkx", "numpy", "osmnx", "pandas", "pathlib", "pyproj", "scipy", "shapely", "sklearn", "time", "xml", "re"}
MAX_CODE_LINES = 12
MAX_MARKED_LINES = 15            # the your-turn cell gets three more
MAX_LINE_CHARS = 110
MAX_TOOLBOX_LINE_CHARS = 150
MAX_MD_WORDS = 120
MARKER = "# change me:"
SECRETS = re.compile(r"hf_[A-Za-z0-9]{20,}|HF_TOKEN=\S|MLY\|[0-9]+\||CENSUS_API_KEY=\S", re.I)


def cells(notebook, kind):
    for index, cell in enumerate(json.loads(notebook.read_text())["cells"]):
        if cell["cell_type"] == kind:
            yield index, "".join(cell["source"])


def code_lines(text):
    return [line for line in text.splitlines() if line.strip()]


def test_the_four_notebooks_carry_plain_names():
    expected = ["07_01_networks", "07_02_access", "07_03_traces", "07_04_flows_and_privacy"]
    assert IDS == expected, f"the kit holds {IDS}"


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
    """At most twelve lines of code per cell (the setup cell excepted, the your-turn cell gets three more), at most 120
    words per text cell (the Further resources cell excepted)."""
    code = list(cells(notebook, "code"))
    for index, text in code[1:]:
        n, limit = len(code_lines(text)), MAX_MARKED_LINES if MARKER in text else MAX_CODE_LINES
        assert n <= limit, f"{notebook.name} cell {index} has {n} code lines"
    for index, text in cells(notebook, "markdown"):
        if text.lstrip().startswith("## Further resources"):
            continue
        words = len(re.sub(r"http\S+", "", text).split())
        assert words <= MAX_MD_WORDS, f"{notebook.name} cell {index} has {words} words"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_notebook_uses_the_toolbox_and_one_change_me(notebook):
    code = [text for _, text in cells(notebook, "code")]
    assert any("import mob_tools as mob" in text for text in code), f"{notebook.name} does not import the toolbox"
    marked = [text for text in code if MARKER in text]
    assert len(marked) == 1, f"{notebook.name} has {len(marked)} cells with a change-me value, one is required"
    assert sum(line.count(MARKER) for line in marked[0].splitlines()) == 1
    for text in code:
        assert "___" not in text, f"{notebook.name} still has a blank to fill"
        assert not re.search(r"^\s*(import|from)\s+(osmnx|networkx|torch)", text, re.M), f"{notebook.name} imports a library the toolbox wraps"
        assert not SECRETS.search(text), f"{notebook.name} carries something that looks like a token"
    tags = [tag for cell in json.loads(notebook.read_text())["cells"] for tag in cell.get("metadata", {}).get("tags", [])]
    assert "break-glass" not in tags, f"{notebook.name} still carries an appendix"
    headings = [text.splitlines()[0] for _, text in cells(notebook, "markdown") if text.startswith("## ")]
    for required in ("## Your turn", "## Check your understanding", "## Where we are", "## Further resources"):
        assert required in headings, f"{notebook.name} lacks {required}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_every_data_path_in_code_exists(notebook):
    """The loaders hide the open() call from the style test, so every data path literal is checked here; the class's
    counts are optional and read only behind their flag."""
    for index, text in cells(notebook, "code"):
        for quoted in re.findall(r"""["'](data/[^"'\n{]+)["']""", text):
            if quoted == "data/class_counts.csv.gz":
                assert 'reference["class_traces"]["published"]' in text, f"{notebook.name} cell {index} reads the class's counts without the flag"
                continue
            assert (notebook.parent / quoted).exists(), f"{notebook.name} cell {index} names {quoted}, which is not there"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_credits_are_in_the_prose(notebook):
    """OpenStreetMap's credit appears wherever a network or a layer is read, the specimen's and Porto's licences where
    their files are, the Census's where the flows are."""
    code = "\n".join(text for _, text in cells(notebook, "code"))
    prose = "\n".join(text for _, text in cells(notebook, "markdown"))
    if ".graphml" in code or "layers.gpkg" in code:
        assert "OpenStreetMap" in prose and "ODbL" in prose, f"{notebook.name} uses OpenStreetMap data without crediting it"
    if "themepark_" in code:
        assert "Utley and Hackl" in prose and "CC BY 4.0" in prose, f"{notebook.name} uses the specimen without its licence"
    if "porto_" in code:
        assert "Moreira-Matias" in prose, f"{notebook.name} uses Porto's trips without their licence"
    if "lodes_" in code or "princeton_inflow" in code:
        assert "Census" in prose, f"{notebook.name} uses the commuting flows without their source"
    if "junction_cells" in code or "dinky.json" in code:
        assert "NJ Transit" in prose, f"{notebook.name} uses the router's minutes or the timetable without the feed's terms"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_lanes_run_without_their_files(notebook):
    """The own-trace lane and the class's counts sit behind a check, so the notebook runs to its end without them."""
    code = "\n".join(text for _, text in cells(notebook, "code"))
    if "my_traces" in code:
        assert "mob.first_trace(" in code and "mob.NO_TRACE" in code, f"{notebook.name} reads my_traces without the fallback"
    if "counts_file" in code or "class_counts" in code:
        assert 'reference["class_traces"]["published"]' in code, f"{notebook.name} reads the class's counts without the flag"


def test_the_toolbox_is_readable_and_self_contained():
    assert MODULE.exists(), "the kit ships mob_tools.py"
    source = MODULE.read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= ALLOWED_IMPORTS, f"mob_tools imports {sorted(imported - ALLOWED_IMPORTS)}"
    for forbidden in ("/Users", "census.env", "MLY|", "hf_"):
        assert forbidden not in source, f"mob_tools.py mentions {forbidden}"
    for number, line in enumerate(source.splitlines(), 1):
        assert len(line) <= MAX_TOOLBOX_LINE_CHARS, f"mob_tools.py line {number} is {len(line)} characters"
    spec = spec_from_file_location("mob_tools_under_test", MODULE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__doc__ and len(module.__all__) >= 60
    for name in module.__all__:
        member = getattr(module, name)
        if name.isupper():
            continue                                   # a documented constant, such as the line printed without a trace
        assert callable(member), f"{name} is not callable"
        assert member.__doc__, f"{name} has no docstring"
    assert isinstance(module.NO_TRACE, str) and "specimen" in module.NO_TRACE
    assert module.CRS == "EPSG:26918" and module.WALK_KPH == 4.8


def test_my_traces_stays_private():
    """The folder for the students' own traces ships empty but for its README, and git ignores whatever lands there."""
    folder = P07_KIT / "my_traces"
    assert (folder / "README.org").exists()
    others = [p.name for p in folder.iterdir() if p.name != "README.org" and not p.name.startswith(".")]
    assert not others, f"my_traces ships {others}"
    ignore = (P07_KIT.parent / ".gitignore").read_text()
    assert "P07/my_traces/*" in ignore and "P07/output/" in ignore
