"""The P05 notebooks keep the shape of the P03b kit, with one addition for the deep learning library.

Students run and read these notebooks; they do not write code. So every code cell is short and calls the
toolbox `seg_tools.py` instead of defining anything, every text cell is short and plain, each notebook has
exactly one "change me" value and no blanks to fill, and nothing reaches outside the kit's folder. A cell
that needs PyTorch sits under `if torch_ok:` with one printed line in the `else`, so the kit runs to the end
without it; no notebook imports the library itself. The toolbox stays readable and self-contained.
"""

import ast
import json
import re
from importlib.util import module_from_spec, spec_from_file_location

import pytest

from conftest import P05_KIT

NOTEBOOKS = sorted(p for p in P05_KIT.glob("05_*.ipynb") if "-mywork" not in p.name)
IDS = [p.stem for p in NOTEBOOKS]
MODULE = P05_KIT / "seg_tools.py"
ALLOWED_IMPORTS = {"json", "math", "time", "copy", "pathlib", "inspect", "warnings", "numpy", "pandas", "scipy", "rasterio", "PIL",
                   "matplotlib", "shapely", "sklearn", "torch", "torchvision"}
MAX_CODE_LINES = 12
MAX_GUARDED_LINES = 15           # three lines of guard around a twelve-line body
MAX_LINE_CHARS = 110
MAX_TOOLBOX_LINE_CHARS = 140
MAX_MD_WORDS = 120
MARKER = "# change me:"
GUARD = "if torch_ok:"


def cells(notebook, kind):
    for index, cell in enumerate(json.loads(notebook.read_text())["cells"]):
        if cell["cell_type"] == kind:
            yield index, "".join(cell["source"])


def code_lines(text):
    return [line for line in text.splitlines() if line.strip()]


def test_the_four_notebooks_carry_plain_names():
    expected = ["05_01_chips_and_masks", "05_02_semantic_segmentation", "05_03_fine_tuning", "05_04_street_level"]
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
    """At most twelve lines of code per cell (the setup cell excepted, a guarded cell gets three more), at most 120
    words per text cell (the Further resources cell excepted)."""
    code = list(cells(notebook, "code"))
    for index, text in code[1:]:
        n, limit = len(code_lines(text)), MAX_GUARDED_LINES if GUARD in text else MAX_CODE_LINES
        assert n <= limit, f"{notebook.name} cell {index} has {n} code lines"
    for index, text in cells(notebook, "markdown"):
        if text.lstrip().startswith("## Further resources"):
            continue
        words = len(re.sub(r"http\S+", "", text).split())
        assert words <= MAX_MD_WORDS, f"{notebook.name} cell {index} has {words} words"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_notebook_uses_the_toolbox_and_one_change_me(notebook):
    code = [text for _, text in cells(notebook, "code")]
    assert any("import seg_tools as seg" in text for text in code), f"{notebook.name} does not import the toolbox"
    marked = [text for text in code if MARKER in text]
    assert len(marked) == 1, f"{notebook.name} has {len(marked)} cells with a change-me value, one is required"
    assert sum(line.count(MARKER) for line in marked[0].splitlines()) == 1
    for text in code:
        assert "___" not in text, f"{notebook.name} still has a blank to fill"
    tags = [tag for cell in json.loads(notebook.read_text())["cells"] for tag in cell.get("metadata", {}).get("tags", [])]
    assert "break-glass" not in tags, f"{notebook.name} still carries an appendix"
    headings = [text.splitlines()[0] for _, text in cells(notebook, "markdown") if text.startswith("## ")]
    for required in ("## Your turn", "## Check your understanding", "## Where we are", "## Further resources"):
        assert required in headings, f"{notebook.name} lacks {required}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_kit_runs_without_the_deep_learning_library(notebook):
    """A cell that needs PyTorch is guarded and says so when the library is missing; no notebook imports it itself."""
    for index, text in cells(notebook, "code"):
        assert not re.search(r"^\s*(import|from)\s+torch", text, re.M), f"{notebook.name} cell {index} imports torch; the toolbox does that"
        needs = any(name in text for name in ("seg.load_model", "seg.predict", "seg.train_decoder", "seg.fresh_decoder", "seg.encoder_maps"))
        if needs:
            assert GUARD in text and "seg.NO_TORCH" in text, f"{notebook.name} cell {index} needs torch without the guard"
        if GUARD in text:
            assert re.search(r"^else:\n\s+print\(seg\.NO_TORCH\)", text, re.M), f"{notebook.name} cell {index}: the guard lacks its else line"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_every_data_path_in_code_exists(notebook):
    """The loaders hide the open() call from the style test, so every data path literal is checked here."""
    for index, text in cells(notebook, "code"):
        for quoted in re.findall(r"""["'](data/[^"'\n]+)["']""", text):
            assert (notebook.parent / quoted).exists(), f"{notebook.name} cell {index} names {quoted}, which is not there"


def test_the_toolbox_is_readable_and_self_contained():
    assert MODULE.exists(), "the kit ships seg_tools.py"
    source = MODULE.read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= ALLOWED_IMPORTS, f"seg_tools imports {sorted(imported - ALLOWED_IMPORTS)}"
    for forbidden in ("/Users", "data/scripts", "mapillary.env", "MLY|"):
        assert forbidden not in source, f"seg_tools.py mentions {forbidden}"
    for number, line in enumerate(source.splitlines(), 1):
        assert len(line) <= MAX_TOOLBOX_LINE_CHARS, f"seg_tools.py line {number} is {len(line)} characters"
    spec = spec_from_file_location("seg_tools_under_test", MODULE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__doc__ and len(module.__all__) >= 30
    for name in module.__all__:
        member = getattr(module, name)
        if name.isupper():
            continue                                   # a documented constant, such as the line printed without torch
        assert callable(member), f"{name} is not callable"
        assert member.__doc__, f"{name} has no docstring"
    assert isinstance(module.NO_TORCH, str) and "skipped" in module.NO_TORCH
