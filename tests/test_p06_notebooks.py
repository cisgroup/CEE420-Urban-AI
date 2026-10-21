"""The P06 notebooks keep the shape of the P05 kit, with one more guard for the models' weights.

Students run and read these notebooks; they do not write code. So every code cell is short and calls the toolbox
`fm_tools.py` instead of defining anything, every text cell is short and plain, each notebook has exactly one
"change me" value and no blanks to fill, and nothing reaches outside the kit's folder. A cell that runs a model
sits under `if torch_ok and ready[...]:` with one printed line in the `else`, so the kit runs to the end without
the deep learning library or the weights; no notebook imports torch itself. The toolbox stays readable and
self-contained, and no token or key reaches the kit.
"""

import ast
import json
import re
from importlib.util import module_from_spec, spec_from_file_location

import pytest

from conftest import P06_KIT

NOTEBOOKS = sorted(p for p in P06_KIT.glob("06_*.ipynb") if "-mywork" not in p.name)
IDS = [p.stem for p in NOTEBOOKS]
MODULE = P06_KIT / "fm_tools.py"
ALLOWED_IMPORTS = {"json", "os", "re", "time", "warnings", "logging", "textwrap", "pathlib", "numpy", "pandas", "scipy", "rasterio", "PIL", "matplotlib", "shapely",
                   "geopandas", "sklearn", "threadpoolctl", "torch", "transformers", "huggingface_hub"}
MAX_CODE_LINES = 12
MAX_GUARDED_LINES = 15           # three lines of guard around a twelve-line body
MAX_LINE_CHARS = 110
MAX_TOOLBOX_LINE_CHARS = 140
MAX_MD_WORDS = 120
MARKER = "# change me:"
GUARD = re.compile(r"""^if torch_ok and ready\[['"](sam2|clip)['"]\]:""", re.M)
MODEL_CALLS = ("fm.load_sam", "fm.embed_image", "fm.prompt(", "fm.load_clip", "fm.clip_image_features", "fm.clip_text_features")
SECRETS = re.compile(r"hf_[A-Za-z0-9]{20,}|HF_TOKEN=\S|MLY\|[0-9]+\|", re.I)


def cells(notebook, kind):
    for index, cell in enumerate(json.loads(notebook.read_text())["cells"]):
        if cell["cell_type"] == kind:
            yield index, "".join(cell["source"])


def code_lines(text):
    return [line for line in text.splitlines() if line.strip()]


def test_the_four_notebooks_carry_plain_names():
    expected = ["06_01_prompts_and_masks", "06_02_concepts_and_scores", "06_03_words_about_pictures", "06_04_a_vector_for_every_place"]
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
    """At most twelve lines of code per cell (the setup cell excepted, a guarded cell and the live lane get three more), at
    most 120 words per text cell (the Further resources cell excepted)."""
    code = list(cells(notebook, "code"))
    for index, text in code[1:]:
        guarded = GUARD.search(text) or "LIVE = False" in text or MARKER in text
        n, limit = len(code_lines(text)), MAX_GUARDED_LINES if guarded else MAX_CODE_LINES
        assert n <= limit, f"{notebook.name} cell {index} has {n} code lines"
    for index, text in cells(notebook, "markdown"):
        if text.lstrip().startswith("## Further resources"):
            continue
        words = len(re.sub(r"http\S+", "", text).split())
        assert words <= MAX_MD_WORDS, f"{notebook.name} cell {index} has {words} words"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_notebook_uses_the_toolbox_and_one_change_me(notebook):
    code = [text for _, text in cells(notebook, "code")]
    assert any("import fm_tools as fm" in text for text in code), f"{notebook.name} does not import the toolbox"
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
def test_the_kit_runs_without_the_models(notebook):
    """A cell that runs a model is guarded and says so when the library or the weights are missing; no notebook imports
    torch itself, and the live lane of the vision-language model stays off."""
    for index, text in cells(notebook, "code"):
        assert not re.search(r"^\s*(import|from)\s+torch", text, re.M), f"{notebook.name} cell {index} imports torch; the toolbox does that"
        if any(call in text for call in MODEL_CALLS):
            assert GUARD.search(text), f"{notebook.name} cell {index} runs a model without the guard"
        if GUARD.search(text):
            assert re.search(r"^else:\n\s+print\(fm\.NO_TORCH if not torch_ok else fm\.NO_MODEL\)", text, re.M), (
                f"{notebook.name} cell {index}: the guard lacks its else line")
        if "LIVE = " in text:
            assert "LIVE = False" in text, f"{notebook.name} cell {index} ships with the live lane switched on"
        assert not SECRETS.search(text), f"{notebook.name} cell {index} carries something that looks like a token"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_every_data_path_in_code_exists(notebook):
    """The loaders hide the open() call from the style test, so every data path literal is checked here."""
    for index, text in cells(notebook, "code"):
        for quoted in re.findall(r"""["'](data/[^"'\n{]+)["']""", text):
            assert (notebook.parent / quoted).exists(), f"{notebook.name} cell {index} names {quoted}, which is not there"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_credits_are_in_the_prose(notebook):
    """The attribution line of the embeddings, the licence of the text model's masks and the OpenStreetMap credit appear
    wherever the notebook reads those files."""
    code = "\n".join(text for _, text in cells(notebook, "code"))
    prose = "\n".join(text for _, text in cells(notebook, "markdown"))
    if "aef_town" in code:
        assert "Google DeepMind" in prose, f"{notebook.name} reads the embeddings without their attribution"
    if "masks_sam3" in code:
        assert "SAM License" in prose, f"{notebook.name} reads the text model's masks without stating their licence"
    if "footprints_" in code or "mask_" in code:
        assert "OpenStreetMap" in prose and "ODbL" in prose, f"{notebook.name} uses OpenStreetMap data without crediting it"


def test_the_toolbox_is_readable_and_self_contained():
    assert MODULE.exists(), "the kit ships fm_tools.py"
    source = MODULE.read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= ALLOWED_IMPORTS, f"fm_tools imports {sorted(imported - ALLOWED_IMPORTS)}"
    for forbidden in ("/Users", "data/scripts", "huggingface.env", "mapillary.env", "MLY|", "hf_"):
        assert forbidden not in source, f"fm_tools.py mentions {forbidden}"
    for number, line in enumerate(source.splitlines(), 1):
        assert len(line) <= MAX_TOOLBOX_LINE_CHARS, f"fm_tools.py line {number} is {len(line)} characters"
    spec = spec_from_file_location("fm_tools_under_test", MODULE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__doc__ and len(module.__all__) >= 60
    for name in module.__all__:
        member = getattr(module, name)
        if name.isupper():
            continue                                   # a documented constant, such as the lines printed without a model
        assert callable(member), f"{name} is not callable"
        assert member.__doc__, f"{name} has no docstring"
    assert isinstance(module.NO_TORCH, str) and "skipped" in module.NO_TORCH
    assert isinstance(module.NO_MODEL, str) and "skipped" in module.NO_MODEL
    assert set(module.MODELS) == {"sam2", "clip", "vlm"}


def test_my_images_stays_private():
    """The folder for the students' own pictures ships empty but for its README, and git ignores whatever lands there."""
    folder = P06_KIT / "my_images"
    assert (folder / "README.org").exists()
    others = [p.name for p in folder.iterdir() if p.name != "README.org" and not p.name.startswith(".")]
    assert not others, f"my_images ships {others}"
    ignore = (P06_KIT.parent / ".gitignore").read_text()
    assert "P06/my_images/*" in ignore and "P06/output/" in ignore
