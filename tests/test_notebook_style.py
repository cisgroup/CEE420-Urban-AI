"""The notebooks are teaching documents, so the writing rules are tested like the data.

Only notebooks are scanned. This file quotes emoji code ranges as escapes, so pointing
these patterns at the test sources themselves would flag this file.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, faces, transport, symbols
    "☀-➿"          # miscellaneous symbols and dingbats, includes the check mark
    "⬀-⯿"          # arrows and stars
    "️"                 # the selector that turns a plain glyph into an emoji
    "]"
)
ESCAPED_EMOJI = re.compile(r"\\[uU]0*(1f[0-9a-f]{3}|2[67][0-9a-f]{2}|2b[0-9a-f]{2})", re.IGNORECASE)
DASHES = {"—": "em dash", "–": "en dash", "―": "horizontal bar", "−": "minus sign"}
READ_PATH = re.compile(
    r"""(?:read_file|read_csv|read_parquet|open)\(\s*["']([^"'\n]+)["']"""
)
ON_RAMP_LINKS = ("geo-python-site.readthedocs.io", "autogis-site.readthedocs.io")
FURTHER_RESOURCES = "## Further resources"


def notebooks():
    found = [
        path
        for path in sorted(REPO.glob("*/*.ipynb"))
        if not any(part.startswith(".") for part in path.relative_to(REPO).parts)
        and "-mywork" not in path.name
    ]
    assert found, "no notebooks were found, so every test in this file would pass vacuously"
    return found


NOTEBOOKS = notebooks()
IDS = [f"{p.parent.name}/{p.name}" for p in NOTEBOOKS]


def sources(notebook, kind=None):
    cells = json.loads(notebook.read_text())["cells"]
    return [
        ("".join(c["source"]), c["cell_type"], i)
        for i, c in enumerate(cells)
        if kind is None or c["cell_type"] == kind
    ]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_no_emoji_in_any_notebook(notebook):
    """House style. Escaped forms count, since a print of an emoji escape shows an emoji."""
    offenders = [
        (index, kind, match.group())
        for text, kind, index in sources(notebook)
        for pattern in (EMOJI, ESCAPED_EMOJI)
        for match in pattern.finditer(text)
    ]
    assert not offenders, f"{notebook.name} contains emoji: {offenders}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_no_em_or_en_dashes_in_any_notebook(notebook):
    """The course style rule, enforced where students actually read."""
    offenders = [
        (index, name)
        for text, _kind, index in sources(notebook)
        for char, name in DASHES.items()
        if char in text
    ]
    assert not offenders, f"{notebook.name} uses a dash character: {offenders}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_every_notebook_offers_further_resources(notebook):
    """A self-contained lesson still owes the reader a way onward, clearly marked optional."""
    prose = "\n".join(text for text, _kind, _i in sources(notebook, "markdown"))
    assert FURTHER_RESOURCES in prose, f"{notebook.name} has no '{FURTHER_RESOURCES}' section"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_lesson_itself_never_sends_a_student_elsewhere(notebook):
    """Everything a student needs is in the notebook.

    Links to the self-study on-ramp are allowed only below 'Further resources', where by
    construction nothing needed lives. Above it, a link is an admission that the notebook
    did not explain something it relies on.
    """
    prose = "\n".join(text for text, _kind, _i in sources(notebook, "markdown"))
    body = prose.split(FURTHER_RESOURCES)[0]
    offenders = [host for host in ON_RAMP_LINKS if host in body]
    assert not offenders, (
        f"{notebook.name} outsources part of the lesson to {offenders}. "
        "Explain it in the notebook, and keep the reading list under 'Further resources'."
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_notebooks_only_use_files_from_their_own_folder(notebook):
    """A notebook that reaches into a sibling folder breaks the moment it travels alone.

    Only paths a notebook actually reads are checked. A path it writes to (the live
    lane in A01 saves what it fetched) does not exist until the notebook has run.
    """
    for text, kind, index in sources(notebook):
        assert "../" not in text, f"{notebook.name} cell {index} ({kind}) reaches outside its folder"
        if kind != "code":
            continue  # prose may name a file without opening it
        for quoted in READ_PATH.findall(text):
            assert not Path(quoted).is_absolute(), f"{notebook.name} cell {index}: absolute path"
            assert (notebook.parent / quoted).exists(), (
                f"{notebook.name} cell {index} reads {quoted}, which is not there"
            )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_no_cell_is_excused_from_execution(notebook):
    """`skip-execution` is how untested code hides in a repository with a green tick.

    If a cell truly cannot run here, run it from a python test instead.
    """
    cells = json.loads(notebook.read_text())["cells"]
    skipped = [
        i for i, c in enumerate(cells) if "skip-execution" in c.get("metadata", {}).get("tags", [])
    ]
    assert not skipped, f"{notebook.name} excuses cells {skipped} from execution"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_notebooks_using_osm_data_credit_openstreetmap(notebook):
    """ODbL requires attribution, and the shipped files carry it. So must the prose."""
    cells = json.loads(notebook.read_text())["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    prose = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    if "_food.geojson" not in code and "_buildings.geojson" not in code:
        pytest.skip("no OpenStreetMap layer in this notebook")
    assert "OpenStreetMap" in prose and "ODbL" in prose, (
        f"{notebook.name} uses OpenStreetMap data without crediting it in the text"
    )
