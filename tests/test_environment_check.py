"""The environment check must test the whole environment, not part of it.

This exists because of a real failure. seaborn was added to the dependencies, the
published image was rebuilt with it, but a stale image on one machine did not have it.
01_01 imported five packages, none of them seaborn, so it printed "READY. Your
environment works." and 01_02 then died on its first cell. A check that certifies an
environment it has not tested is worse than no check, because it converts a clear early
failure into a confusing later one.
"""

import json
import re
from importlib import import_module
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / "P01" / "01_01_environment_check.ipynb"

STDLIB = {
    "json", "pathlib", "sys", "platform", "re", "math", "os", "textwrap",
    "itertools", "random", "warnings", "time", "datetime", "collections",
    "subprocess", "importlib", "shutil",
}


def top_level_imports(notebook: Path) -> set[str]:
    found = set()
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for line in "".join(cell["source"]).splitlines():
            match = re.match(r"\s*(?:import|from)\s+([A-Za-z_]\w*)", line)
            if match:
                found.add(match.group(1))
    return found - STDLIB


def test_the_check_imports_everything_the_notebooks_import():
    checked = top_level_imports(CHECK)
    used = set()
    for notebook in sorted(REPO.glob("[PF]0*/*.ipynb")):
        used |= top_level_imports(notebook)

    # A name that cannot be imported anywhere is a deliberate typo in an error-reading
    # lesson, e.g. "geopandaz", rather than a dependency the environment owes us.
    real = set()
    for name in used:
        try:
            import_module(name)
            real.add(name)
        except Exception:  # noqa: BLE001
            pass

    missing = sorted(real - checked)
    assert not missing, (
        "01_01_environment_check does not import "
        + ", ".join(missing)
        + ", so it would pass on an environment where those notebooks fail"
    )
