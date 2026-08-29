"""The break-glass appendix must actually work: extract its fenced code and run it."""

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "P01" / "01-first-map.ipynb"
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def fenced_blocks():
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if "break-glass" in cell.get("metadata", {}).get("tags", []):
            yield from FENCE.findall("".join(cell["source"]))


def test_break_glass_solutions_run_and_are_right(ground_truth):
    blocks = list(fenced_blocks())
    assert len(blocks) >= 2, "expected at least the checks solution and the Duel solution"
    cwd = os.getcwd()
    os.chdir(REPO / "P01")
    try:
        for block in blocks:
            namespace: dict = {}
            exec(block, namespace)  # noqa: S102
            if "answer" in namespace:
                assert namespace["answer"] == ground_truth["duel"]["food_within_400m_of_e225"]
    finally:
        os.chdir(cwd)
