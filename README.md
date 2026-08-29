# CEE420 Urban AI, course code

Princeton University, Fall 2026. Everything we run together in the Wednesday precepts lives here.
Each week adds a folder. You get the new material with one click in GitHub Desktop.

| Folder | Session |
|---|---|
| `P01/` | Getting Started: Python, Maps & Your Coding Agent (Wed Sep 9) |
| `A01/` | Assignment 01: Your Town, Your Agent, Your Map (due Wed Sep 23) |

## Before the first precept

Do this at home, not in class. The downloads are large and the campus wifi is not.

1. Follow **[docs/setup-mac.md](docs/setup-mac.md)** or **[docs/setup-windows.md](docs/setup-windows.md)**.
   You install VS Code, Docker Desktop, and GitHub Desktop, then clone this repository and open it once
   in its container.
2. Open `P01/00-check.ipynb` and click **Run All**. A map of Princeton should appear.
3. Reply **DONE** in the Canvas pre-flight thread, or tell us what broke. Either answer helps.

No laptop that can run a browser is out of the game: if the local install fights you, use
**[docs/setup-codespaces.md](docs/setup-codespaces.md)** and work in the browser instead.

## Working in this repository

Your environment is set up when Princeton appears.

- **Before you edit a notebook, make it yours**: File > Save As, add `-mywork` to the name
  (`01-first-map-mywork.ipynb`). Files named `-mywork` are ignored by git, so your work is yours and
  `git pull` will never overwrite it or refuse to run.
- **Getting each week's code**: open GitHub Desktop, select this repository, click **Pull origin**.
  New folders appear; nothing you wrote is touched.
- **If a pull ever complains**, see [docs/troubleshooting.md](docs/troubleshooting.md). The short
  version: save your file under a `-mywork` name, then discard changes to the original.

## The three ways to run the code

| Path | When | Guide |
|---|---|---|
| VS Code + Dev Container | The normal path, your own laptop | [mac](docs/setup-mac.md), [windows](docs/setup-windows.md) |
| GitHub Codespaces | Docker will not run on your machine, or you are on a borrowed one | [codespaces](docs/setup-codespaces.md) |
| Local Python with uv | You already manage Python yourself and prefer to | `uv sync && uv run jupyter lab` |

The container and the `uv` path install exactly the same packages, pinned in `pyproject.toml` and
`uv.lock`. Nothing else is needed, and no notebook in this course should ever ask you to `pip install`.

## Your coding agent

The course agent for the first weeks is Google Antigravity, on its free tier. Setup and sign-in are in
[docs/agent-antigravity.md](docs/agent-antigravity.md). Any agent you already have is welcome too. What
the course asks is not which tool you use but that you can say what it did and what you verified.

## Licenses

Course code and notebooks are MIT (see `LICENSE`), so reuse them freely. Data files carry their own
terms, listed in `P01/data/README.md`: OpenStreetMap extracts are ODbL 1.0 and require attribution,
public-record boundaries are free to use.
