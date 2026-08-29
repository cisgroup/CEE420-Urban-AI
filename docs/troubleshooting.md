# Troubleshooting

Rules of the house. **Two minutes, then switch lanes**: if something will not work, do not sink the
session into it. Move to [Codespaces](setup-codespaces.md), keep up with the class, and fix the local
setup later at the install clinic. Nobody debugs an installation during class, and nothing you miss
that way is lost.

---

## Getting the code, and keeping your own work

### The one habit that prevents all git pain

Before editing any notebook, **File > Save As** and put `-mywork` in the name:
`01-first-map-mywork.ipynb`. Files with `-mywork` are ignored by git. Your work is yours, and a pull
can never overwrite it or refuse to run.

### "Pull origin" is greyed out or nothing happens

You are already up to date, or GitHub Desktop is pointed at a different repository. Check the
repository name in the top left.

### A pull refuses because you have local changes

This means you edited a course file directly instead of a `-mywork` copy. Two minutes to fix:

1. Open the file you edited. **File > Save As**, add `-mywork` to the name. Your work is now safe.
2. In GitHub Desktop, find the original file in the **Changes** list, right-click it, choose
   **Discard changes**.
3. Click **Pull origin** again.

In a Codespace, the same thing lives in the **Source Control** panel: save your copy, then use the
**Discard changes** arrow on the original file.

### You want to start completely fresh

Delete the whole `CEE420-Urban-AI` folder and clone it again with GitHub Desktop. First, copy any
`-mywork` files somewhere safe. Everything else is replaceable.

---

## Docker and the container

### "Cannot connect to the Docker daemon"

Docker Desktop is not running. Open it and wait for **Engine running**, then retry in VS Code.

### VS Code never offers "Reopen in Container"

Either the Dev Containers extension is missing (Extensions sidebar, search `Dev Containers`, install),
or you opened the wrong folder. You must open the folder that directly contains `README.md` and the
`P01` folder, not its parent and not a subfolder.

### The container download stalls or fails

Cancel and retry; it resumes rather than starting over. On a hotel or airport network, try again on
better wifi. If it fails repeatedly, use Codespaces for now.

### It opened, but nothing works and there is no green label

You are not actually in the container. The bottom left corner must read **Dev Container: CEE420 Urban
AI (FA26)**. If it does not, run `Reopen in Container` from the command palette (`Cmd+Shift+P` or
`Ctrl+Shift+P`).

### We shipped an environment update

Rare, and you will be told on Canvas. One command: `Cmd+Shift+P` or `Ctrl+Shift+P`, then
**Dev Containers: Rebuild Container**.

---

## Notebooks

### "Select kernel" and none of the options look right

Choose **Urban AI (Python 3.12)**. If you only see options mentioning your own computer's Python, you
are not inside the container; see the green label above.

### ModuleNotFoundError, for example "No module named geopandas"

Same cause, almost always: the notebook is running on your computer's Python instead of the course
environment. Check the green label, then the kernel picker in the top right of the notebook.

### A cell errors and I cannot tell why

Read the last line of the message first; that is the actual error. Then hand it to your agent with the
error text and one question: what was wrong, in one line? Debugging with an agent is a skill this
course wants you to practise, not something to feel bad about.

### The interactive map is blank or the editor freezes

Interactive maps draw every feature into the page. The boundary and the food points are small enough;
all 7,893 buildings are not. If you asked for too much, close the notebook without saving, reopen it,
and filter your data first.

Interactive maps also need internet for their background tiles. If the campus wifi is unhappy, use
`.plot()` instead of `.explore()` and carry on; nothing in the course depends on the pretty version.

### Everything is slow

Close other applications, especially browsers with many tabs. On 8 GB machines Docker and Chrome
compete for memory. If it stays unusable, Codespaces will be faster than your laptop.

---

## Still stuck

Post in the Canvas Q&A forum with three things: what you were doing, the exact message, and which
operating system you are on. Someone else has usually hit it, and the answer helps everyone. Anything
you would rather not post publicly, email me.
