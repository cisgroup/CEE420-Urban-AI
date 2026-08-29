# Setup on Windows

Time: about 45 minutes, most of it downloading and one restart. Do this at home before the first
precept, on the laptop you will bring to class. You need about 10 GB of free disk space and 8 GB of
memory.

Windows needs two things macOS does not: a Linux subsystem called WSL2, and virtualization enabled in
your firmware. Steps 1 and 2 handle both. They are the only genuinely fiddly part; after that the path
is the same for everyone.

If any step fails, stop and write down the exact message. An honest report in the Canvas thread is
worth more than an hour of fighting it alone, and we have a clinic for exactly this.

---

## 1. Check your machine can run this

**Memory.** Right-click the taskbar, open **Task Manager**, choose the **Performance** tab, click
**Memory**. You want 8 GB or more. With 4 GB, stop here and use
[Codespaces](setup-codespaces.md) instead; that is a perfectly good lane, not a consolation prize.

**Virtualization.** In the same Performance tab, click **CPU** and look at the bottom right for
**Virtualization**. It should say **Enabled**.

[SCREENSHOT: win-01-task-manager-virtualization.png, Task Manager CPU page with Virtualization: Enabled]

If it says Disabled, it must be switched on in your computer's firmware, which is a menu that appears
before Windows starts:

1. Windows menu > Settings > **System** > **Recovery** > **Advanced startup** > **Restart now**.
2. Choose **Troubleshoot** > **Advanced options** > **UEFI Firmware Settings** > **Restart**.
3. In the firmware menu, find the virtualization setting. Its name depends on the maker:
   **Intel VT-x**, **Intel Virtualization Technology**, **SVM Mode** (AMD), or **Virtualization**. It
   usually sits under Advanced, CPU Configuration, or Security.
4. Enable it, save, and exit. Windows restarts. Check Task Manager again.

If you cannot find it, do not force it. Use Codespaces and tell us which laptop model you have.

**Windows version.** Press `Win+R`, type `winver`, press Return. Docker needs Windows 11 version 23H2
or newer, or Windows 10 version 22H2. Anything older will not run it, so use
[Codespaces](setup-codespaces.md) instead. If Windows Update has been waiting for you, this is the
moment to let it run.

## 2. Install WSL2

1. Click the Windows menu, type `PowerShell`, right-click **Windows PowerShell**, choose
   **Run as administrator**.
2. Type this and press Return:

   ```powershell
   wsl --install
   ```

3. Let it finish, then **restart your computer**. This restart is not optional.
4. After the restart, open PowerShell again (normal, not administrator) and run:

   ```powershell
   wsl --update
   ```

[SCREENSHOT: win-02-wsl-install.png, PowerShell running wsl --install successfully]

If Windows opens an Ubuntu window asking for a username and password, you may close it. The course
does not need it, though setting one does no harm.

## 3. Install VS Code

1. Go to <https://code.visualstudio.com/download> and click the **Windows** button.
2. Run the installer. Accept the defaults; the offered **Add to PATH** option is worth keeping.

![The VS Code download page](img/win-03-vscode-download.png)

## 4. Install Docker Desktop

Docker runs the course environment, a small prepared Linux system that already contains Python and
every package we use. You will never have to install a Python package by hand in this course.

1. Go to <https://www.docker.com/products/docker-desktop/> and download for **Windows**.
2. Run the installer. When it asks, make sure **Use WSL 2 instead of Hyper-V** is ticked.
3. Restart if it asks. Then launch **Docker Desktop** from the Windows menu.
4. Accept the service agreement, and accept **Use recommended settings** if offered.
5. You may skip the sign-in it suggests. **You do not need a Docker account for this course.**
6. Wait until the Docker Desktop window says **Engine running** in the bottom left.

[SCREENSHOT: win-04-docker-wsl2-option.png, the installer with the WSL 2 checkbox ticked]
[SCREENSHOT: win-05-docker-running.png, Docker Desktop showing Engine running]

**Leave Docker running before class.** If Docker Desktop is closed, the next steps fail with a
confusing message about a daemon.

## 5. Install GitHub Desktop and get the course code

1. Go to <https://desktop.github.com/> and download for Windows. Run it.
2. You can skip signing in. Our repository is public, so you need no account to get the code.
3. Choose **Clone a repository from the Internet**, then the **URL** tab, and paste:

   ```
   https://github.com/cisgroup/CEE420-Urban-AI
   ```

4. Note the **Local path** it offers, something like `C:\Users\you\Documents\GitHub\CEE420-Urban-AI`.
   Keep it in your user folder; do not put it on a network drive or in OneDrive, which cause odd
   permission problems later. Click **Clone**.

[SCREENSHOT: win-06-github-desktop-clone.png, the Clone a repository dialog with the URL filled in]

## 6. Connect VS Code to Docker

1. Open VS Code.
2. Click the **Extensions** icon in the left bar (four squares, one detached).
3. Search for `Dev Containers`, the one published by Microsoft, and click **Install**.

[SCREENSHOT: win-07-devcontainers-extension.png, the Dev Containers extension page inside VS Code]

## 7. Open the course in its container

This step downloads the environment, roughly 1 to 2 GB, so do it on decent wifi and expect several
minutes the first time. It happens exactly once; every later start takes seconds.

1. In VS Code choose **File > Open Folder** and select the folder GitHub Desktop cloned in step 5.
2. VS Code shows a notification in the bottom right: **Reopen in Container**. Click it.
   If you miss it, press `Ctrl+Shift+P`, type `Reopen in Container`, and press Return.
3. Watch the bottom left corner. It ends up reading **Dev Container: CEE420 Urban AI (FA26)**. That
   green label is how you know you are inside the course environment rather than on bare Windows.

[SCREENSHOT: win-08-reopen-in-container.png, the Reopen in Container notification]
[SCREENSHOT: win-09-container-connected.png, the green Dev Container label, bottom left]

## 8. Prove it works

1. In the file list on the left, open the `P01` folder and click `00-check.ipynb`.
2. Click **Run All** in the toolbar above the notebook.
3. If VS Code asks you to select a kernel, choose **Urban AI (Python 3.12)**. Ignore anything
   mentioning a Python on your own Windows.
4. A map of Princeton should appear, followed by a green READY line.

[SCREENSHOT: win-10-select-kernel.png, the kernel picker with Urban AI (Python 3.12) highlighted]
[SCREENSHOT: win-11-princeton-appears.png, the rendered Princeton boundary and the READY message]

**Your environment is set up when Princeton appears.** Reply DONE in the Canvas thread.

## 9. Getting each week's code

Every Wednesday brings a new folder. To get it:

1. Open **GitHub Desktop**.
2. Make sure **CEE420-Urban-AI** is the current repository, top left.
3. Click **Pull origin**.

One rule keeps this painless: **before you edit any notebook, save your own copy** with File > Save As
and `-mywork` in the name, for example `01-first-map-mywork.ipynb`. Your copies are invisible to git,
so a pull can never overwrite your work or refuse to run.

If something does go wrong, [docs/troubleshooting.md](troubleshooting.md) has the two-minute fix.

## When it does not work

- **"Cannot connect to the Docker daemon" or "Docker Desktop is starting"**: open Docker Desktop and
  wait for Engine running, then retry.
- **Docker says WSL 2 is not installed or out of date**: run `wsl --update` in PowerShell, restart,
  and open Docker again.
- **Virtualization still says Disabled after the firmware change**: some machines have a second switch
  (often named Hyper-V or SVM). Bring the laptop to the clinic, we will look together.
- **Everything is painfully slow**: check that the cloned folder is in your user folder rather than
  OneDrive or a network drive, and close other heavy apps.
- **Anything else**: bring it to the clinic or the Canvas thread. Do not spend your evening on it, and
  do not plan to fix it during class.
