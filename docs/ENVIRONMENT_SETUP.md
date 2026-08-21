# Environment setup

## Hardware

- CPU/APU: AMD Ryzen AI 9 365 ("Strix Point")
- iGPU: AMD Radeon 880M (gfx1103, RDNA3)
- RAM: 32 GB
- OS: Windows 11 Home (build 26200)

## Why training runs in WSL2, not native Windows

ROCm on native Windows currently (as of 2026-08) only supports discrete
RDNA3/RDNA4 GPUs (gfx1100/1101/1200/1201 - Radeon RX 7000/9000 series). The
Radeon 880M (gfx1103) is not on that list and does not work with native
Windows ROCm.

AMD's WSL2 path (ROCDXG), first shipped in **AMD Software: Adrenalin
Edition 26.2.2** (released 2026-02-26), added official support for Ryzen AI
300 "Strix Point" APUs (this chip) alongside Strix Halo and RX 7000/9000.
That pairs with **ROCm 7.2.1** and **PyTorch 2.9**, targeting **Python
3.12** inside the WSL2 Linux environment.

So the GPU training environment is: Windows host driver (WSL-capable) ->
WSL2 Ubuntu -> ROCm 7.2.1 -> PyTorch 2.9 (ROCm build) -> Python 3.12.

## Setup steps

### 1. Update the AMD GPU driver (Windows host, admin required)

Download **AMD Software: Adrenalin Edition 26.2.2 or later** from:
https://www.amd.com/en/support/downloads/previous-drivers.html/processors/ryzen/ryzen-ai-300-series/amd-ryzen-ai-9-hx-365.html

Install it and reboot. This driver is what exposes the GPU to WSL2 via
ROCDXG; without it, WSL2 has no GPU access regardless of what's installed
inside the Linux environment.

### 2. Install WSL2 + Ubuntu (Windows host, admin required)

In an **elevated** PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot when prompted. On first launch, Ubuntu will ask you to create a
UNIX username/password.

### 3. Verify the GPU is visible inside WSL2

```bash
# inside WSL2 Ubuntu
ls /dev/dxg   # should exist if the Windows driver exposed the GPU to WSL
```

### 4. Install ROCm 7.2.1 inside WSL2

Follow AMD's current WSL install guide (check for the latest version
number, since these change):
https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html

### 5. Install Python 3.12 and PyTorch (ROCm build) inside WSL2

```bash
sudo apt install python3.12 python3.12-venv
python3.12 -m venv ~/venvs/facelib
source ~/venvs/facelib/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0  # confirm exact ROCm tag on pytorch.org
pip install -r requirements/base.txt
```

### 6. Verify PyTorch actually sees the GPU

Do not assume a successful install means the GPU is used. Confirm with
PyTorch itself:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

(PyTorch's ROCm backend is exposed through the `torch.cuda` API, so
`torch.cuda.is_available()` is the correct check even though this is an AMD
GPU, not NVIDIA.)

If this prints `False`, stop and debug here before writing/running any
training code -- do not silently fall back to CPU training.

## Local smoke-testing on Windows (not for real training)

A separate CPU-only virtual environment (`.venv-smoketest/`, gitignored)
exists on the Windows host for quickly checking that model/loss code is
shape-correct and runs, without needing the full WSL/ROCm stack. It uses
Python 3.12 + CPU-only PyTorch. This is a convenience for iterating on code
only -- actual training must run in the WSL2 ROCm environment described
above.

```powershell
py -3.12 -m venv .venv-smoketest
.venv-smoketest\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-smoketest\Scripts\pip install -r requirements\base.txt
.venv-smoketest\Scripts\python src\models\mobilefacenet.py
.venv-smoketest\Scripts\python src\losses\arcface.py
```

## Status

- [x] Git installed (Git for Windows 2.55.0.3)
- [x] Python 3.12 installed on Windows host (for smoke-testing only)
- [ ] AMD driver updated to 26.2.2+ (admin action required)
- [ ] WSL2 + Ubuntu installed (admin action required)
- [ ] ROCm 7.2.1 installed inside WSL2
- [ ] PyTorch (ROCm build) installed and verified against the GPU
