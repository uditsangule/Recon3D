from __future__ import annotations

"""
System information helpers for logging experiment/reconstruction environments.

Features:
- OS: name, version, kernel, arch, python
- CPU: model, cores (logical/physical), base/max freq
- RAM: total/available/swap
- GPU(s): vendor, name, driver, vram total/used/free, temps
  * NVIDIA via GPUtil (if installed) or `nvidia-smi`
  * AMD via `rocm-smi`
  * Apple/macOS via `system_profiler SPDisplaysDataType -json`
  * Windows via PowerShell CIM (Win32_VideoController)
  * Linux generic fallback via `lspci` (VGA controllers)

All dependencies are optional. Missing bits are filled with "unknown".

Usage:
    from three_d_recon.utils.system_info import collect_system_info, print_system_report
    info = collect_system_info()
    print_system_report(info)
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------
# Optional dependencies
# ---------------------------
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:
    import GPUtil  # type: ignore
except Exception:  # pragma: no cover
    GPUtil = None

try:
    import distro  # type: ignore
except Exception:  # pragma: no cover
    distro = None


# ---------------------------
# Helpers
# ---------------------------

def _run(cmd: List[str], timeout: float = 3.0) -> Tuple[int, str, str]:
    """
    Run a command returning (returncode, stdout, stderr). Trims output.
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def _bytes_to_human(n: Optional[Union[int, float]]) -> str:
    if n is None:
        return "unknown"
    try:
        n = float(n)
    except Exception:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def _parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


# ---------------------------
# Data models
# ---------------------------

@dataclass
class OSInfo:
    system: str
    release: str
    version: str
    kernel: str
    arch: str
    python: str
    distro: Optional[str] = None


@dataclass
class CPUInfo:
    model: str
    physical_cores: Optional[int]
    logical_cores: Optional[int]
    base_freq_mhz: Optional[float]
    max_freq_mhz: Optional[float]


@dataclass
class RAMInfo:
    total: Optional[int]
    available: Optional[int]
    used: Optional[int]
    percent: Optional[float]
    swap_total: Optional[int]
    swap_used: Optional[int]


@dataclass
class GPUInfo:
    vendor: str
    name: str
    driver: Optional[str]
    vram_total: Optional[int]        # bytes
    vram_used: Optional[int]         # bytes
    vram_free: Optional[int]         # bytes
    temperature: Optional[float]     # C
    details: Dict[str, Any]


@dataclass
class SystemInfo:
    os: OSInfo
    cpu: CPUInfo
    ram: RAMInfo
    gpus: List[GPUInfo]
    extras: Dict[str, Any]


# ---------------------------
# OS
# ---------------------------

def get_os_info() -> OSInfo:
    system = platform.system()
    release = platform.release()
    version = platform.version()
    kernel = platform.uname().release
    arch = platform.machine() or platform.processor() or "unknown"
    py = f"{platform.python_implementation()} {platform.python_version()}"
    dist_str = None

    if system == "Linux":
        # Prefer python 3.10+ method; fallback to 'distro' package.
        try:
            osr = platform.freedesktop_os_release()
            dist_str = f"{osr.get('NAME','Linux')} {osr.get('VERSION','')}".strip()
        except Exception:
            if distro:
                try:
                    dist_str = " ".join(filter(None, [distro.name(pretty=True), distro.version()]))
                except Exception:
                    pass
    elif system == "Darwin":
        # macOS product version
        code, out, _ = _run(["sw_vers", "-productVersion"])
        if code == 0 and out:
            dist_str = f"macOS {out}"
        else:
            dist_str = "macOS"
    elif system == "Windows":
        dist_str = f"Windows {platform.version()}"

    return OSInfo(system=system, release=release, version=version, kernel=kernel, arch=arch, python=py, distro=dist_str)


# ---------------------------
# CPU
# ---------------------------

def get_cpu_info() -> CPUInfo:
    model = platform.processor() or ""
    if not model:
        # Try /proc/cpuinfo on Linux
        if os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    txt = f.read()
                m = re.search(r"model name\s+:\s+(.+)", txt)
                if m:
                    model = m.group(1).strip()
            except Exception:
                pass
    if not model and platform.system() == "Darwin":
        code, out, _ = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if code == 0 and out:
            model = out

    physical = None
    logical = None
    base_mhz = None
    max_mhz = None

    if psutil:
        try:
            physical = psutil.cpu_count(logical=False)
            logical = psutil.cpu_count(logical=True)
        except Exception:
            pass
        try:
            freq = psutil.cpu_freq()
            if freq:
                base_mhz = float(freq.min) if freq.min else None
                max_mhz = float(freq.max) if freq.max else None
        except Exception:
            pass
    else:
        logical = os.cpu_count()

    return CPUInfo(model=model or "unknown",
                   physical_cores=physical, logical_cores=logical,
                   base_freq_mhz=base_mhz, max_freq_mhz=max_mhz)


# ---------------------------
# RAM
# ---------------------------

def get_ram_info() -> RAMInfo:
    total = available = used = percent = swap_total = swap_used = None
    if psutil:
        try:
            vm = psutil.virtual_memory()
            total, available, used, percent = vm.total, vm.available, vm.used, float(vm.percent)
        except Exception:
            pass
        try:
            sm = psutil.swap_memory()
            swap_total, swap_used = sm.total, sm.used
        except Exception:
            pass
    else:
        # Fallback for POSIX
        if hasattr(os, "sysconf"):
            try:
                pagesize = os.sysconf("SC_PAGE_SIZE")
                phys_pages = os.sysconf("SC_PHYS_PAGES")
                total = pagesize * phys_pages
            except Exception:
                pass
    return RAMInfo(total=total, available=available, used=used, percent=percent,
                   swap_total=swap_total, swap_used=swap_used)


# ---------------------------
# GPU: NVIDIA (GPUtil or nvidia-smi)
# ---------------------------

def _gpus_via_gputil() -> List[GPUInfo]:
    out: List[GPUInfo] = []
    if not GPUtil:
        return out
    try:
        for g in GPUtil.getGPUs():
            total = int(g.memoryTotal * 1024 * 1024)
            used = int(g.memoryUsed * 1024 * 1024)
            free = int(g.memoryFree * 1024 * 1024)
            out.append(GPUInfo(
                vendor="NVIDIA",
                name=str(g.name),
                driver=None,  # GPUtil doesn't expose driver
                vram_total=total,
                vram_used=used,
                vram_free=free,
                temperature=float(g.temperature) if g.temperature is not None else None,
                details={"uuid": getattr(g, "uuid", None), "load": getattr(g, "load", None)}
            ))
    except Exception:
        pass
    return out


def _gpus_via_nvidia_smi() -> List[GPUInfo]:
    out: List[GPUInfo] = []
    if shutil.which("nvidia-smi") is None:
        return out

    # Query in CSV (no units) to simplify parsing
    fields = [
        "name", "driver_version", "memory.total", "memory.used", "memory.free",
        "temperature.gpu"
    ]
    cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    code, stdout, _ = _run(cmd, timeout=4.0)
    if code != 0 or not stdout:
        return out

    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        name, driver, mem_total, mem_used, mem_free, temp_c = parts
        out.append(GPUInfo(
            vendor="NVIDIA",
            name=name,
            driver=driver,
            vram_total=int(float(mem_total)) * 1024 * 1024,
            vram_used=int(float(mem_used)) * 1024 * 1024,
            vram_free=int(float(mem_free)) * 1024 * 1024,
            temperature=float(temp_c) if temp_c else None,
            details={}
        ))
    return out


# ---------------------------
# GPU: AMD (rocm-smi)
# ---------------------------

def _gpus_via_rocm_smi() -> List[GPUInfo]:
    out: List[GPUInfo] = []
    if shutil.which("rocm-smi") is None:
        return out

    # Try JSON first (newer rocm-smi)
    code, stdout, _ = _run(["rocm-smi", "--showmeminfo", "vram", "--json"], timeout=5.0)
    if code == 0 and stdout:
        try:
            data = json.loads(stdout)
            # data structure varies; handle common forms
            for dev, info in data.items():
                name = info.get("Card series") or info.get("Card SKU") or "AMD GPU"
                driver = info.get("Driver version")
                mem = info.get("VRAM Total Memory (B)") or info.get("VRAM Total Memory (bytes)")
                used = info.get("VRAM Used Memory (B)") or info.get("VRAM Used Memory (bytes)")
                free = None if mem is None or used is None else int(mem) - int(used)
                out.append(GPUInfo(
                    vendor="AMD",
                    name=str(name),
                    driver=str(driver) if driver else None,
                    vram_total=_parse_int(mem),
                    vram_used=_parse_int(used),
                    vram_free=free,
                    temperature=None,
                    details={"device": dev}
                ))
            return out
        except Exception:
            pass

    # Fallback: parse text output
    code, stdout, _ = _run(["rocm-smi"], timeout=5.0)
    if code != 0 or not stdout:
        return out

    # Heuristic parsing
    names = re.findall(r"GPU\[\d+\].*?:\s*(.*)", stdout)
    totals = re.findall(r"VRAM Total.*?:\s*([\d\.]+)\s*([GM]B)", stdout, flags=re.I)
    useds = re.findall(r"VRAM Used.*?:\s*([\d\.]+)\s*([GM]B)", stdout, flags=re.I)
    for idx, nm in enumerate(names):
        def conv(pair_list, k):
            try:
                v, unit = pair_list[k]
                v = float(v)
                unit = unit.upper()
                if unit == "GB":
                    v *= 1024**3
                elif unit == "MB":
                    v *= 1024**2
                return int(v)
            except Exception:
                return None
        total = conv(totals, idx)
        used = conv(useds, idx)
        free = None if (total is None or used is None) else total - used
        out.append(GPUInfo(
            vendor="AMD", name=nm.strip(), driver=None,
            vram_total=total, vram_used=used, vram_free=free,
            temperature=None, details={}
        ))
    return out


# ---------------------------
# GPU: macOS system_profiler
# ---------------------------

def _gpus_via_macos_system_profiler() -> List[GPUInfo]:
    out: List[GPUInfo] = []
    if platform.system() != "Darwin":
        return out

    cmd = ["system_profiler", "SPDisplaysDataType", "-json"]
    code, stdout, _ = _run(cmd, timeout=8.0)
    if code != 0 or not stdout:
        return out

    try:
        data = json.loads(stdout)
        gpus = data.get("SPDisplaysDataType", [])
        for g in gpus:
            name = g.get("_name", "GPU")
            vram = g.get("spdisplays_vram")
            # vram often like "8 GB"; parse number
            vram_bytes = None
            if isinstance(vram, str):
                m = re.search(r"([\d\.]+)\s*(GB|MB)", vram, flags=re.I)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2).upper()
                    vram_bytes = int(val * (1024**3 if unit == "GB" else 1024**2))
            out.append(GPUInfo(
                vendor="Apple/AMD/NVIDIA",
                name=name,
                driver=None,
                vram_total=vram_bytes,
                vram_used=None,
                vram_free=None,
                temperature=None,
                details={k: v for k, v in g.items() if k.startswith("spdisplays_")}
            ))
    except Exception:
        pass
    return out


# ---------------------------
# GPU: Windows PowerShell CIM
# ---------------------------

def _gpus_via_windows_cim() -> List[GPUInfo]:
    out: List[GPUInfo] = []
    if platform.system() != "Windows":
        return out

    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return out
    script = r"Get-CimInstance Win32_VideoController | " \
             r"Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json"
    code, stdout, _ = _run([powershell, "-NoProfile", "-Command", script], timeout=6.0)
    if code != 0 or not stdout:
        return out
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            data = [data]
        for g in data:
            name = g.get("Name", "GPU")
            driver = g.get("DriverVersion")
            ram = g.get("AdapterRAM")
            out.append(GPUInfo(
                vendor="Unknown",
                name=name,
                driver=str(driver) if driver else None,
                vram_total=int(ram) if isinstance(ram, int) else None,
                vram_used=None,
                vram_free=None,
                temperature=None,
                details={}
            ))
    except Exception:
        pass
    return out


# ---------------------------
# GPU: Linux lspci (generic hint)
# ---------------------------

def _gpus_via_lspci() -> List[GPUInfo]:
    out: List[GPUInfo] = []
    if platform.system() != "Linux" or shutil.which("lspci") is None:
        return out

    code, stdout, _ = _run(["lspci", "-nn"], timeout=4.0)
    if code != 0 or not stdout:
        return out
    for line in stdout.splitlines():
        if " VGA " in line or " 3D " in line:
            # Example: "01:00.0 VGA compatible controller: NVIDIA Corporation ... (rev a1)"
            name = line.split(":", 2)[-1].strip()
            vendor = "NVIDIA" if "NVIDIA" in line else ("AMD" if "AMD" in line or "ATI" in line else ("Intel" if "Intel" in line else "Unknown"))
            out.append(GPUInfo(
                vendor=vendor, name=name, driver=None,
                vram_total=None, vram_used=None, vram_free=None,
                temperature=None, details={}
            ))
    return out


# ---------------------------
# GPU aggregator
# ---------------------------

def get_gpu_info() -> List[GPUInfo]:
    # Try most informative sources first
    gpus: List[GPUInfo] = []
    gpus += _gpus_via_gputil()
    if gpus:
        return gpus

    gpus += _gpus_via_nvidia_smi()
    if gpus:
        return gpus

    gpus += _gpus_via_rocm_smi()
    if gpus:
        return gpus

    gpus += _gpus_via_macos_system_profiler()
    if gpus:
        return gpus

    # Windows CIM
    gpus += _gpus_via_windows_cim()
    if gpus:
        return gpus

    # Last resort generic hint
    gpus += _gpus_via_lspci()
    return gpus


# ---------------------------
# Collector / Reporter
# ---------------------------

def collect_system_info() -> SystemInfo:
    os_info = get_os_info()
    cpu_info = get_cpu_info()
    ram_info = get_ram_info()
    gpus = get_gpu_info()

    extras: Dict[str, Any] = {}
    # CUDA env hints
    extras["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    # NVCC version
    if shutil.which("nvcc"):
        code, out, _ = _run(["nvcc", "--version"], timeout=4.0)
        if code == 0 and out:
            m = re.search(r"release (\d+\.\d+)", out)
            if m:
                extras["CUDA"] = m.group(1)
            else:
                extras["CUDA"] = out.splitlines()[-1].strip()

    # PyTorch/CUDA runtime (optional)
    try:
        import torch  # type: ignore
        extras["torch"] = torch.__version__
        if torch.cuda.is_available():
            extras["torch_cuda"] = torch.version.cuda
            extras["torch_devices"] = torch.cuda.device_count()
    except Exception:
        pass

    return SystemInfo(os=os_info, cpu=cpu_info, ram=ram_info, gpus=gpus, extras=extras)


def system_info_to_dict(info: SystemInfo) -> Dict[str, Any]:
    d = asdict(info)
    # Convert bytes to human for readability, but keep raw in 'raw_*'
    for key in ["total", "available", "used", "swap_total", "swap_used"]:
        raw = d["ram"].get(key)
        d["ram"][f"raw_{key}"] = raw
        d["ram"][key] = _bytes_to_human(raw)

    for g in d["gpus"]:
        for k in ["vram_total", "vram_used", "vram_free"]:
            raw = g.get(k)
            g[f"raw_{k}"] = raw
            g[k] = _bytes_to_human(raw)
    return d


def print_system_report(info: Optional[SystemInfo] = None) -> None:
    """
    Pretty-prints a human-readable report to stdout.
    """
    if info is None:
        info = collect_system_info()
    d = system_info_to_dict(info)

    # OS
    osd = d["os"]
    print("=== OS ===")
    print(f"{osd['system']} {osd.get('distro') or ''}".strip())
    print(f"Kernel: {osd['kernel']}  Arch: {osd['arch']}  Python: {osd['python']}")
    print()

    # CPU
    c = d["cpu"]
    print("=== CPU ===")
    print(f"Model: {c['model']}")
    print(f"Cores: physical={c['physical_cores']} logical={c['logical_cores']}")
    base = f"{c['base_freq_mhz']} MHz" if c['base_freq_mhz'] else "unknown"
    mx = f"{c['max_freq_mhz']} MHz" if c['max_freq_mhz'] else "unknown"
    print(f"Freq: base={base} max={mx}")
    print()

    # RAM
    r = d["ram"]
    print("=== RAM ===")
    print(f"Total: {r['total']}  Used: {r['used']}  Free: {r['available']}  Swap: {r['swap_used']}/{r['swap_total']}")
    print()

    # GPUs
    print("=== GPUs ===")
    if not d["gpus"]:
        print("No GPUs detected.")
    else:
        for i, g in enumerate(d["gpus"]):
            print(f"[{i}] {g['vendor']} {g['name']}")
            print(f"    Driver: {g['driver'] or 'unknown'}")
            print(f"    VRAM: total={g['vram_total']} used={g['vram_used']} free={g['vram_free']}")
            temp = g.get("temperature")
            print(f"    Temp: {temp if temp is not None else 'unknown'} °C")
    print()

    # Extras
    print("=== Extras ===")
    for k, v in d["extras"].items():
        print(f"{k}: {v}")


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    info = collect_system_info()
    print_system_report(info)
    # Also emit JSON (machine-readable)
    as_json = json.dumps(system_info_to_dict(info), indent=2)
    print("\n=== JSON ===")
    print(as_json)


if __name__ == "__main__":
    print_system_report()
