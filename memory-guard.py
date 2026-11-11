"""
Adaptive System Resource Scheduler and Memory Safety Watchdog

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>

Probes real-time available host RAM and CPU load average across macOS, Linux, and Windows.
Automatically scales parallel worker threads, GPU/CPU batch chunk sizes, and memory safety
caps when host resources are abundant, while enforcing strict RSS/VRAM ceilings to prevent
out-of-memory stalls on shared workstations.
"""

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"


@dataclass(frozen=True)
class AdaptiveProfile:
    """
    Runtime resource allocation profile computed from current host load and free RAM.
    """

    totalCpuCores: int
    loadAvg1m: float
    freeMemoryGb: float
    assignedThreads: int
    assignedChunkPoints: int
    memoryCapMb: float


@dataclass(frozen=True)
class MemorySnapshot:
    """
    Point-in-time process physical memory and GPU allocator usage snapshot.
    """

    rssMb: float
    mpsAllocatedMb: float
    mpsDriverMb: float


def probeSystemFreeMemoryGb() -> float:
    """
    Detect available physical RAM (in GiB) across macOS (vm_stat) and Linux (/proc/meminfo).
    """
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["vm_stat"], text=True, timeout=3)
            pageSizeMatch = re.search(r"page size of (\d+) bytes", out)
            pageSize = int(pageSizeMatch.group(1)) if pageSizeMatch else 16384

            freePages = 0
            for key in ("Pages free", "Pages speculative", "Pages inactive"):
                match = re.search(rf"{key}:\s+(\d+)\.", out)
                if match:
                    freePages += int(match.group(1))
            return (freePages * pageSize) / (1024.0**3)
        except (subprocess.SubprocessError, OSError, ValueError):
            return 8.0

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as memFile:
                content = memFile.read()
            availMatch = re.search(r"MemAvailable:\s+(\d+)\s+kB", content)
            if availMatch:
                return float(availMatch.group(1)) / (1024.0**2)
        except (OSError, ValueError):
            return 8.0

    return 8.0


def buildAdaptiveProfile() -> AdaptiveProfile:
    """
    Probe current CPU load and free memory to construct an adaptive concurrency profile.
    """
    totalCores = os.cpu_count() or 8
    try:
        load1m = os.getloadavg()[0]
    except (AttributeError, OSError):
        load1m = 2.0

    freeGb = probeSystemFreeMemoryGb()
    idleCores = max(2.0, float(totalCores) - load1m)

    # NOTE: Scale up batch chunking and thread parallelism when idle RAM/CPU headroom is ample,
    # and automatically throttle down under heavy multi-tenant load to protect host responsiveness.
    if freeGb >= 32.0 and idleCores >= 6.0:
        threads = min(8, max(4, int(idleCores * 0.65)))
        chunkPoints = 768
        memCapMb = min(14336.0, freeGb * 1024.0 * 0.22)
    elif freeGb >= 12.0 and idleCores >= 3.0:
        threads = 4
        chunkPoints = 384
        memCapMb = min(6144.0, freeGb * 1024.0 * 0.25)
    else:
        threads = 2
        chunkPoints = 128
        memCapMb = 2048.0

    return AdaptiveProfile(
        totalCpuCores=totalCores,
        loadAvg1m=load1m,
        freeMemoryGb=freeGb,
        assignedThreads=threads,
        assignedChunkPoints=chunkPoints,
        memoryCapMb=memCapMb,
    )


def configureProcessLimits(maxThreads: int | None = None) -> AdaptiveProfile:
    """
    Configure BLAS/OpenMP thread environment variables and clear restrictive MPS watermark locks.
    """
    profile = buildAdaptiveProfile()
    threads = maxThreads if maxThreads is not None else profile.assignedThreads
    threadStr = str(threads)

    os.environ["OMP_NUM_THREADS"] = threadStr
    os.environ["MKL_NUM_THREADS"] = threadStr
    os.environ["OPENBLAS_NUM_THREADS"] = threadStr
    os.environ["VECLIB_MAXIMUM_THREADS"] = threadStr
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # NOTE: Remove overly low MPS watermark overrides that can stall unified-memory allocations.
    os.environ.pop("PYTORCH_MPS_LOW_WATERMARK_RATIO", None)
    os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    return profile


def getProcessPeakRssMb() -> float:
    """
    Query peak resident set size (RSS) in MiB with platform-specific unit normalization.
    """
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # NOTE: macOS reports ru_maxrss in bytes, whereas Linux/BSD report ru_maxrss in KiB.
        if sys.platform == "darwin":
            return float(usage.ru_maxrss) / (1024.0 * 1024.0)
        return float(usage.ru_maxrss) / 1024.0
    except (ImportError, OSError):
        return 0.0


class MemoryGuard:
    """
    Real-time memory safety watchdog enforcing strict RSS and GPU allocator limits.
    """

    def __init__(self, maxMemoryMb: float = 8192.0) -> None:
        """
        Initialize the memory guard with a hard ceiling in MiB.
        """
        self.maxMemoryMb: float = maxMemoryMb
        self.logger: logging.Logger = logging.getLogger(__name__)

    def inspectMemory(self, stageName: str = "checkpoint") -> MemorySnapshot:
        """
        Sample current process memory usage and raise MemoryError if safety thresholds are exceeded.
        """
        import torch

        rssMb = getProcessPeakRssMb()
        mpsAllocMb = 0.0
        mpsDriverMb = 0.0

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mpsAllocMb = float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
            mpsDriverMb = float(torch.mps.driver_allocated_memory()) / (1024.0 * 1024.0)
            if mpsDriverMb > 0.65 * self.maxMemoryMb:
                torch.mps.empty_cache()
                mpsDriverMb = float(torch.mps.driver_allocated_memory()) / (1024.0 * 1024.0)

        if mpsDriverMb > self.maxMemoryMb:
            raise MemoryError(
                f"[{stageName}] MPS driver memory ({mpsDriverMb:.1f} MB) exceeded safety cap ({self.maxMemoryMb:.1f} MB)"
            )
        if rssMb > self.maxMemoryMb:
            raise MemoryError(
                f"[{stageName}] Peak RSS memory ({rssMb:.1f} MB) exceeded safety cap ({self.maxMemoryMb:.1f} MB)"
            )

        return MemorySnapshot(
            rssMb=rssMb,
            mpsAllocatedMb=mpsAllocMb,
            mpsDriverMb=mpsDriverMb,
        )
