"""
Cross-Platform Native Ray-Tracing Accelerator Compiler

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>

Automatically detects the host operating system and C++17 toolchain to compile:
1. On macOS (Apple Silicon / Metal-capable): `libmetal_raytracer.dylib` via Metal.framework;
2. On macOS / Linux / Windows: `libraytracer.{dylib,so,dll}` multi-core C++17 CPU accelerator.
If no C++ compiler is present, the simulation transparently falls back to PyTorch execution.
"""

import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"


def resolveCppCompiler() -> str | None:
    """
    Locate a modern C++17 compiler on the host system, preferring system Xcode clang++ on macOS.
    """
    if sys.platform == "darwin" and Path("/usr/bin/clang++").exists():
        return "/usr/bin/clang++"
    for candidate in ("g++", "clang++", "c++"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def compileAccelerators(projectDir: Path | None = None) -> dict[str, bool]:
    """
    Compile available GPU and multi-core CPU ray-tracing shared libraries for the current host.
    """
    logger = logging.getLogger("agy.compiler")
    rootDir = projectDir if projectDir is not None else Path(__file__).resolve().parent
    results = {"metalGpu": False, "nativeCpu": False}

    # 1. Apple Silicon / macOS Metal GPU Compute Shader library
    metalSrc = rootDir / "ray-tracer-metal.mm"
    metalOut = rootDir / "libmetal_raytracer.dylib"
    if sys.platform == "darwin" and metalSrc.exists():
        clangPath = "/usr/bin/clang++" if Path("/usr/bin/clang++").exists() else shutil.which("clang++")
        if clangPath:
            try:
                subprocess.run(
                    [
                        clangPath,
                        "-O3",
                        "-std=c++17",
                        "-fobjc-arc",
                        "-shared",
                        "-fPIC",
                        str(metalSrc),
                        "-framework",
                        "Metal",
                        "-framework",
                        "Foundation",
                        "-o",
                        str(metalOut),
                    ],
                    check=True,
                    timeout=45,
                )
                results["metalGpu"] = True
                logger.info("Compiled Apple Metal GPU ray-tracing library: %s", metalOut.name)
            except (subprocess.SubprocessError, OSError) as exc:
                logger.warning("Metal GPU library compilation skipped (%s)", exc)

    # 2. Cross-platform C++17 multi-core CPU shared library (macOS / Linux / Windows MinGW)
    nativeSrc = rootDir / "ray-tracer-native.cc"
    if sys.platform == "darwin":
        ext = ".dylib"
    elif sys.platform.startswith("win"):
        ext = ".dll"
    else:
        ext = ".so"

    nativeOut = rootDir / f"libraytracer{ext}"
    cxxBin = resolveCppCompiler()
    if nativeSrc.exists() and cxxBin:
        cmd = [cxxBin, "-O3", "-std=c++17", "-shared"]
        if not sys.platform.startswith("win"):
            cmd.append("-fPIC")
        if sys.platform.startswith("linux"):
            cmd.append("-pthread")
        cmd.extend([str(nativeSrc), "-o", str(nativeOut)])
        try:
            subprocess.run(cmd, check=True, timeout=45)
            results["nativeCpu"] = True
            logger.info(
                "Compiled multi-core C++17 ray-tracing library (%s/%s): %s",
                sys.platform,
                platform.machine(),
                nativeOut.name,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("Native C++ library compilation skipped (%s); PyTorch fallback remains active.", exc)

    return results


compileAllAccelerators = compileAccelerators


def main() -> None:
    """
    CLI entry point for building native ray-tracing accelerators.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    compileAccelerators()


if __name__ == "__main__":
    main()
