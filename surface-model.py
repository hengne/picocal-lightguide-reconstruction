"""
3D CAD Geometry Parser and Periodic Bicubic B-Spline Inner Surface Model

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>

Extracts nominal conical frustum and PMT bore parameters from OpenCASCADE / FreeCAD STEP files,
and parameterizes arbitrary inner-wall machining bulges, depressions, and reflective film
wrinkles via a C^2-continuous periodic-in-theta bicubic B-spline surface R(theta, z).
"""

import logging
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"

DEFAULT_INPUT_RADIUS_MM: float = 8.75
DEFAULT_OUTPUT_RADIUS_MM: float = 4.00
DEFAULT_CONE_LENGTH_MM: float = 100.0
DEFAULT_PMT_BORE_RADIUS_MM: float = 8.18
DEFAULT_PMT_BORE_DEPTH_MM: float = 10.0


@dataclass(frozen=True)
class StepConeGeometry:
    """
    Nominal single-channel conical light-guide geometry extracted from the CAD STEP file.
    """

    inputRadiusMm: float
    outputRadiusMm: float
    coneLengthMm: float
    halfConeAngleRad: float
    pmtBoreRadiusMm: float
    pmtBoreDepthMm: float
    cellCentersMm: list[tuple[float, float]]

    @property
    def coneSlope(self) -> float:
        """
        Nominal conical generatrix slope dR/dz.
        """
        return (self.outputRadiusMm - self.inputRadiusMm) / self.coneLengthMm


def parseStepGeometry(stepFilePath: Path) -> StepConeGeometry:
    """
    Parse an ISO-10303 STEP file to automatically extract entrance/exit radii, cone length, and PMT bore radius.
    """
    logger = logging.getLogger(__name__)
    if not stepFilePath.exists():
        logger.warning("STEP file %s not found; falling back to nominal geometry parameters.", stepFilePath)
        return StepConeGeometry(
            inputRadiusMm=DEFAULT_INPUT_RADIUS_MM,
            outputRadiusMm=DEFAULT_OUTPUT_RADIUS_MM,
            coneLengthMm=DEFAULT_CONE_LENGTH_MM,
            halfConeAngleRad=math.atan((DEFAULT_INPUT_RADIUS_MM - DEFAULT_OUTPUT_RADIUS_MM) / DEFAULT_CONE_LENGTH_MM),
            pmtBoreRadiusMm=DEFAULT_PMT_BORE_RADIUS_MM,
            pmtBoreDepthMm=DEFAULT_PMT_BORE_DEPTH_MM,
            cellCentersMm=[(-10.0, -10.0), (10.0, -10.0), (-10.0, 10.0), (10.0, 10.0)],
        )

    text = stepFilePath.read_text(encoding="utf-8", errors="ignore")

    # NOTE: Inspect CIRCLE entities in the STEP file to identify entrance, exit, and PMT counterbore radii.
    circleRadii = sorted({round(float(m), 4) for m in re.findall(r"CIRCLE\s*\([^,]+,\s*([0-9.Ee+-]+)\s*\)", text)})
    conicalMatch = re.search(r"CONICAL_SURFACE\s*\([^,]+,\s*([0-9.Ee+-]+)\s*,\s*([0-9.Ee+-]+)\s*\)", text)

    outRadius = float(conicalMatch.group(1)) if conicalMatch else DEFAULT_OUTPUT_RADIUS_MM
    halfAngle = float(conicalMatch.group(2)) if conicalMatch else math.atan(0.0475)
    inRadius = max(circleRadii) if circleRadii else DEFAULT_INPUT_RADIUS_MM

    boreCandidates = [r for r in circleRadii if outRadius < r < inRadius]
    boreRadius = boreCandidates[0] if boreCandidates else DEFAULT_PMT_BORE_RADIUS_MM

    # NOTE: Derive exact axial reflective cone length from entrance/exit radii and half-cone angle.
    coneLength = round((inRadius - outRadius) / math.tan(halfAngle), 4)

    logger.info(
        "Parsed STEP geometry: R_in=%.2f mm, R_out=%.2f mm, L=%.2f mm, R_pmt_bore=%.2f mm",
        inRadius,
        outRadius,
        coneLength,
        boreRadius,
    )
    return StepConeGeometry(
        inputRadiusMm=inRadius,
        outputRadiusMm=outRadius,
        coneLengthMm=coneLength,
        halfConeAngleRad=halfAngle,
        pmtBoreRadiusMm=boreRadius,
        pmtBoreDepthMm=DEFAULT_PMT_BORE_DEPTH_MM,
        cellCentersMm=[(-10.0, -10.0), (10.0, -10.0), (-10.0, 10.0), (10.0, 10.0)],
    )


class BicubicCylindricalSurface:
    """
    Cylindrical periodic-in-theta bicubic B-spline freeform inner surface model:
        R(theta, z) = R_CAD(z) + deltaR_axi(z) + deltaR_nonAxi(theta, z)

    Guarantees C^2 smoothness across the 2*pi azimuthal seam so analytical surface normals
    remain artifact-free everywhere during multi-bounce specular ray tracing.
    """

    def __init__(
        self,
        geometry: StepConeGeometry,
        axiProfileMm: np.ndarray,
        nonAxiGridMm: np.ndarray,
        device: torch.device,
    ) -> None:
        """
        Initialize the bicubic B-spline surface and cache control weights on the target device.
        """
        self.geometry: StepConeGeometry = geometry
        self.device: torch.device = device
        self.numTheta: int = int(nonAxiGridMm.shape[0])
        self.numZ: int = int(nonAxiGridMm.shape[1])

        # NOTE: Enforce zero azimuthal mean at every axial slice z to decouple axisymmetric and non-axisymmetric modes.
        centeredNonAxi = nonAxiGridMm - np.mean(nonAxiGridMm, axis=0, keepdims=True)
        self.axiProfileMm: np.ndarray = axiProfileMm.astype(np.float32)
        self.nonAxiGridMm: np.ndarray = centeredNonAxi.astype(np.float32)

        totalCtrl = self.nonAxiGridMm + self.axiProfileMm[None, :]
        self.totalCtrlTensor: torch.Tensor = torch.tensor(
            totalCtrl, dtype=torch.float32, device=self.device
        )

    def updateControlGrid(self, axiProfileMm: np.ndarray, nonAxiGridMm: np.ndarray) -> None:
        """
        Update control grid weights in place without allocating new device tensors.
        """
        centeredNonAxi = nonAxiGridMm - np.mean(nonAxiGridMm, axis=0, keepdims=True)
        self.axiProfileMm = axiProfileMm.astype(np.float32)
        self.nonAxiGridMm = centeredNonAxi.astype(np.float32)
        totalCtrl = self.nonAxiGridMm + self.axiProfileMm[None, :]
        self.totalCtrlTensor.copy_(torch.from_numpy(totalCtrl))

    @torch.no_grad()
    def evaluateRadiusAndGradients(
        self, theta: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate surface radius R(theta, z) and analytical partial derivatives (dR/dtheta, dR/dz).
        """
        nTh = self.numTheta
        nZ = self.numZ
        coneLen = self.geometry.coneLengthMm
        slope = self.geometry.coneSlope
        r0Cad = self.geometry.inputRadiusMm
        ctrl = self.totalCtrlTensor

        u = ((theta + math.pi) / (2.0 * math.pi)) * nTh
        uFloor = torch.floor(u)
        fu = u - uFloor
        i0 = uFloor.long() % nTh

        v = torch.clamp(z / coneLen, 0.0, 1.0 - 1e-7) * (nZ - 1)
        vFloor = torch.floor(v)
        fv = v - vFloor
        j0 = vFloor.long()

        fu2 = fu * fu
        fu3 = fu2 * fu
        wu = [
            (1.0 - 3.0 * fu + 3.0 * fu2 - fu3) / 6.0,
            (4.0 - 6.0 * fu2 + 3.0 * fu3) / 6.0,
            (1.0 + 3.0 * fu + 3.0 * fu2 - 3.0 * fu3) / 6.0,
            fu3 / 6.0,
        ]
        dwu = [
            (-3.0 + 6.0 * fu - 3.0 * fu2) / 6.0,
            (-12.0 * fu + 9.0 * fu2) / 6.0,
            (3.0 + 6.0 * fu - 9.0 * fu2) / 6.0,
            3.0 * fu2 / 6.0,
        ]

        fv2 = fv * fv
        fv3 = fv2 * fv
        wv = [
            (1.0 - 3.0 * fv + 3.0 * fv2 - fv3) / 6.0,
            (4.0 - 6.0 * fv2 + 3.0 * fv3) / 6.0,
            (1.0 + 3.0 * fv + 3.0 * fv2 - 3.0 * fv3) / 6.0,
            fv3 / 6.0,
        ]
        dwv = [
            (-3.0 + 6.0 * fv - 3.0 * fv2) / 6.0,
            (-12.0 * fv + 9.0 * fv2) / 6.0,
            (3.0 + 6.0 * fv - 9.0 * fv2) / 6.0,
            3.0 * fv2 / 6.0,
        ]

        idxU = [(i0 - 1) % nTh, i0, (i0 + 1) % nTh, (i0 + 2) % nTh]
        idxV = [torch.clamp(j0 + k - 1, 0, nZ - 1) for k in range(4)]

        val = torch.zeros_like(z)
        dvalDu = torch.zeros_like(z)
        dvalDv = torch.zeros_like(z)

        # NOTE: Use fused in-place addcmul_ operations to keep peak temporary memory negligible.
        for a in range(4):
            iu = idxU[a]
            for b in range(4):
                cVal = ctrl[iu, idxV[b]]
                val.addcmul_(cVal, wu[a] * wv[b])
                dvalDu.addcmul_(cVal, dwu[a] * wv[b])
                dvalDv.addcmul_(cVal, wu[a] * dwv[b])

        radius = r0Cad + slope * z + val
        drDtheta = dvalDu * (nTh / (2.0 * math.pi))
        drDz = slope + dvalDv * ((nZ - 1) / coneLen)
        return radius, drDtheta, drDz

    def sampleDenseGrid(
        self, resTheta: int = 180, resZ: int = 201
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample total inner radius R(theta, z) and non-axisymmetric deformation deltaR_nonAxi(theta, z) on a regular grid.
        """
        theta1d = np.linspace(-math.pi, math.pi, resTheta, endpoint=False, dtype=np.float32)
        z1d = np.linspace(0.0, self.geometry.coneLengthMm, resZ, dtype=np.float32)
        thGrid, zGrid = np.meshgrid(theta1d, z1d, indexing="ij")

        with torch.no_grad():
            thT = torch.tensor(thGrid, dtype=torch.float32, device=self.device)
            zT = torch.tensor(zGrid, dtype=torch.float32, device=self.device)
            rT, _, _ = self.evaluateRadiusAndGradients(thT, zT)
            rNp = rT.cpu().numpy()

        idealR = self.geometry.inputRadiusMm + self.geometry.coneSlope * zGrid
        totalDeltaR = rNp - idealR
        nonAxiDeltaR = totalDeltaR - np.mean(totalDeltaR, axis=0, keepdims=True)
        return theta1d, z1d, rNp, nonAxiDeltaR

    def exportGeant4GridFile(self, outputPath: Path, resTheta: int = 60, resZ: int = 61) -> None:
        """
        Export a cylindrical radius grid text file consumed by the Geant4 G4TessellatedSolid builder.
        """
        _, _, rNp, _ = self.sampleDenseGrid(resTheta=resTheta, resZ=resZ)
        lines = [f"{resTheta} {resZ} {self.geometry.coneLengthMm:.6f}"]
        for i in range(resTheta):
            rowStr = " ".join(f"{rNp[i, j]:.6f}" for j in range(resZ))
            lines.append(rowStr)
        outputPath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def exportBinaryStl(self, outputPath: Path, resTheta: int = 120, resZ: int = 121, wallThicknessMm: float = 1.5) -> None:
        """
        Export the reconstructed deformed light-guide shell as a watertight binary 3D STL solid.
        """
        theta1d, z1d, rInner, _ = self.sampleDenseGrid(resTheta=resTheta, resZ=resZ)
        rOuter = rInner + wallThicknessMm

        def getVertex(rMat: np.ndarray, i: int, j: int) -> np.ndarray:
            ii = i % resTheta
            th = float(theta1d[ii])
            rad = float(rMat[ii, j])
            return np.array([rad * math.cos(th), rad * math.sin(th), float(z1d[j])], dtype=np.float32)

        triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(resTheta):
            for j in range(resZ - 1):
                p00 = getVertex(rInner, i, j)
                p10 = getVertex(rInner, i + 1, j)
                p01 = getVertex(rInner, i, j + 1)
                p11 = getVertex(rInner, i + 1, j + 1)
                triangles.append((p00, p01, p11))
                triangles.append((p00, p11, p10))

                q00 = getVertex(rOuter, i, j)
                q10 = getVertex(rOuter, i + 1, j)
                q01 = getVertex(rOuter, i, j + 1)
                q11 = getVertex(rOuter, i + 1, j + 1)
                triangles.append((q00, q10, q11))
                triangles.append((q00, q11, q01))

            triangles.append((getVertex(rInner, i, 0), getVertex(rInner, i + 1, 0), getVertex(rOuter, i + 1, 0)))
            triangles.append((getVertex(rInner, i, 0), getVertex(rOuter, i + 1, 0), getVertex(rOuter, i, 0)))

            jl = resZ - 1
            triangles.append((getVertex(rInner, i, jl), getVertex(rOuter, i + 1, jl), getVertex(rInner, i + 1, jl)))
            triangles.append((getVertex(rInner, i, jl), getVertex(rOuter, i, jl), getVertex(rOuter, i + 1, jl)))

        with outputPath.open("wb") as stlFile:
            header = b"LHCb PicoCal Reconstructed Deformed Light Guide - Hengne Li".ljust(80, b"\x00")
            stlFile.write(header)
            stlFile.write(struct.pack("<I", len(triangles)))
            for v0, v1, v2 in triangles:
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normLen = float(np.linalg.norm(normal))
                if normLen > 1e-12:
                    normal = normal / normLen
                stlFile.write(struct.pack("<3f3f3f3fH", *normal, *v0, *v1, *v2, 0))
