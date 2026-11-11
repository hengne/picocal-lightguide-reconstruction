/**
 * Cross-Platform Multi-Core C++17 Ray-Tracing Kernel for Bicubic B-Spline Light Guides
 *
 * Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
 *
 * Implements lock-free dynamic work stealing across CPU cores using standard C++17 threads
 * and atomics, compiling cleanly on Linux, macOS, and Windows without external dependencies.
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

namespace {

constexpr float PI_F = 3.14159265358979323846f;
constexpr float TWO_PI_F = 6.28318530717958647692f;

inline void evalBicubicSurface(
    float theta,
    float z,
    float r0Cad,
    float slope,
    float coneLen,
    int nTh,
    int nZ,
    const float* ctrlGrid,
    float& outR,
    float& outDrDth,
    float& outDrDz
) {
    float u = ((theta + PI_F) / TWO_PI_F) * static_cast<float>(nTh);
    float uFloor = std::floor(u);
    float fu = u - uFloor;
    int i0 = static_cast<int>(uFloor) % nTh;
    if (i0 < 0) {
        i0 += nTh;
    }

    float zNorm = std::clamp(z / coneLen, 0.0f, 0.9999999f) * static_cast<float>(nZ - 1);
    float vFloor = std::floor(zNorm);
    float fv = zNorm - vFloor;
    int j0 = static_cast<int>(vFloor);

    float fu2 = fu * fu;
    float fu3 = fu2 * fu;
    float wu[4] = {
        (1.0f - 3.0f * fu + 3.0f * fu2 - fu3) * (1.0f / 6.0f),
        (4.0f - 6.0f * fu2 + 3.0f * fu3) * (1.0f / 6.0f),
        (1.0f + 3.0f * fu + 3.0f * fu2 - 3.0f * fu3) * (1.0f / 6.0f),
        fu3 * (1.0f / 6.0f)
    };
    float dwu[4] = {
        (-3.0f + 6.0f * fu - 3.0f * fu2) * (1.0f / 6.0f),
        (-12.0f * fu + 9.0f * fu2) * (1.0f / 6.0f),
        (3.0f + 6.0f * fu - 9.0f * fu2) * (1.0f / 6.0f),
        (3.0f * fu2) * (1.0f / 6.0f)
    };

    float fv2 = fv * fv;
    float fv3 = fv2 * fv;
    float wv[4] = {
        (1.0f - 3.0f * fv + 3.0f * fv2 - fv3) * (1.0f / 6.0f),
        (4.0f - 6.0f * fv2 + 3.0f * fv3) * (1.0f / 6.0f),
        (1.0f + 3.0f * fv + 3.0f * fv2 - 3.0f * fv3) * (1.0f / 6.0f),
        fv3 * (1.0f / 6.0f)
    };
    float dwv[4] = {
        (-3.0f + 6.0f * fv - 3.0f * fv2) * (1.0f / 6.0f),
        (-12.0f * fv + 9.0f * fv2) * (1.0f / 6.0f),
        (3.0f + 6.0f * fv - 9.0f * fv2) * (1.0f / 6.0f),
        (3.0f * fv2) * (1.0f / 6.0f)
    };

    int idxU[4] = {
        (i0 - 1 + nTh) % nTh,
        i0,
        (i0 + 1) % nTh,
        (i0 + 2) % nTh
    };
    int idxV[4] = {
        std::clamp(j0 - 1, 0, nZ - 1),
        std::clamp(j0, 0, nZ - 1),
        std::clamp(j0 + 1, 0, nZ - 1),
        std::clamp(j0 + 2, 0, nZ - 1)
    };

    float val = 0.0f;
    float dvalDu = 0.0f;
    float dvalDv = 0.0f;
    for (int a = 0; a < 4; ++a) {
        const float* rowPtr = ctrlGrid + idxU[a] * nZ;
        float wua = wu[a];
        float dwua = dwu[a];
        for (int b = 0; b < 4; ++b) {
            float c = rowPtr[idxV[b]];
            val += c * wua * wv[b];
            dvalDu += c * dwua * wv[b];
            dvalDv += c * wua * dwv[b];
        }
    }

    outR = r0Cad + slope * z + val;
    outDrDth = dvalDu * (static_cast<float>(nTh) / TWO_PI_F);
    outDrDz = slope + dvalDv * (static_cast<float>(nZ - 1) / coneLen);
}

} // namespace

extern "C" void tracePointsParallel(
    int numPoints,
    const float* pxArr,
    const float* pyArr,
    int numRays,
    const float* rayDx,
    const float* rayDy,
    const float* rayDz,
    float r0Cad,
    float slope,
    float coneLen,
    int nTh,
    int nZ,
    const float* ctrlGrid,
    float reflectivity,
    int maxBounces,
    int numThreads,
    float* outEfficiencies
) {
    // NOTE: Dynamically dispatch 8 scan points per atomic fetch to avoid thread load imbalance
    // caused by zero-cost exterior corner pixels outside the circular entrance aperture.
    std::atomic<int> nextIndex{0};
    int workerCount = std::max(1, std::min(numThreads, numPoints));
    std::vector<std::thread> workers;
    workers.reserve(workerCount);

    auto workerFn = [&]() {
        while (true) {
            int basePt = nextIndex.fetch_add(8, std::memory_order_relaxed);
            if (basePt >= numPoints) {
                break;
            }
            int limitPt = std::min(numPoints, basePt + 8);
            for (int p = basePt; p < limitPt; ++p) {
                float px0 = pxArr[p];
                float py0 = pyArr[p];
                float r0 = std::sqrt(px0 * px0 + py0 * py0);
                float th0 = std::atan2(py0, px0);

                float rInLimit = 0.0f, tmp1 = 0.0f, tmp2 = 0.0f;
                evalBicubicSurface(th0, 0.0f, r0Cad, slope, coneLen, nTh, nZ, ctrlGrid, rInLimit, tmp1, tmp2);
                if (r0 >= rInLimit - 1e-4f) {
                    outEfficiencies[p] = 0.0f;
                    continue;
                }

                double accSum = 0.0;
                for (int r = 0; r < numRays; ++r) {
                    float x = px0;
                    float y = py0;
                    float z = 0.0f;
                    float dx = rayDx[r];
                    float dy = rayDy[r];
                    float dz = rayDz[r];
                    float weight = 1.0f;

                    for (int bounce = 0; bounce < maxBounces; ++bounce) {
                        float tzOut = (dz > 1e-6f) ? ((coneLen - z) / dz) : 1e9f;
                        float tzIn = (dz < -1e-6f && bounce > 0) ? (-z / dz) : 1e9f;

                        float tVal = 0.0f;
                        float quadA = dx * dx + dy * dy - (slope * slope) * dz * dz;
                        float safeA = (std::fabs(quadA) > 1e-6f) ? quadA : 1.0f;
                        if (bounce == 0) {
                            float quadB = 2.0f * (x * dx + y * dy - rInLimit * slope * dz);
                            float quadC = x * x + y * y - rInLimit * rInLimit;
                            float disc = std::max(0.0f, quadB * quadB - 4.0f * quadA * quadC);
                            tVal = (-quadB + std::sqrt(disc)) / (2.0f * safeA);
                        } else {
                            float rCurr = std::sqrt(x * x + y * y + 1e-10f);
                            float quadB = 2.0f * (x * dx + y * dy - rCurr * slope * dz);
                            tVal = -quadB / safeA;
                        }
                        tVal = std::clamp(tVal, 0.05f, 150.0f);

                        float rSurf = 0.0f, drDth = 0.0f, drDz = 0.0f;
                        float lastF = 0.0f;
                        for (int iter = 0; iter < 3; ++iter) {
                            float xt = x + tVal * dx;
                            float yt = y + tVal * dy;
                            float zt = std::clamp(z + tVal * dz, 0.0f, coneLen);
                            float rt = std::sqrt(xt * xt + yt * yt + 1e-12f);
                            float tht = std::atan2(yt, xt);
                            evalBicubicSurface(tht, zt, r0Cad, slope, coneLen, nTh, nZ, ctrlGrid, rSurf, drDth, drDz);
                            lastF = rt - rSurf;
                            float dfDt = (xt * dx + yt * dy) / rt - drDth * ((xt * dy - yt * dx) / (rt * rt)) - drDz * dz;
                            if (std::fabs(dfDt) > 1e-5f) {
                                tVal = std::clamp(tVal - lastF / dfDt, 1e-3f, 160.0f);
                            }
                        }

                        float zWall = z + tVal * dz;
                        bool validWall = (tVal > 2e-3f) && (zWall > 0.0f) && (zWall < coneLen) && (std::fabs(lastF) < 0.03f);
                        float tWall = validWall ? tVal : 1e9f;

                        if (tzOut <= tWall && tzOut <= tzIn && tzOut < 1e8f) {
                            float xOut = x + tzOut * dx;
                            float yOut = y + tzOut * dy;
                            float rEnd = 0.0f;
                            evalBicubicSurface(
                                std::atan2(yOut, xOut), coneLen, r0Cad, slope, coneLen, nTh, nZ, ctrlGrid, rEnd, tmp1, tmp2
                            );
                            if (xOut * xOut + yOut * yOut <= (rEnd + 0.025f) * (rEnd + 0.025f)) {
                                accSum += static_cast<double>(weight);
                            }
                            break;
                        }
                        if (tzIn <= tWall && tzIn < 1e8f) {
                            break;
                        }
                        if (tWall >= 1e8f) {
                            break;
                        }

                        x += tWall * dx;
                        y += tWall * dy;
                        z += tWall * dz;
                        weight *= reflectivity;

                        float rHit = std::sqrt(x * x + y * y + 1e-12f);
                        evalBicubicSurface(
                            std::atan2(y, x), z, r0Cad, slope, coneLen, nTh, nZ, ctrlGrid, rSurf, drDth, drDz
                        );
                        float gx = x / rHit - drDth * (-y / (rHit * rHit));
                        float gy = y / rHit - drDth * (x / (rHit * rHit));
                        float gz = -drDz;
                        float invNorm = 1.0f / std::sqrt(gx * gx + gy * gy + gz * gz + 1e-12f);
                        float nx = -gx * invNorm;
                        float ny = -gy * invNorm;
                        float nz = -gz * invNorm;

                        float dot = dx * nx + dy * ny + dz * nz;
                        dx -= 2.0f * dot * nx;
                        dy -= 2.0f * dot * ny;
                        dz -= 2.0f * dot * nz;
                    }
                }
                outEfficiencies[p] = static_cast<float>(accSum / static_cast<double>(numRays));
            }
        }
    };

    for (int t = 0; t < workerCount; ++t) {
        workers.emplace_back(workerFn);
    }
    for (auto& th : workers) {
        th.join();
    }
}
