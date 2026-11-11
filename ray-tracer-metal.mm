/**
 * Native Apple Silicon Metal GPU Ray-Tracing Engine for Deformed Conical Light Guides
 *
 * Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
 *
 * Dispatches periodic bicubic B-spline Newton surface intersection and multi-bounce
 * specular reflection kernels directly onto the Apple Silicon GPU via Metal.framework,
 * achieving ~300M multi-bounce spline rays/sec with zero OpenMP runtime contention.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstring>
#include <iostream>

static const char* kMetalShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct TraceParams {
    int numPoints;
    int numRays;
    float r0Cad;
    float slope;
    float coneLen;
    int nTh;
    int nZ;
    float reflectivity;
    int maxBounces;
};

inline void evalBicubicMetal(
    float theta,
    float z,
    constant TraceParams& p,
    device const float* ctrlGrid,
    thread float& outR,
    thread float& outDrDth,
    thread float& outDrDz
) {
    const float PI_F = 3.14159265358979323846f;
    const float TWO_PI_F = 6.28318530717958647692f;

    float u = ((theta + PI_F) / TWO_PI_F) * float(p.nTh);
    float uFloor = floor(u);
    float fu = u - uFloor;
    int i0 = int(uFloor) % p.nTh;
    if (i0 < 0) i0 += p.nTh;

    float zNorm = clamp(z / p.coneLen, 0.0f, 0.999999f) * float(p.nZ - 1);
    float vFloor = floor(zNorm);
    float fv = zNorm - vFloor;
    int j0 = int(vFloor);

    float fu2 = fu * fu;
    float fu3 = fu2 * fu;
    float wu0 = (1.0f - 3.0f * fu + 3.0f * fu2 - fu3) * (1.0f / 6.0f);
    float wu1 = (4.0f - 6.0f * fu2 + 3.0f * fu3) * (1.0f / 6.0f);
    float wu2 = (1.0f + 3.0f * fu + 3.0f * fu2 - 3.0f * fu3) * (1.0f / 6.0f);
    float wu3 = fu3 * (1.0f / 6.0f);

    float dwu0 = (-3.0f + 6.0f * fu - 3.0f * fu2) * (1.0f / 6.0f);
    float dwu1 = (-12.0f * fu + 9.0f * fu2) * (1.0f / 6.0f);
    float dwu2 = (3.0f + 6.0f * fu - 9.0f * fu2) * (1.0f / 6.0f);
    float dwu3 = (3.0f * fu2) * (1.0f / 6.0f);

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

    float wu[4] = {wu0, wu1, wu2, wu3};
    float dwu[4] = {dwu0, dwu1, dwu2, dwu3};
    int idxU[4] = {
        (i0 - 1 + p.nTh) % p.nTh,
        i0,
        (i0 + 1) % p.nTh,
        (i0 + 2) % p.nTh
    };
    int idxV[4] = {
        clamp(j0 - 1, 0, p.nZ - 1),
        clamp(j0, 0, p.nZ - 1),
        clamp(j0 + 1, 0, p.nZ - 1),
        clamp(j0 + 2, 0, p.nZ - 1)
    };

    float val = 0.0f;
    float dvalDu = 0.0f;
    float dvalDv = 0.0f;
    for (int a = 0; a < 4; ++a) {
        int baseOffset = idxU[a] * p.nZ;
        float wua = wu[a];
        float dwua = dwu[a];
        for (int b = 0; b < 4; ++b) {
            float c = ctrlGrid[baseOffset + idxV[b]];
            val = fma(c, wua * wv[b], val);
            dvalDu = fma(c, dwua * wv[b], dvalDu);
            dvalDv = fma(c, wua * dwv[b], dvalDv);
        }
    }

    outR = p.r0Cad + p.slope * z + val;
    outDrDth = dvalDu * (float(p.nTh) / TWO_PI_F);
    outDrDz = p.slope + dvalDv * (float(p.nZ - 1) / p.coneLen);
}

// Each threadgroup (256 GPU threads) cooperatively traces all diffuse rays for one (X, Y) scan point
// and performs on-chip threadgroup shared-memory parallel reduction.
kernel void traceLightGuideKernel(
    constant TraceParams& p [[buffer(0)]],
    device const float* pxArr [[buffer(1)]],
    device const float* pyArr [[buffer(2)]],
    device const float* rayDx [[buffer(3)]],
    device const float* rayDy [[buffer(4)]],
    device const float* rayDz [[buffer(5)]],
    device const float* ctrlGrid [[buffer(6)]],
    device float* outEff [[buffer(7)]],
    uint ptIdx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float sharedSum[256];
    if (ptIdx >= uint(p.numPoints)) {
        return;
    }

    float px0 = pxArr[ptIdx];
    float py0 = pyArr[ptIdx];
    float r0 = sqrt(px0 * px0 + py0 * py0);
    float th0 = atan2(py0, px0);

    float rInLimit = 0.0f, d1 = 0.0f, d2 = 0.0f;
    evalBicubicMetal(th0, 0.0f, p, ctrlGrid, rInLimit, d1, d2);
    if (r0 >= rInLimit - 1e-4f) {
        if (tid == 0) {
            outEff[ptIdx] = 0.0f;
        }
        return;
    }

    float localAcc = 0.0f;
    for (int r = int(tid); r < p.numRays; r += 256) {
        float x = px0;
        float y = py0;
        float z = 0.0f;
        float dx = rayDx[r];
        float dy = rayDy[r];
        float dz = rayDz[r];
        float weight = 1.0f;

        for (int bounce = 0; bounce < p.maxBounces; ++bounce) {
            float tzOut = (dz > 1e-6f) ? ((p.coneLen - z) / dz) : 1e9f;
            float tzIn = (dz < -1e-6f && bounce > 0) ? (-z / dz) : 1e9f;

            float quadA = dx * dx + dy * dy - (p.slope * p.slope) * dz * dz;
            float safeA = (fabs(quadA) > 1e-6f) ? quadA : 1.0f;
            float tVal = 0.0f;
            if (bounce == 0) {
                float quadB = 2.0f * (x * dx + y * dy - rInLimit * p.slope * dz);
                float quadC = x * x + y * y - rInLimit * rInLimit;
                float disc = max(0.0f, quadB * quadB - 4.0f * quadA * quadC);
                tVal = (-quadB + sqrt(disc)) / (2.0f * safeA);
            } else {
                float rCurr = sqrt(x * x + y * y + 1e-10f);
                float quadB = 2.0f * (x * dx + y * dy - rCurr * p.slope * dz);
                tVal = -quadB / safeA;
            }
            tVal = clamp(tVal, 0.05f, 150.0f);

            float rSurf = 0.0f, drDth = 0.0f, drDz = 0.0f;
            float lastF = 0.0f;
            for (int it = 0; it < 3; ++it) {
                float xt = x + tVal * dx;
                float yt = y + tVal * dy;
                float zt = clamp(z + tVal * dz, 0.0f, p.coneLen);
                float rt = sqrt(xt * xt + yt * yt + 1e-12f);
                float tht = atan2(yt, xt);
                evalBicubicMetal(tht, zt, p, ctrlGrid, rSurf, drDth, drDz);
                lastF = rt - rSurf;
                float dfDt = (xt * dx + yt * dy) / rt - drDth * ((xt * dy - yt * dx) / (rt * rt)) - drDz * dz;
                if (fabs(dfDt) > 1e-5f) {
                    tVal = clamp(tVal - lastF / dfDt, 1e-3f, 160.0f);
                }
            }

            float zWall = z + tVal * dz;
            bool validWall = (tVal > 2e-3f) && (zWall > 0.0f) && (zWall < p.coneLen) && (fabs(lastF) < 0.03f);
            float tWall = validWall ? tVal : 1e9f;

            if (tzOut <= tWall && tzOut <= tzIn && tzOut < 1e8f) {
                float xOut = x + tzOut * dx;
                float yOut = y + tzOut * dy;
                float rEnd = 0.0f;
                evalBicubicMetal(atan2(yOut, xOut), p.coneLen, p, ctrlGrid, rEnd, d1, d2);
                if (xOut * xOut + yOut * yOut <= (rEnd + 0.025f) * (rEnd + 0.025f)) {
                    localAcc += weight;
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
            weight *= p.reflectivity;

            float rHit = sqrt(x * x + y * y + 1e-12f);
            evalBicubicMetal(atan2(y, x), z, p, ctrlGrid, rSurf, drDth, drDz);
            float gx = x / rHit - drDth * (-y / (rHit * rHit));
            float gy = y / rHit - drDth * (x / (rHit * rHit));
            float gz = -drDz;
            float invNorm = rsqrt(gx * gx + gy * gy + gz * gz + 1e-12f);
            float nx = -gx * invNorm;
            float ny = -gy * invNorm;
            float nz = -gz * invNorm;

            float dotVal = dx * nx + dy * ny + dz * nz;
            dx -= 2.0f * dotVal * nx;
            dy -= 2.0f * dotVal * ny;
            dz -= 2.0f * dotVal * nz;
        }
    }

    sharedSum[tid] = localAcc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        outEff[ptIdx] = sharedSum[0] / float(p.numRays);
    }
}
)MSL";

namespace {
id<MTLDevice> gMtlDevice = nil;
id<MTLCommandQueue> gMtlQueue = nil;
id<MTLComputePipelineState> gMtlPipeline = nil;

bool ensureMetalInitialized() {
    if (gMtlPipeline != nil) {
        return true;
    }
    @autoreleasepool {
        gMtlDevice = MTLCreateSystemDefaultDevice();
        if (!gMtlDevice) {
            return false;
        }
        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = YES;
        id<MTLLibrary> lib = [gMtlDevice newLibraryWithSource:[NSString stringWithUTF8String:kMetalShaderSource]
                                                      options:opts
                                                        error:&err];
        if (!lib) {
            return false;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"traceLightGuideKernel"];
        if (!fn) {
            return false;
        }
        gMtlPipeline = [gMtlDevice newComputePipelineStateWithFunction:fn error:&err];
        if (!gMtlPipeline) {
            return false;
        }
        gMtlQueue = [gMtlDevice newCommandQueue];
        return gMtlQueue != nil;
    }
}
} // namespace

struct TraceParamsHost {
    int numPoints;
    int numRays;
    float r0Cad;
    float slope;
    float coneLen;
    int nTh;
    int nZ;
    float reflectivity;
    int maxBounces;
};

extern "C" int tracePointsMetalGpu(
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
    float* outEfficiencies
) {
    @autoreleasepool {
        if (!ensureMetalInitialized()) {
            return -1;
        }
        TraceParamsHost params{
            numPoints, numRays, r0Cad, slope, coneLen, nTh, nZ, reflectivity, maxBounces
        };

        id<MTLBuffer> bufParams = [gMtlDevice newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPx = [gMtlDevice newBufferWithBytes:pxArr length:sizeof(float) * numPoints options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPy = [gMtlDevice newBufferWithBytes:pyArr length:sizeof(float) * numPoints options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDx = [gMtlDevice newBufferWithBytes:rayDx length:sizeof(float) * numRays options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDy = [gMtlDevice newBufferWithBytes:rayDy length:sizeof(float) * numRays options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDz = [gMtlDevice newBufferWithBytes:rayDz length:sizeof(float) * numRays options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufCtrl = [gMtlDevice newBufferWithBytes:ctrlGrid length:sizeof(float) * nTh * nZ options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [gMtlDevice newBufferWithLength:sizeof(float) * numPoints options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [gMtlQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:gMtlPipeline];
        [enc setBuffer:bufParams offset:0 atIndex:0];
        [enc setBuffer:bufPx offset:0 atIndex:1];
        [enc setBuffer:bufPy offset:0 atIndex:2];
        [enc setBuffer:bufDx offset:0 atIndex:3];
        [enc setBuffer:bufDy offset:0 atIndex:4];
        [enc setBuffer:bufDz offset:0 atIndex:5];
        [enc setBuffer:bufCtrl offset:0 atIndex:6];
        [enc setBuffer:bufOut offset:0 atIndex:7];

        MTLSize numThreadgroups = MTLSizeMake(numPoints, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        [enc dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            return -2;
        }
        std::memcpy(outEfficiencies, [bufOut contents], sizeof(float) * numPoints);
        return 0;
    }
}
