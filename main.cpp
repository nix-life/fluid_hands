#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <algorithm>

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "net.h"
#ifdef _OPENMP
#include <omp.h>
#endif

constexpr int IX(int x, int y, int z, int N) {
    return x + y * N + z * N * N;
}

struct FluidCube {
    int size;
    float dt;
    float diff;
    float visc;
    int iter = 4;
    float buoyancy = 0.03f;
    float heatBuoyancy = 0.14f;
    float cooling = 0.97f;
    float dissipation = 0.985f;
    float vorticity = 0.05f;
    float exposure = 0.8f;
    float maxVel = 0.5f;
    float upX = 0.0f, upY = 1.0f, upZ = 0.0f;
    bool closedTop = false;
    float targetMass = 0.0f;
    float sharpen = 0.0f;

    bool useCG = true;
    int advMode = 1;
    bool measure = false;
    float cgTol = 3e-2f;
    int cgMaxIter = 40;
    int lastCgIters = 0;
    float lastCgRelRes = 0.0f;
    float divRmsBefore = 0.0f;
    float divRmsAfter = 0.0f;

    std::vector<float> s;
    std::vector<float> density;
    std::vector<float> heat;
    std::vector<float> heat0;
    std::vector<float> curlX;
    std::vector<float> curlY;
    std::vector<float> curlZ;
    std::vector<float> curlM;
    std::vector<float> pressA;
    std::vector<float> pressB;
    std::vector<float> cgR;
    std::vector<float> cgZ;
    std::vector<float> cgD;
    std::vector<float> cgQ;
    std::vector<float> macHat;
    std::vector<float> macBack;
    std::vector<float> diagP;
    std::vector<float> invDiagP;

    std::vector<float> Vx;
    std::vector<float> Vy;
    std::vector<float> Vz;

    std::vector<float> Vx0;
    std::vector<float> Vy0;
    std::vector<float> Vz0;

    FluidCube(int size, float dt, float diff, float visc)
        : size(size), dt(dt), diff(diff), visc(visc),
          s(size * size * size, 0.0f),
          density(size * size * size, 0.0f),
          heat(size * size * size, 0.0f),
          heat0(size * size * size, 0.0f),
          curlX(size * size * size, 0.0f),
          curlY(size * size * size, 0.0f),
          curlZ(size * size * size, 0.0f),
          curlM(size * size * size, 0.0f),
          pressA(size * size * size, 0.0f),
          pressB(size * size * size, 0.0f),
          cgR(size * size * size, 0.0f),
          cgZ(size * size * size, 0.0f),
          cgD(size * size * size, 0.0f),
          cgQ(size * size * size, 0.0f),
          macHat(size * size * size, 0.0f),
          macBack(size * size * size, 0.0f),
          Vx(size * size * size, 0.0f),
          Vy(size * size * size, 0.0f),
          Vz(size * size * size, 0.0f),
          Vx0(size * size * size, 0.0f),
          Vy0(size * size * size, 0.0f),
          Vz0(size * size * size, 0.0f),
          diagP(size * size * size, 0.0f),
          invDiagP(size * size * size, 0.0f)
    {
        computeDiag();
    }

    void computeDiag() {
        std::fill(diagP.begin(), diagP.end(), 0.0f);
        std::fill(invDiagP.begin(), invDiagP.end(), 0.0f);
        for (int z = 1; z < size - 1; z++)
            for (int y = 1; y < size - 1; y++)
                for (int x = 1; x < size - 1; x++) {
                    int dg = 6;
                    if (x == 1) dg--; if (x == size - 2) dg--;
                    if (y == 1) dg--;
                    if (closedTop && y == size - 2) dg--;
                    if (z == 1) dg--; if (z == size - 2) dg--;
                    int id = IX(x, y, z, size);
                    diagP[id] = (float)dg;
                    invDiagP[id] = 1.0f / (float)dg;
                }
    }

    void setClosedTop(bool c) {
        closedTop = c;
        computeDiag();
    }

    void refill(float ux, float uy, float uz, float level) {
        int N = size, NN = N * N;
        std::fill(Vx.begin(), Vx.end(), 0.0f);
        std::fill(Vy.begin(), Vy.end(), 0.0f);
        std::fill(Vz.begin(), Vz.end(), 0.0f);
        std::fill(pressA.begin(), pressA.end(), 0.0f);
        std::fill(pressB.begin(), pressB.end(), 0.0f);
        std::fill(heat.begin(), heat.end(), 0.0f);
        std::fill(heat0.begin(), heat0.end(), 0.0f);
        std::fill(density.begin(), density.end(), 0.0f);
        std::vector<std::pair<float, int>> hs;
        hs.reserve((N - 2) * (N - 2) * (N - 2));
        for (int z = 1; z < N - 1; z++)
            for (int y = 1; y < N - 1; y++)
                for (int x = 1; x < N - 1; x++) {
                    float rx = (x + 0.5f) / N - 0.5f;
                    float ry = (y + 0.5f) / N - 0.5f;
                    float rz = (z + 0.5f) / N - 0.5f;
                    hs.push_back({rx * ux + ry * uy + rz * uz, z * NN + y * N + x});
                }
        int k = (int)std::lround(level * hs.size());
        k = std::min(std::max(k, 1), (int)hs.size());
        std::nth_element(hs.begin(), hs.begin() + k, hs.end(),
                         [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                             return a.first < b.first;
                         });
        for (int i = 0; i < k; i++) density[hs[i].second] = 1.0f;
        targetMass = 0.0f;
        for (float d : density) targetMass += d;
    }

    void addDensity(int x, int y, int z, float amount) {
        int N = size;
        if (x < 0 || y < 0 || z < 0 || x >= N || y >= N || z >= N) return;
        density[IX(x, y, z, N)] += amount;
    }

    void addVelocity(int x, int y, int z, float ax, float ay, float az) {
        int N = size;
        if (x < 0 || y < 0 || z < 0 || x >= N || y >= N || z >= N) return;
        int i = IX(x, y, z, N);
        Vx[i] += ax; Vy[i] += ay; Vz[i] += az;
    }

    void addDensitySphere(float cx, float cy, float cz, float r, float amount) {
        int x0 = (int)std::floor(cx - r), x1 = (int)std::ceil(cx + r);
        int y0 = (int)std::floor(cy - r), y1 = (int)std::ceil(cy + r);
        int z0 = (int)std::floor(cz - r), z1 = (int)std::ceil(cz + r);
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    float dx = x - cx, dy = y - cy, dz = z - cz;
                    float d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 <= r * r) addDensity(x, y, z, amount * (1.0f - d2 / (r * r + 1e-6f)));
                }
    }

    void addVelocitySphere(float cx, float cy, float cz, float r, float ax, float ay, float az) {
        int x0 = (int)std::floor(cx - r), x1 = (int)std::ceil(cx + r);
        int y0 = (int)std::floor(cy - r), y1 = (int)std::ceil(cy + r);
        int z0 = (int)std::floor(cz - r), z1 = (int)std::ceil(cz + r);
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    float dx = x - cx, dy = y - cy, dz = z - cz;
                    if (dx * dx + dy * dy + dz * dz <= r * r) addVelocity(x, y, z, ax, ay, az);
                }
    }

    void reset() {
        std::fill(density.begin(), density.end(), 0.0f);
        std::fill(Vx.begin(), Vx.end(), 0.0f);
        std::fill(Vy.begin(), Vy.end(), 0.0f);
        std::fill(Vz.begin(), Vz.end(), 0.0f);
        std::fill(heat.begin(), heat.end(), 0.0f);
        std::fill(heat0.begin(), heat0.end(), 0.0f);
        std::fill(pressA.begin(), pressA.end(), 0.0f);
        std::fill(pressB.begin(), pressB.end(), 0.0f);
    }

    void blendVelocitySphere(float cx, float cy, float cz, float r,
                             float ax, float ay, float az, float t) {
        int N = size;
        int x0 = (int)std::floor(cx - r), x1 = (int)std::ceil(cx + r);
        int y0 = (int)std::floor(cy - r), y1 = (int)std::ceil(cy + r);
        int z0 = (int)std::floor(cz - r), z1 = (int)std::ceil(cz + r);
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    if (x < 0 || y < 0 || z < 0 || x >= N || y >= N || z >= N) continue;
                    float dx = x - cx, dy = y - cy, dz = z - cz;
                    if (dx * dx + dy * dy + dz * dz > r * r) continue;
                    int id = IX(x, y, z, N);
                    Vx[id] += (ax - Vx[id]) * t;
                    Vy[id] += (ay - Vy[id]) * t;
                    Vz[id] += (az - Vz[id]) * t;
                }
    }

    void raiseDensitySphere(float cx, float cy, float cz, float r, float target) {
        int N = size;
        int x0 = (int)std::floor(cx - r), x1 = (int)std::ceil(cx + r);
        int y0 = (int)std::floor(cy - r), y1 = (int)std::ceil(cy + r);
        int z0 = (int)std::floor(cz - r), z1 = (int)std::ceil(cz + r);
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    if (x < 0 || y < 0 || z < 0 || x >= N || y >= N || z >= N) continue;
                    float dx = x - cx, dy = y - cy, dz = z - cz;
                    float d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 > r * r) continue;
                    int id = IX(x, y, z, N);
                    density[id] = std::max(density[id], target * (1.0f - d2 / (r * r)));
                }
    }

    int forceMode = 0;
    void addFrameForces(float dVx, float dVy, float dVz,
                        float wx, float wy, float wz,
                        float ax, float ay, float az, float k) {
        int N = size;
        #pragma omp parallel for
        for (int z = 1; z < N - 1; z++) {
            float rz = (z + 0.5f) / N - 0.5f;
            int zN = z * N * N;
            for (int y = 1; y < N - 1; y++) {
                float ry = (y + 0.5f) / N - 0.5f;
                int base = zN + y * N;
                for (int i = 1; i < N - 1; i++) {
                    float rx = (i + 0.5f) / N - 0.5f;
                    int id = base + i;
                    float vx = Vx[id], vy = Vy[id], vz = Vz[id];
                    float arx = ay * rz - az * ry, ary = az * rx - ax * rz, arz = ax * ry - ay * rx;
                    float c2x = 2.0f * (wy * vz - wz * vy), c2y = 2.0f * (wz * vx - wx * vz),
                          c2z = 2.0f * (wx * vy - wy * vx);
                    float wrx = wy * rz - wz * ry, wry = wz * rx - wx * rz, wrz = wx * ry - wy * rx;
                    float wwrx = wy * wrz - wz * wry, wwry = wz * wrx - wx * wrz,
                          wwrz = wx * wry - wy * wrx;
                    float wgt = forceMode ? std::min(1.0f, 2.0f * density[id]) : 1.0f;
                    if (forceMode == 2) {
                        Vx[id] -= k * (dVx + (arx + c2x + wwrx) * dt) * wgt;
                        Vy[id] -= k * (dVy + (ary + c2y + wwry) * dt) * wgt;
                        Vz[id] -= k * (dVz + (arz + c2z + wwrz) * dt) * wgt;
                    } else {
                        Vx[id] -= k * (dVx * wgt + (arx + c2x + wwrx) * dt);
                        Vy[id] -= k * (dVy * wgt + (ary + c2y + wwry) * dt);
                        Vz[id] -= k * (dVz * wgt + (arz + c2z + wwrz) * dt);
                    }
                }
            }
        }
    }

    void raiseHeatSphere(float cx, float cy, float cz, float r, float target) {
        int N = size;
        int x0 = (int)std::floor(cx - r), x1 = (int)std::ceil(cx + r);
        int y0 = (int)std::floor(cy - r), y1 = (int)std::ceil(cy + r);
        int z0 = (int)std::floor(cz - r), z1 = (int)std::ceil(cz + r);
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    if (x < 0 || y < 0 || z < 0 || x >= N || y >= N || z >= N) continue;
                    float dx = x - cx, dy = y - cy, dz = z - cz;
                    float d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 > r * r) continue;
                    int id = IX(x, y, z, N);
                    heat[id] = std::max(heat[id], target * (1.0f - d2 / (r * r)));
                }
    }

    void step() {
        int N = size, NN = N * N;
        {
            float buoy = buoyancy, hbuoy = heatBuoyancy;
            float ux = upX, uy = upY, uz = upZ;
            const float* d = density.data();
            const float* ht = heat.data();
            float* vx = Vx.data();
            float* vy = Vy.data();
            float* vz = Vz.data();
            #pragma omp parallel for
            for (int i = 0; i < N * NN; i++) {
                float b = (buoy * d[i] + hbuoy * ht[i]) * dt;
                vx[i] += b * ux; vy[i] += b * uy; vz[i] += b * uz;
            }
        }
        vorticityConfinement();
        {
            float mv = maxVel;
            float* vx = Vx.data();
            float* vy = Vy.data();
            float* vz = Vz.data();
            #pragma omp parallel for
            for (int i = 0; i < N * NN; i++) {
                vx[i] = std::min(std::max(vx[i], -mv), mv);
                vy[i] = std::min(std::max(vy[i], -mv), mv);
                vz[i] = std::min(std::max(vz[i], -mv), mv);
            }
        }
        diffuse(1, Vx0.data(), Vx.data(), visc);
        diffuse(2, Vy0.data(), Vy.data(), visc);
        diffuse(3, Vz0.data(), Vz.data(), visc);

        project(Vx0.data(), Vy0.data(), Vz0.data(), pressA.data(), Vy.data());

        bool macV = advMode >= 2;
        advect(1, Vx.data(), Vx0.data(), Vx0.data(), Vy0.data(), Vz0.data(), macV);
        advect(2, Vy.data(), Vy0.data(), Vx0.data(), Vy0.data(), Vz0.data(), macV);
        advect(3, Vz.data(), Vz0.data(), Vx0.data(), Vy0.data(), Vz0.data(), macV);

        if (measure) divRmsBefore = divRmsCell();
        project(Vx.data(), Vy.data(), Vz.data(), pressB.data(), Vx0.data());
        if (measure) divRmsAfter = divRmsCell();

        diffuse(0, s.data(), density.data(), diff);
        advect(0, density.data(), s.data(), Vx.data(), Vy.data(), Vz.data(), advMode >= 1);
        if (sharpen > 0.0f) {
            float sh = sharpen;
            float* d = density.data();
            #pragma omp parallel for
            for (int z = 1; z < N - 1; z++)
                for (int y = 1; y < N - 1; y++)
                    for (int x = 1; x < N - 1; x++) {
                        int id = z * NN + y * N + x;
                        float v = d[id] + sh * 4.0f * d[id] * (1.0f - d[id]) * (d[id] - 0.5f);
                        d[id] = std::min(std::max(v, 0.0f), 1.0f);
                    }
        }
        std::copy(heat.begin(), heat.end(), heat0.begin());
        advect(0, heat.data(), heat0.data(), Vx.data(), Vy.data(), Vz.data(), false);

        {
            float diss = dissipation, cool = cooling;
            float* d = density.data();
            float* ht = heat.data();
            #pragma omp parallel for
            for (int i = 0; i < N * NN; i++) {
                d[i] *= diss;
                ht[i] = std::max(ht[i] * cool, 0.0f);
            }
        }
        if (targetMass > 0.0f) {
            float* d = density.data();
            for (int pass = 0; pass < 2; pass++) {
                double tot = 0.0, wsum = 0.0;
                #pragma omp parallel for reduction(+ : tot, wsum)
                for (int z = 1; z < N - 1; z++)
                    for (int y = 1; y < N - 1; y++)
                        for (int x = 1; x < N - 1; x++) {
                            int id = z * NN + y * N + x;
                            if (d[id] < 0.0f) d[id] = 0.0f;
                            tot += (double)d[id];
                            wsum += (double)d[id] * (1.0 - d[id]);
                        }
                if (tot <= 1e-6) break;
                double deficit = targetMass - tot;
                if (std::fabs(deficit) <= 0.001 * targetMass) break;
                if (wsum > 1e-3) {
                    // add/remove mass at the interface, where d(1-d) is largest
                    float k = (float)(deficit / wsum);
                    #pragma omp parallel for
                    for (int z = 1; z < N - 1; z++)
                        for (int y = 1; y < N - 1; y++)
                            for (int x = 1; x < N - 1; x++) {
                                int id = z * NN + y * N + x;
                                d[id] = std::min(std::max(d[id] + k * d[id] * (1.0f - d[id]), 0.0f), 1.0f);
                            }
                } else {
                    float sc = (float)(targetMass / tot);
                    #pragma omp parallel for
                    for (int z = 1; z < N - 1; z++)
                        for (int y = 1; y < N - 1; y++)
                            for (int x = 1; x < N - 1; x++)
                                d[z * NN + y * N + x] *= sc;
                    break;
                }
            }
            set_bnd(0, d, N);
        }
    }

private:
    void vorticityConfinement() {
        int N = size;
        float h = 0.5f;
        const float* u = Vx.data();
        const float* v = Vy.data();
        const float* w = Vz.data();
        int NN = N * N;
        #pragma omp parallel for
        for (int z = 1; z < N - 1; z++) {
            int zN = z * NN;
            for (int y = 1; y < N - 1; y++) {
                int base = zN + y * N;
                for (int i = 1; i < N - 1; i++) {
                    int id = base + i;
                    float cx = h * (w[id + N] - w[id - N] - v[id + NN] + v[id - NN]);
                    float cy = h * (u[id + NN] - u[id - NN] - w[id + 1] + w[id - 1]);
                    float cz = h * (v[id + 1] - v[id - 1] - u[id + N] + u[id - N]);
                    curlX[id] = cx;
                    curlY[id] = cy;
                    curlZ[id] = cz;
                    curlM[id] = std::sqrt(cx * cx + cy * cy + cz * cz);
                }
            }
        }
        float eps = vorticity;
        float* uf = Vx.data();
        float* vf = Vy.data();
        float* wf = Vz.data();
        const float* m = curlM.data();
        #pragma omp parallel for
        for (int z = 1; z < N - 1; z++) {
            int zN = z * NN;
            for (int y = 1; y < N - 1; y++) {
                int base = zN + y * N;
                for (int i = 1; i < N - 1; i++) {
                    int id = base + i;
                    float gx = h * (m[id + 1] - m[id - 1]);
                    float gy = h * (m[id + N] - m[id - N]);
                    float gz = h * (m[id + NN] - m[id - NN]);
                    float len = std::sqrt(gx * gx + gy * gy + gz * gz) + 1e-5f;
                    gx /= len; gy /= len; gz /= len;
                    float ox = curlX[id], oy = curlY[id], oz = curlZ[id];
                    uf[id] += eps * dt * (gy * oz - gz * oy);
                    vf[id] += eps * dt * (gz * ox - gx * oz);
                    wf[id] += eps * dt * (gx * oy - gy * ox);
                }
            }
        }
    }

    void set_bnd(int b, float* x, int N) {
        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                x[IX(0, i, j, N)]     = b == 1 ? -x[IX(1, i, j, N)]     : x[IX(1, i, j, N)];
                x[IX(N - 1, i, j, N)] = b == 1 ? -x[IX(N - 2, i, j, N)] : x[IX(N - 2, i, j, N)];
                x[IX(i, 0, j, N)]     = b == 2 ? -x[IX(i, 1, j, N)]     : x[IX(i, 1, j, N)];
                x[IX(i, N - 1, j, N)] = closedTop
                        ? (b == 2 ? -x[IX(i, N - 2, j, N)] : x[IX(i, N - 2, j, N)])
                        : (b == 4 ? 0.0f : x[IX(i, N - 2, j, N)]);
                x[IX(i, j, 0, N)]     = b == 3 ? -x[IX(i, j, 1, N)]     : x[IX(i, j, 1, N)];
                x[IX(i, j, N - 1, N)] = b == 3 ? -x[IX(i, j, N - 2, N)] : x[IX(i, j, N - 2, N)];
            }
        }
        // edges: average of the two adjacent interior edge cells
        for (int k = 1; k < N - 1; k++) {
            x[IX(0, 0, k, N)]         = 0.5f * (x[IX(1, 0, k, N)]         + x[IX(0, 1, k, N)]);
            x[IX(N - 1, 0, k, N)]     = 0.5f * (x[IX(N - 2, 0, k, N)]     + x[IX(N - 1, 1, k, N)]);
            x[IX(0, N - 1, k, N)]     = 0.5f * (x[IX(1, N - 1, k, N)]     + x[IX(0, N - 2, k, N)]);
            x[IX(N - 1, N - 1, k, N)] = 0.5f * (x[IX(N - 2, N - 1, k, N)] + x[IX(N - 1, N - 2, k, N)]);
            x[IX(0, k, 0, N)]         = 0.5f * (x[IX(1, k, 0, N)]         + x[IX(0, k, 1, N)]);
            x[IX(N - 1, k, 0, N)]     = 0.5f * (x[IX(N - 2, k, 0, N)]     + x[IX(N - 1, k, 1, N)]);
            x[IX(0, k, N - 1, N)]     = 0.5f * (x[IX(1, k, N - 1, N)]     + x[IX(0, k, N - 2, N)]);
            x[IX(N - 1, k, N - 1, N)] = 0.5f * (x[IX(N - 2, k, N - 1, N)] + x[IX(N - 1, k, N - 2, N)]);
            x[IX(k, 0, 0, N)]         = 0.5f * (x[IX(k, 1, 0, N)]         + x[IX(k, 0, 1, N)]);
            x[IX(k, N - 1, 0, N)]     = 0.5f * (x[IX(k, N - 2, 0, N)]     + x[IX(k, N - 1, 1, N)]);
            x[IX(k, 0, N - 1, N)]     = 0.5f * (x[IX(k, 1, N - 1, N)]     + x[IX(k, 0, N - 2, N)]);
            x[IX(k, N - 1, N - 1, N)] = 0.5f * (x[IX(k, N - 2, N - 1, N)] + x[IX(k, N - 1, N - 2, N)]);
        }
        // corners: average of three adjacent cells
        x[IX(0, 0, 0, N)]             = (x[IX(1, 0, 0, N)]         + x[IX(0, 1, 0, N)]         + x[IX(0, 0, 1, N)])         / 3.0f;
        x[IX(N - 1, 0, 0, N)]         = (x[IX(N - 2, 0, 0, N)]     + x[IX(N - 1, 1, 0, N)]     + x[IX(N - 1, 0, 1, N)])     / 3.0f;
        x[IX(0, N - 1, 0, N)]         = (x[IX(1, N - 1, 0, N)]     + x[IX(0, N - 2, 0, N)]     + x[IX(0, N - 1, 1, N)])     / 3.0f;
        x[IX(N - 1, N - 1, 0, N)]     = (x[IX(N - 2, N - 1, 0, N)] + x[IX(N - 1, N - 2, 0, N)] + x[IX(N - 1, N - 1, 1, N)]) / 3.0f;
        x[IX(0, 0, N - 1, N)]         = (x[IX(1, 0, N - 1, N)]     + x[IX(0, 1, N - 1, N)]     + x[IX(0, 0, N - 2, N)])     / 3.0f;
        x[IX(N - 1, 0, N - 1, N)]     = (x[IX(N - 2, 0, N - 1, N)] + x[IX(N - 1, 1, N - 1, N)] + x[IX(N - 1, 0, N - 2, N)]) / 3.0f;
        x[IX(0, N - 1, N - 1, N)]     = (x[IX(1, N - 1, N - 1, N)] + x[IX(0, N - 2, N - 1, N)] + x[IX(0, N - 1, N - 2, N)]) / 3.0f;
        x[IX(N - 1, N - 1, N - 1, N)] = (x[IX(N - 2, N - 1, N - 1, N)] + x[IX(N - 1, N - 2, N - 1, N)] + x[IX(N - 1, N - 1, N - 2, N)]) / 3.0f;
    }

    void lin_solve(int b, float* x, const float* x0, float a, float c) {
        int N = size;
        float cInv = 1.0f / c;
        for (int k = 0; k < iter; k++) {
            for (int parity = 0; parity < 2; parity++) {
                #pragma omp parallel for
                for (int z = 1; z < N - 1; z++) {
                    int zN = z * N * N;
                    for (int y = 1; y < N - 1; y++) {
                        int base = zN + y * N;
                        for (int i = 1; i < N - 1; i++) {
                            if (((i + y + z) & 1) != parity) continue;
                            int id = base + i;
                            x[id] = (x0[id] + a * (x[id - 1] + x[id + 1] +
                                                   x[id - N] + x[id + N] +
                                                   x[id - N * N] + x[id + N * N])) * cInv;
                        }
                    }
                }
            }
            set_bnd(b, x, N);
        }
    }

    void diffuse(int b, float* x, const float* x0, float df) {
        int N = size;
        if (df <= 0.0f) {
            std::copy(x0, x0 + N * N * N, x);
            set_bnd(b, x, N);
            return;
        }
        float a = dt * df * (N - 2) * (N - 2);
        lin_solve(b, x, x0, a, 1 + 6 * a);
    }

    float divRmsCell() const {
        int N = size, NN = N * N;
        double s = 0.0;
        long cnt = 0;
        const float* u = Vx.data();
        const float* v = Vy.data();
        const float* w = Vz.data();
        #pragma omp parallel for reduction(+ : s)
        for (int z = 1; z < N - 1; z++) {
            int zN = z * NN;
            for (int y = 1; y < N - 1; y++) {
                int base = zN + y * N;
                for (int i = 1; i < N - 1; i++) {
                    int id = base + i;
                    float dv = 0.5f * (u[id + 1] - u[id - 1] +
                                       v[id + N] - v[id - N] +
                                       w[id + NN] - w[id - NN]);
                    s += (double)dv * dv;
                }
            }
        }
        cnt = (long)(N - 2) * (N - 2) * (N - 2);
        return (float)std::sqrt(s / cnt);
    }

    void matvecP(const float* p, float* out) {
        int N = size, NN = N * N;
        const float* dg = diagP.data();
        #pragma omp parallel for
        for (int z = 1; z < N - 1; z++) {
            int zN = z * NN;
            for (int y = 1; y < N - 1; y++) {
                int base = zN + y * N;
                for (int i = 1; i < N - 1; i++) {
                    int id = base + i;
                    float acc = dg[id] * p[id];
                    if (i > 1)     acc -= p[id - 1];
                    if (i < N - 2) acc -= p[id + 1];
                    if (y > 1)     acc -= p[id - N];
                    if (y < N - 2) acc -= p[id + N];
                    // y == N-2: Dirichlet p = 0 when open (in diagP), Neumann (skipped) when closedTop
                    if (z > 1)     acc -= p[id - NN];
                    if (z < N - 2) acc -= p[id + NN];
                    out[id] = acc;
                }
            }
        }
    }

    void pcgSolve(float* p, const float* b) {
        int N = size, NN = N * N;
        const float* dg = diagP.data();
        const float* inv = invDiagP.data();
        float* r = cgR.data();
        float* z = cgZ.data();
        float* d = cgD.data();
        float* q = cgQ.data();
        lastCgIters = 0;
        lastCgRelRes = 0.0f;
        matvecP(p, q);  // masked: p's ghost layer holds set_bnd values
        double bnorm2 = 0.0, rz = 0.0;
        #pragma omp parallel for reduction(+ : bnorm2, rz)
        for (int zz = 1; zz < N - 1; zz++) {
            int zN = zz * NN;
            for (int y = 1; y < N - 1; y++) {
                int base = zN + y * N;
                for (int i = 1; i < N - 1; i++) {
                    int id = base + i;
                    bnorm2 += (double)b[id] * b[id];
                    float rv = b[id] - q[id];
                    float zv = rv * inv[id];
                    r[id] = rv;
                    z[id] = zv;
                    d[id] = zv;
                    rz += (double)rv * zv;
                }
            }
        }
        if (bnorm2 < 1e-20) return;
        double rnorm2 = bnorm2;

        for (int k = 0; k < cgMaxIter; k++) {
            // q = A d (branch-free: cgD ghost layer is permanently 0) fused with dq
            double dq = 0.0;
            #pragma omp parallel for reduction(+ : dq)
            for (int zz = 1; zz < N - 1; zz++) {
                int zN = zz * NN;
                for (int y = 1; y < N - 1; y++) {
                    int base = zN + y * N;
                    for (int i = 1; i < N - 1; i++) {
                        int id = base + i;
                        float qv = dg[id] * d[id] - (d[id - 1] + d[id + 1] +
                                                    d[id - N] + d[id + N] +
                                                    d[id - NN] + d[id + NN]);
                        q[id] = qv;
                        dq += (double)d[id] * qv;
                    }
                }
            }
            if (std::fabs(dq) < 1e-30) { lastCgIters = k; break; }
            float alpha = (float)(rz / dq);
            // fused: p += a d, r -= a q, z = r * invDiag; accumulate rnorm2 + rzNew
            rnorm2 = 0.0;
            double rzNew = 0.0;
            #pragma omp parallel for reduction(+ : rnorm2, rzNew)
            for (int zz = 1; zz < N - 1; zz++) {
                int zN = zz * NN;
                for (int y = 1; y < N - 1; y++) {
                    int base = zN + y * N;
                    for (int i = 1; i < N - 1; i++) {
                        int id = base + i;
                        p[id] += alpha * d[id];
                        float rv = r[id] - alpha * q[id];
                        float zv = rv * inv[id];
                        r[id] = rv;
                        z[id] = zv;
                        rnorm2 += (double)rv * rv;
                        rzNew += (double)rv * zv;
                    }
                }
            }
            lastCgIters = k + 1;
            lastCgRelRes = (float)std::sqrt(rnorm2 / bnorm2);
            if (lastCgRelRes < cgTol) break;
            float beta = (float)(rzNew / rz);
            rz = rzNew;
            #pragma omp parallel for
            for (int zz = 1; zz < N - 1; zz++)
                for (int y = 1; y < N - 1; y++)
                    for (int i = 1; i < N - 1; i++) {
                        int id = zz * NN + y * N + i;
                        d[id] = z[id] + beta * d[id];
                    }
        }
    }

    void project(float* u, float* v, float* w, float* p, float* div) {
        int N = size;
        float h = 1.0f / (N - 2);
        #pragma omp parallel for
        for (int z = 1; z < N - 1; z++) {
            int zN = z * N * N;
            for (int y = 1; y < N - 1; y++) {
                int base = zN + y * N;
                for (int i = 1; i < N - 1; i++) {
                    int id = base + i;
                    div[id] = -0.5f * h * (u[id + 1] - u[id - 1] +
                                           v[id + N] - v[id - N] +
                                           w[id + N * N] - w[id - N * N]);
                }
            }
        }
        set_bnd(0, div, N);
        set_bnd(4, p, N);
        if (closedTop) {
            // pure-Neumann system: compatibility requires zero-mean RHS
            int NN = N * N;
            double mean = 0.0;
            #pragma omp parallel for reduction(+ : mean)
            for (int z = 1; z < N - 1; z++)
                for (int y = 1; y < N - 1; y++)
                    for (int i = 1; i < N - 1; i++)
                        mean += (double)div[z * NN + y * N + i];
            mean /= (double)((N - 2) * (N - 2) * (N - 2));
            float mf = (float)mean;
            #pragma omp parallel for
            for (int z = 1; z < N - 1; z++)
                for (int y = 1; y < N - 1; y++)
                    for (int i = 1; i < N - 1; i++)
                        div[z * NN + y * N + i] -= mf;
        }
        if (useCG) pcgSolve(p, div);
        else lin_solve(4, p, div, 1.0f, 6.0f);
        if (closedTop) {
            int NN = N * N;
            double mean = 0.0;
            #pragma omp parallel for reduction(+ : mean)
            for (int z = 1; z < N - 1; z++)
                for (int y = 1; y < N - 1; y++)
                    for (int i = 1; i < N - 1; i++)
                        mean += (double)p[z * NN + y * N + i];
            mean /= (double)((N - 2) * (N - 2) * (N - 2));
            float mf = (float)mean;
            #pragma omp parallel for
            for (int z = 1; z < N - 1; z++)
                for (int y = 1; y < N - 1; y++)
                    for (int i = 1; i < N - 1; i++)
                        p[z * NN + y * N + i] -= mf;
        }
        set_bnd(4, p, N);
        #pragma omp parallel for
        for (int z = 1; z < N - 1; z++) {
            int zN = z * N * N;
            for (int y = 1; y < N - 1; y++) {
                int base = zN + y * N;
                for (int i = 1; i < N - 1; i++) {
                    int id = base + i;
                    u[id] -= 0.5f * (p[id + 1] - p[id - 1]) / h;
                    v[id] -= 0.5f * (p[id + N] - p[id - N]) / h;
                    w[id] -= 0.5f * (p[id + N * N] - p[id - N * N]) / h;
                }
            }
        }
        set_bnd(1, u, N);
        set_bnd(2, v, N);
        set_bnd(3, w, N);
    }

    void semiLag(float* out, const float* in,
                 const float* u, const float* v, const float* w, float dts) {
        int N = size;
        float dt0 = dts * (N - 2);
        int NN = N * N;
        #pragma omp parallel for
        for (int z = 1; z < N - 1; z++) {
            int zN = z * NN;
            for (int y = 1; y < N - 1; y++) {
                int base = zN + y * N;
                for (int i = 1; i < N - 1; i++) {
                    int id = base + i;
                    float x = i - dt0 * u[id];
                    float yy = y - dt0 * v[id];
                    float zz = z - dt0 * w[id];
                    x = std::min(std::max(x, 0.5f), N - 1.5f);
                    yy = std::min(std::max(yy, 0.5f), N - 1.5f);
                    zz = std::min(std::max(zz, 0.5f), N - 1.5f);
                    int i0 = (int)x, i1 = i0 + 1;
                    int j0 = (int)yy, j1 = j0 + 1;
                    int k0 = (int)zz, k1 = k0 + 1;
                    float s1 = x - i0, s0 = 1.0f - s1;
                    float t1 = yy - j0, t0 = 1.0f - t1;
                    float u1 = zz - k0, u0 = 1.0f - u1;
                    out[id] =
                        u0 * (t0 * (s0 * in[i0 + j0 * N + k0 * NN] + s1 * in[i1 + j0 * N + k0 * NN]) +
                              t1 * (s0 * in[i0 + j1 * N + k0 * NN] + s1 * in[i1 + j1 * N + k0 * NN])) +
                        u1 * (t0 * (s0 * in[i0 + j0 * N + k1 * NN] + s1 * in[i1 + j0 * N + k1 * NN]) +
                              t1 * (s0 * in[i0 + j1 * N + k1 * NN] + s1 * in[i1 + j1 * N + k1 * NN]));
                }
            }
        }
    }

    void advect(int b, float* d, const float* d0,
                const float* u, const float* v, const float* w, bool mac) {
        int N = size;
        if (!mac) {
            semiLag(d, d0, u, v, w, dt);
            set_bnd(b, d, N);
            return;
        }
        float* hat = macHat.data();
        float* back = macBack.data();
        semiLag(hat, d0, u, v, w, dt);
        set_bnd(b, hat, N);
        semiLag(back, hat, u, v, w, -dt);
        set_bnd(b, back, N);
        float dt0 = dt * (N - 2);
        int NN = N * N;
        #pragma omp parallel for
        for (int z = 1; z < N - 1; z++) {
            int zN = z * NN;
            for (int y = 1; y < N - 1; y++) {
                int base = zN + y * N;
                for (int i = 1; i < N - 1; i++) {
                    int id = base + i;
                    float x = i - dt0 * u[id];
                    float yy = y - dt0 * v[id];
                    float zz = z - dt0 * w[id];
                    x = std::min(std::max(x, 0.5f), N - 1.5f);
                    yy = std::min(std::max(yy, 0.5f), N - 1.5f);
                    zz = std::min(std::max(zz, 0.5f), N - 1.5f);
                    int i0 = (int)x, i1 = i0 + 1;
                    int j0 = (int)yy, j1 = j0 + 1;
                    int k0 = (int)zz, k1 = k0 + 1;
                    float mn = d0[i0 + j0 * N + k0 * NN], mx = mn;
                    const int ci[8] = {i0, i1, i0, i1, i0, i1, i0, i1};
                    const int cj[8] = {j0, j0, j1, j1, j0, j0, j1, j1};
                    const int ck[8] = {k0, k0, k0, k0, k1, k1, k1, k1};
                    for (int c = 1; c < 8; c++) {
                        float sv = d0[ci[c] + cj[c] * N + ck[c] * NN];
                        mn = std::min(mn, sv);
                        mx = std::max(mx, sv);
                    }
                    float val = hat[id] + 0.5f * (d0[id] - back[id]);
                    d[id] = (val < mn || val > mx) ? hat[id] : val;
                }
            }
        }
        set_bnd(b, d, N);
    }
};

static const char* volumeVS = R"GLSL(
#version 330

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;

out vec3 fragPosition;
out vec2 fragTexCoord;
out vec3 fragNormal;
out vec4 fragColor;

void main() {
    vec4 worldPos = matModel * vec4(vertexPosition, 1.0);
    fragPosition = worldPos.xyz;
    fragTexCoord = vertexTexCoord;
    fragNormal = normalize(vec3(matNormal * vec4(vertexNormal, 0.0)));
    fragColor = vertexColor;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
)GLSL";

static const char* volumeFS = R"GLSL(
#version 330

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;

out vec4 finalColor;

uniform sampler2D atlas;
uniform vec3 camPos;
uniform mat4 invBox;
uniform int tilesX;
uniform int tilesY;
uniform int gridN;
uniform int colorMode;
uniform float marchSteps;
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform float lightIntensity;
uniform float ambient;
uniform int lighting;
uniform int shadowSteps;
uniform float shadowSigma;
uniform float emission;
uniform float toneExposure;
uniform int frame;
uniform int ortho;
uniform vec3 viewDir;
uniform float iso;
uniform float sigmaScale;

float ign(vec2 px, int f) {
    return fract(52.9829189 * fract(dot(px + float(f - (f / 64) * 64) * 5.588238,
                                        vec2(0.06711056, 0.00583715))));
}

vec2 sampleField(vec3 p) {
    p = clamp(p, 0.0, 0.99999);
    float zf = clamp(p.z * float(gridN) - 0.5, 0.0, float(gridN - 1));
    int z0 = int(floor(zf));
    int z1 = min(z0 + 1, gridN - 1);
    float fz = zf - float(z0);
    vec2 inSlice = clamp(p.xy * float(gridN), 0.5, float(gridN) - 0.5);
    float N = float(gridN);
    vec2 atlasSize = vec2(float(tilesX) * N, float(tilesY) * N);
    vec2 uv0 = (vec2(float(z0 - (z0 / tilesX) * tilesX), float(z0 / tilesX)) * N + inSlice) / atlasSize;
    vec2 uv1 = (vec2(float(z1 - (z1 / tilesX) * tilesX), float(z1 / tilesX)) * N + inSlice) / atlasSize;
    vec2 a = texture(atlas, uv0).ra;
    vec2 b = texture(atlas, uv1).ra;
    return mix(a, b, fz);
}

float sampleDensity(vec3 p) {
    return sampleField(p).x;
}

vec3 ramp(float d, float acc) {
    if (colorMode == 0) {
        return mix(vec3(0.55, 0.6, 0.7), vec3(1.0), clamp(d * 1.5, 0.0, 1.0));
    } else if (colorMode == 1) {
        return mix(vec3(0.10, 0.10, 0.12), vec3(0.32, 0.32, 0.36), d);
    } else if (colorMode == 3) {
        vec3 c = mix(vec3(0.1, 0.15, 0.5), vec3(1.0, 0.45, 0.05), clamp(d * 2.0, 0.0, 1.0));
        c = mix(c, vec3(1.0, 0.95, 0.8), clamp(d * 2.0 - 1.0, 0.0, 1.0));
        return c * 1.6;
    } else {
        float h = fract(acc * 0.7 + d);
        return clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    }
}

vec2 boxIntersect(vec3 ro, vec3 rd) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (vec3(0.0) - ro) * inv;
    vec3 t1 = (vec3(1.0) - ro) * inv;
    vec3 tmin = min(t0, t1), tmax = max(t0, t1);
    return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

float hgPhase(float mu, float g) {
    float g2 = g * g;
    // normalized so isotropic = 1 (4*pi * HG)
    return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5);
}

vec3 flameColor(float h) {
    h = clamp(h, 0.0, 1.0);
    vec3 c = mix(vec3(0.6, 0.05, 0.0), vec3(1.0, 0.35, 0.02), clamp((h - 0.15) / 0.30, 0.0, 1.0));
    c = mix(c, vec3(1.0, 0.75, 0.15), clamp((h - 0.45) / 0.30, 0.0, 1.0));
    c = mix(c, vec3(1.0, 0.9, 0.6), clamp((h - 0.92) / 0.08, 0.0, 1.0));
    return c;
}

void main() {
    vec3 rd = ortho != 0 ? normalize(viewDir) : normalize(fragPosition - camPos);
    vec3 ro = ortho != 0 ? fragPosition - rd * 4.0 : camPos;
    vec3 roL = (invBox * vec4(ro, 1.0)).xyz;
    vec3 rdL = normalize((invBox * vec4(rd, 0.0)).xyz);
    vec2 t = boxIntersect(roL, rdL);
    float t0 = max(t.x, 0.0);
    float t1 = t.y;
    if (t1 <= t0) discard;

    float steps = marchSteps;
    float stepLen = (t1 - t0) / steps;
    float sigma = sigmaScale;
    float T = 1.0;
    vec3 col = vec3(0.0);
    vec3 L = normalize(mat3(invBox) * lightDir);
    float jitter = ign(gl_FragCoord.xy, frame);
    float sjitter = fract(jitter + 0.61803398);
    float phase = hgPhase(dot(-rdL, L), 0.2);
    bool lit = (lighting != 0);

    for (int i = 0; i < 256; i++) {
        if (float(i) >= steps) break;
        vec3 pos = roL + rdL * (t0 + (float(i) + jitter) * stepLen);
        vec2 f = sampleField(pos);
        float d = smoothstep(iso - 0.08, iso + 0.08, f.x), h = f.y;
        if (d > 0.003 || h > 0.003) {
            float a = 1.0 - exp(-d * sigma * stepLen);
            vec3 albedo = ramp(d, 1.0 - T);
            vec3 radiance = albedo;
            if (lit) {
                float Tl = 1.0;
                if (d >= 0.05 && T > 0.05) {
                    vec2 tl = boxIntersect(pos, L);
                    float shadowLen = min(tl.y, 0.5);
                    float sstep = shadowLen / float(shadowSteps);
                    float depth = 0.0;
                    for (int j = 0; j < 64; j++) {
                        if (j >= shadowSteps) break;
                        vec3 sp = pos + L * ((float(j) + sjitter) * sstep);
                        depth += sampleDensity(sp) * sstep;
                    }
                    Tl = exp(-shadowSigma * depth);
                }
                radiance = albedo * (ambient + lightColor * Tl * phase * lightIntensity);
            }
            col += T * a * radiance;
            if (colorMode != 0 && h > 0.003) {
                col += T * flameColor(h) * h * h * emission * stepLen * 40.0;
            }
            T *= exp(-d * sigma * stepLen);
            if (T < 0.01) break;
        }
    }
    float alpha = 1.0 - T;
    if (alpha < 0.005) discard;
    col = vec3(1.0) - exp(-col * toneExposure);
    finalColor = vec4(col, alpha);
}
)GLSL";

static Color sliceColor(float d, int mode) {
    d = std::min(std::max(d, 0.0f), 1.0f);
    if (mode == 0) {
        unsigned char v = (unsigned char)(d * 255);
        return Color{v, v, v, 255};
    }
    if (mode == 1 || mode == 3) {
        float t = std::min(d * 2.0f, 1.0f);
        Color a{13, 26, 102, 255}, b{255, 115, 13, 255}, c{255, 242, 204, 255};
        Color lo = {(unsigned char)(a.r + (b.r - a.r) * t), (unsigned char)(a.g + (b.g - a.g) * t),
                    (unsigned char)(a.b + (b.b - a.b) * t), 255};
        float u = std::min(std::max(d * 2.0f - 1.0f, 0.0f), 1.0f);
        return Color{(unsigned char)(lo.r + (c.r - lo.r) * u), (unsigned char)(lo.g + (c.g - lo.g) * u),
                     (unsigned char)(lo.b + (c.b - lo.b) * u), 255};
    }
    return Color{(unsigned char)(d * 255), (unsigned char)(128 + 127 * std::sin(d * 6.28f)),
                 (unsigned char)(255 - d * 255), 255};
}

static int soakFrames = 0;
static bool soakSlice = false, testDrag = false, lumCheck = false;
int main(int argc, char** argv) {
    bool shotVolume = false, shotSlice = false, argGS = false, argSL = false, argNoLight = false;
    bool noTracker = false, argFake = false, argTestSlosh = false;
    int argCam = 0, argPort = 9876, argForces = 1;
    double argCurlPhase = 0.0;
    float argInertia = -1.0f;
    float restSize = 0.6f;
    int argMode = -1;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--screenshot") == 0) shotVolume = true;
        if (std::strcmp(argv[i], "--screenshot-slice") == 0) shotSlice = true;
        if (std::strcmp(argv[i], "--gs") == 0) argGS = true;
        if (std::strcmp(argv[i], "--sl") == 0) argSL = true;
        if (std::strcmp(argv[i], "--nolight") == 0) argNoLight = true;
        if (std::strcmp(argv[i], "--soak") == 0 && i + 1 < argc) soakFrames = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--sliceview") == 0) soakSlice = true;
        if (std::strcmp(argv[i], "--testdrag") == 0) testDrag = true;
        if (std::strcmp(argv[i], "--lumcheck") == 0) lumCheck = true;
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) argMode = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--no-tracker") == 0) noTracker = true;
        if (std::strcmp(argv[i], "--fake") == 0) argFake = true;
        if (std::strcmp(argv[i], "--cam") == 0 && i + 1 < argc) argCam = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) argPort = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--restsize") == 0 && i + 1 < argc) restSize = (float)std::atof(argv[++i]);
        if (std::strcmp(argv[i], "--testslosh") == 0) { argTestSlosh = true; noTracker = true; }
        if (std::strcmp(argv[i], "--inertia") == 0 && i + 1 < argc) argInertia = (float)std::atof(argv[++i]);
        if (std::strcmp(argv[i], "--forces") == 0 && i + 1 < argc) argForces = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--curlphase") == 0 && i + 1 < argc) argCurlPhase = std::atof(argv[++i]);
    }

#ifdef _OPENMP
    omp_set_num_threads(std::max(1, omp_get_num_procs() - 2));
#endif

    const int N = 48;
    FluidCube fluid(N, 0.1f, 0.0f, 0.0f);
    if (argGS) fluid.useCG = false;
    if (argSL) fluid.advMode = 0;
    fluid.dissipation = 1.0f;
    fluid.setClosedTop(true);
    fluid.forceMode = argForces;
    fluid.sharpen = 0.15f;
    fluid.maxVel = 0.8f;
    if (shotVolume || shotSlice) fluid.measure = true;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1280, 720, "3D Fluid - Hand Tracking");
    SetTargetFPS((shotVolume || shotSlice || soakFrames || argTestSlosh) ? 0 : 60);

    int tilesX = (int)std::ceil(std::sqrt((double)N));
    int tilesY = (N + tilesX - 1) / tilesX;
    int atlasW = tilesX * N, atlasH = tilesY * N;
    std::vector<unsigned char> atlasBuf(atlasW * atlasH * 2, 0);

    Image atlasImg;
    atlasImg.data = atlasBuf.data();
    atlasImg.width = atlasW;
    atlasImg.height = atlasH;
    atlasImg.mipmaps = 1;
    atlasImg.format = PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA;
    Texture2D atlasTex = LoadTextureFromImage(atlasImg);
    SetTextureFilter(atlasTex, TEXTURE_FILTER_BILINEAR);
    atlasImg.data = nullptr;

    std::vector<Color> sliceBuf(N * N);
    Image sliceImg{sliceBuf.data(), N, N, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
    Texture2D sliceTex = LoadTextureFromImage(sliceImg);
    SetTextureFilter(sliceTex, TEXTURE_FILTER_BILINEAR);

    Shader volShader = LoadShaderFromMemory(volumeVS, volumeFS);
    bool shaderOk = volShader.id > 0 && volShader.id != rlGetShaderIdDefault();
    if (!shaderOk) TraceLog(LOG_WARNING, "Volume shader failed, falling back to slice view");

    int locAtlas = GetShaderLocation(volShader, "atlas");
    int locCamPos = GetShaderLocation(volShader, "camPos");
    int locInvBox = GetShaderLocation(volShader, "invBox");
    int locTilesX = GetShaderLocation(volShader, "tilesX");
    int locTilesY = GetShaderLocation(volShader, "tilesY");
    int locGridN = GetShaderLocation(volShader, "gridN");
    int locColorMode = GetShaderLocation(volShader, "colorMode");
    int locMarch = GetShaderLocation(volShader, "marchSteps");
    int locLightDir = GetShaderLocation(volShader, "lightDir");
    int locLightColor = GetShaderLocation(volShader, "lightColor");
    int locLightIntensity = GetShaderLocation(volShader, "lightIntensity");
    int locAmbient = GetShaderLocation(volShader, "ambient");
    int locLighting = GetShaderLocation(volShader, "lighting");
    int locShadowSteps = GetShaderLocation(volShader, "shadowSteps");
    int locShadowSigma = GetShaderLocation(volShader, "shadowSigma");
    int locEmission = GetShaderLocation(volShader, "emission");
    int locFrame = GetShaderLocation(volShader, "frame");
    int locToneExposure = GetShaderLocation(volShader, "toneExposure");
    int locOrtho = GetShaderLocation(volShader, "ortho");
    int locViewDir = GetShaderLocation(volShader, "viewDir");
    int locIso = GetShaderLocation(volShader, "iso");
    int locSigmaScale = GetShaderLocation(volShader, "sigmaScale");

    Mesh cubeMesh = GenMeshCube(1.0f, 1.0f, 1.0f);
    Model cubeModel = LoadModelFromMesh(cubeMesh);
    cubeModel.materials[0].shader = volShader;
    Model sliceModel = LoadModelFromMesh(GenMeshPlane(1.0f, 1.0f, 1, 1));
    sliceModel.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = sliceTex;
    sliceModel.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = WHITE;

    Vector3 boxCenter{0.5f, 0.5f, 0.5f};

    Camera3D cam{};
    cam.up = {0, 1, 0};
    cam.fovy = 1.0f;
    cam.projection = CAMERA_ORTHOGRAPHIC;
    cam.position = {0.5f, 0.5f, 3.5f};
    cam.target = boxCenter;

    bool paused = false, sliceView = !shaderOk || shotSlice || soakSlice;
    int colorMode = 2;
    if (argMode >= 0 && argMode <= 3) colorMode = argMode;
    bool showHelp = true;
    float stepMs = 0.0f;
    double stepMsAccum = 0.0;
    int stepCount = 0;
    int frame = 0;
    bool mouseDown = false;
    Vector3 prevHit{};
    bool needRefill = true;
    float lightAzim = 0.7f, lightElev = 0.6f;
    bool lightingOn = !argNoLight;
    double metMs = 0, metCg = 0, metRes = 0, metBef = 0, metAft = 0, metFrm = 0;
    int metCount = 0;
    double smMs = 0, smCg = 0, smRes = 0; int smN = 0;

    netInit(argPort);
    if (!noTracker) {
        std::string targs = "--port " + std::to_string(argPort) +
                            " --cam " + std::to_string(argCam) +
                            (argFake ? " --fake" : "") +
                            (argCurlPhase != 0.0 ? " --curlphase " + std::to_string(argCurlPhase) : "");
        if (!trackerLaunch(targs, true))
            TraceLog(LOG_WARNING, "failed to launch tracker.py (use --no-tracker and run it manually)");
    }
    TrackerFrame tf;
    double lastHandPacketTime = 0.0;
    double handPacketClock = 0.0;
    double handPktEma = 0.0;
    bool showWebcam = true, showSkeleton = true;
    Texture2D camTex{};
    int camTexW = 0, camTexH = 0;

    float velScale = 0.5f;
    float iso = 0.5f, sigmaScale = 60.0f;
    float inertia = argInertia >= 0 ? argInertia : 1.0f;
    float fillLevel = 0.35f;
    float gravity = 0.6f;
    float openT = -1.0f, hratMin = 1e9f, hratMax = 0.0f;
    int fingerTips[5] = {4, 8, 12, 16, 20};
    int palmSet[5] = {0, 5, 9, 13, 17};
    struct Slot {
        bool active = false;
        double lastSeen = 0.0, activeSince = 0.0, missingSince = 0.0;
        Vector3 target[6]{};
        Vector3 pos[6]{};
        Vector3 vel[6]{};
        int hand = -1;
    } slots[2];
    bool boxFollow = true;
    int holderSlot = -1;
    Quaternion grabQH{0, 0, 0, 1}, grabQB{0, 0, 0, 1};
    Vector3 poseC{0.5f, 0.5f, 0.5f}, prevPoseC = poseC;
    Quaternion poseQ{0, 0, 0, 1}, prevPoseQ = poseQ;
    float poseS = restSize, prevPoseS = poseS;
    Vector3 VbPrev{0, 0, 0}, omegaPrev{0, 0, 0};
    bool forcePrimed = false;
    Quaternion lastTq{0, 0, 0, 1}, lastHQ[2]{};
    Vector3 lastTc{0.5f, 0.5f, 0.5f};
    float lastTs = restSize;
    float mxW = 0, mxDV = 0, mxAL = 0;
    Matrix boxM = MatrixIdentity(), invBox = MatrixIdentity();

    auto mouseHitOnPlane = [&](Vector2 m, Vector3& out) -> bool {
        Ray ray = GetScreenToWorldRay(m, cam);
        Vector3 n = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
        float denom = Vector3DotProduct(ray.direction, n);
        if (std::fabs(denom) < 1e-5f) return false;
        float t = Vector3DotProduct(Vector3Subtract(boxCenter, ray.position), n) / denom;
        if (t <= 0) return false;
        out = Vector3Add(ray.position, Vector3Scale(ray.direction, t));
        return true;
    };

    auto handQuat = [&](int h) {
        const HandData& hd = tf.hands[h];
        auto W = [&](int i) { return Vector3{hd.wlm[i][0], -hd.wlm[i][1], -hd.wlm[i][2]}; };
        Vector3 u = Vector3Normalize(Vector3Subtract(W(9), W(0)));
        Vector3 sv = Vector3Subtract(W(17), W(5));
        sv = Vector3Normalize(Vector3Subtract(sv, Vector3Scale(u, Vector3DotProduct(sv, u))));
        Vector3 n = Vector3CrossProduct(sv, u);
        Matrix m{};
        m.m0 = sv.x; m.m1 = sv.y; m.m2 = sv.z;
        m.m4 = u.x; m.m5 = u.y; m.m6 = u.z;
        m.m8 = n.x; m.m9 = n.y; m.m10 = n.z; m.m15 = 1.0f;
        Quaternion q = QuaternionFromMatrix(m);
        if (h >= 0 && h < 2) {
            Quaternion& p = lastHQ[h];
            float d = q.x * p.x + q.y * p.y + q.z * p.z + q.w * p.w;
            if (p.w * p.w + p.x * p.x + p.y * p.y + p.z * p.z > 0.5f && d < 0)
                q = {-q.x, -q.y, -q.z, -q.w};
            p = q;
        }
        return q;
    };

    auto injectAt = [&](Vector3 hit, bool hasPrev, Vector3 prev) {
        Vector3 g{hit.x * N, hit.y * N, hit.z * N};
        Vector3 drag{0, 0, 0};
        if (hasPrev) drag = Vector3Subtract(hit, prev);
        float scale = 30.0f, mv = fluid.maxVel;
        float ax = std::min(std::max(drag.x * scale, -mv), mv);
        float ay = std::min(std::max(drag.y * scale, -mv), mv);
        float az = std::min(std::max(drag.z * scale, -mv), mv);
        fluid.blendVelocitySphere(g.x, g.y, g.z, 5.0f, ax, ay, az, 0.85f);
    };

    while (!WindowShouldClose()) {
        frame++;

        double nowT = GetTime();
        if (netPoll(tf)) {
            double dtPkt = std::max(nowT - handPacketClock, 1.0 / 120.0);
            handPacketClock = nowT;
            lastHandPacketTime = nowT;
            handPktEma = handPktEma == 0.0 ? 1.0 / dtPkt : handPktEma * 0.9 + (1.0 / dtPkt) * 0.1;
            bool used[2] = {false, false};
            for (int h = 0; h < tf.numHands; h++) {
                auto toBox = [&](int i) {
                    return Vector3{0.5f + (tf.hands[h].lm[i][0] - 0.5f) * tf.camAspect,
                                   1.0f - tf.hands[h].lm[i][1], 0.5f};
                };
                Vector3 palm{0, 0, 0};
                for (int i : palmSet) palm = Vector3Add(palm, toBox(i));
                palm = Vector3Scale(palm, 1.0f / 5.0f);
                int best = -1; float bestD = 0.35f;
                for (int s = 0; s < 2; s++) {
                    if (!slots[s].active || used[s]) continue;
                    float d = Vector3Distance(slots[s].target[5], palm);
                    if (d < bestD) { bestD = d; best = s; }
                }
                if (best < 0)
                    for (int s = 0; s < 2; s++) if (!slots[s].active) { best = s; break; }
                if (best < 0) continue;
                Slot& sl = slots[best];
                bool wasActive = sl.active;
                sl.active = true; sl.lastSeen = nowT; sl.hand = h; used[best] = true;
                if (!wasActive) { for (auto& v : sl.vel) v = {0, 0, 0}; sl.activeSince = nowT; }
                Vector3 nt[6];
                for (int k = 0; k < 5; k++) nt[k] = toBox(fingerTips[k]);
                nt[5] = palm;
                for (int k = 0; k < 6; k++) {
                    Vector3 dv = Vector3Scale(Vector3Subtract(nt[k], sl.target[k]), (float)(1.0 / dtPkt));
                    if (!wasActive) { dv = {0, 0, 0}; sl.pos[k] = nt[k]; }
                    sl.vel[k].x += (dv.x - sl.vel[k].x) * 0.5f;
                    sl.vel[k].y += (dv.y - sl.vel[k].y) * 0.5f;
                    sl.vel[k].z += (dv.z - sl.vel[k].z) * 0.5f;
                    sl.target[k] = nt[k];
                }
            }
            for (int s = 0; s < 2; s++) {
                if (used[s]) { slots[s].missingSince = 0; continue; }
                if (s == holderSlot && slots[s].active) {
                    if (slots[s].missingSince == 0) slots[s].missingSince = nowT;
                    if (nowT - slots[s].missingSince <= 0.30) continue;
                }
                slots[s].active = false;
            }
        }
        for (int si = 0; si < 2; si++) {
            Slot& sl = slots[si];
            if (sl.active && nowT - sl.lastSeen > (si == holderSlot ? 0.45 : 0.25)) sl.active = false;
            for (int k = 0; k < 6; k++)
                sl.pos[k] = Vector3Add(sl.pos[k], Vector3Scale(Vector3Subtract(sl.target[k], sl.pos[k]), 0.5f));
        }

        int holder = -1;
        for (int s = 0; s < 2; s++)
            if (slots[s].active && nowT - slots[s].activeSince >= 0.2 &&
                (holder < 0 || slots[s].activeSince < slots[holder].activeSince ||
                 (slots[s].activeSince == slots[holder].activeSince && slots[s].hand < slots[holder].hand)))
                holder = s;
        if (holder != holderSlot) {
            if (holder >= 0 && slots[holder].hand >= 0 && slots[holder].hand < tf.numHands) {
                grabQH = handQuat(slots[holder].hand);
                grabQB = poseQ;
            }
            holderSlot = holder;
        }
        Vector3 tc{0.5f, 0.5f, 0.5f};
        Quaternion tq{0, 0, 0, 1};
        float ts = restSize;
        openT = -1.0f;
        if (boxFollow && holderSlot >= 0 && slots[holderSlot].active) {
            Slot& hs = slots[holderSlot];
            if (hs.hand >= 0 && hs.hand < tf.numHands) {
                const HandData& hd = tf.hands[hs.hand];
                auto W = [&](int i) {
                    return Vector3{hd.wlm[i][0], -hd.wlm[i][1], -hd.wlm[i][2]};
                };
                Vector3 w0w = W(0), w9w = W(9);
                float rat = 0.0f;
                for (int i : {8, 12, 16, 20}) rat += Vector3Distance(W(i), w0w);
                rat /= 4.0f * std::max(Vector3Distance(w9w, w0w), 1e-4f);
                hratMin = std::min(hratMin, rat); hratMax = std::max(hratMax, rat);
                float t = std::min(std::max((rat - 1.1f) / 0.8f, 0.0f), 1.0f);
                openT = t;
                auto M2 = [&](int i) {
                    return Vector2{0.5f + (hd.lm[i][0] - 0.5f) * tf.camAspect,
                                   1.0f - hd.lm[i][1]};
                };
                float scaleRef = std::max(Vector2Distance(M2(0), M2(9)),
                                          Vector2Distance(M2(5), M2(17)));
                ts = std::min(std::max(scaleRef * (0.9f + 1.5f * t), 0.12f), 0.9f);
                tq = QuaternionMultiply(
                    QuaternionMultiply(handQuat(hs.hand), QuaternionInvert(grabQH)), grabQB);
                Vector3 tip = hs.target[0];
                for (int k = 1; k < 5; k++)
                    if (hs.target[k].y > tip.y) tip = hs.target[k];
                Matrix Rm = QuaternionToMatrix(poseQ);
                float halfY = 0.5f * ts * (std::fabs(Rm.m1) + std::fabs(Rm.m5) +
                                           std::fabs(Rm.m9));
                float halfX = 0.5f * ts * (std::fabs(Rm.m0) + std::fabs(Rm.m4) +
                                           std::fabs(Rm.m8));
                tc = {tip.x, tip.y + halfY + 0.03f, tip.z};
                float xl = 0.5f * GetScreenWidth() / (float)GetScreenHeight();
                auto clampAxis = [](float v, float lo, float hi) {
                    if (lo > hi) return (lo + hi) * 0.5f;
                    return std::min(std::max(v, lo), hi);
                };
                tc.x = clampAxis(tc.x, 0.5f - xl + halfX + 0.02f,
                                 0.5f + xl - halfX - 0.02f);
                tc.y = clampAxis(tc.y, halfY + 0.02f, 1.0f - halfY - 0.02f);
                lastTc = tc; lastTq = tq; lastTs = ts;
            } else { tc = lastTc; tq = lastTq; ts = lastTs; }
        }
        if (argTestSlosh) {
            static float tsCX = 0.0f, tsAng = 0.0f;
            if (frame >= 300 && frame <= 310) tsCX += 0.03f;
            if (frame >= 400 && frame <= 420) tsAng += 4.5f;
            if (frame >= 520 && frame <= 540) tsAng += 4.5f;
            tc = {0.5f + tsCX, 0.5f, 0.5f};
            tq = QuaternionFromAxisAngle({0, 0, 1}, tsAng * DEG2RAD);
            ts = restSize;
        }
        poseC = Vector3Add(poseC, Vector3Scale(Vector3Subtract(tc, poseC), 0.3f));
        if (tq.x * poseQ.x + tq.y * poseQ.y + tq.z * poseQ.z + tq.w * poseQ.w < 0)
            tq = {-tq.x, -tq.y, -tq.z, -tq.w};
        poseQ = QuaternionSlerp(poseQ, tq, 0.25f);
        poseS += (ts - poseS) * 0.2f;
        boxM = MatrixMultiply(MatrixScale(poseS, poseS, poseS),
                              MatrixMultiply(QuaternionToMatrix(poseQ),
                                             MatrixTranslate(poseC.x, poseC.y, poseC.z)));
        invBox = MatrixMultiply(MatrixInvert(boxM), MatrixTranslate(0.5f, 0.5f, 0.5f));
        cubeModel.transform = boxM;

        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            needRefill = true;
            VbPrev = {0, 0, 0}; omegaPrev = {0, 0, 0}; forcePrimed = false;
            prevPoseC = poseC; prevPoseQ = poseQ; prevPoseS = poseS;
        }
        if (IsKeyPressed(KEY_X)) boxFollow = !boxFollow;
        if (IsKeyPressed(KEY_C)) colorMode = (colorMode + 1) % 4;
        if (IsKeyPressed(KEY_V)) sliceView = !sliceView;
        if (IsKeyPressed(KEY_B)) showWebcam = !showWebcam;
        if (IsKeyPressed(KEY_H)) showSkeleton = !showSkeleton;
        if (IsKeyPressed(KEY_SPACE)) needRefill = true;
        if (IsKeyPressed(KEY_F1)) showHelp = !showHelp;
        auto keyHit = [](int k) { return IsKeyPressed(k) || IsKeyPressedRepeat(k); };
        bool big = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
        if (keyHit(KEY_ONE)) gravity = std::max(0.0f, gravity - (big ? 0.25f : 0.05f));
        if (keyHit(KEY_TWO)) gravity = std::min(2.0f, gravity + (big ? 0.25f : 0.05f));
        if (keyHit(KEY_LEFT_BRACKET)) {
            int s = fluid.useCG ? (big ? 25 : 5) : (big ? 5 : 1);
            if (fluid.useCG) fluid.cgMaxIter = std::max(5, fluid.cgMaxIter - s);
            else fluid.iter = std::max(1, fluid.iter - s);
        }
        if (keyHit(KEY_RIGHT_BRACKET)) {
            int s = fluid.useCG ? (big ? 25 : 5) : (big ? 5 : 1);
            if (fluid.useCG) fluid.cgMaxIter = std::min(200, fluid.cgMaxIter + s);
            else fluid.iter = std::min(64, fluid.iter + s);
        }
        if (keyHit(KEY_FIVE)) fluid.vorticity = std::max(0.0f, fluid.vorticity - (big ? 1.0f : 0.2f));
        if (keyHit(KEY_SIX)) fluid.vorticity = std::min(10.0f, fluid.vorticity + (big ? 1.0f : 0.2f));
        if (keyHit(KEY_SEVEN)) sigmaScale = std::max(5.0f, sigmaScale / (big ? 2.0f : 1.25f));
        if (keyHit(KEY_EIGHT)) sigmaScale = std::min(300.0f, sigmaScale * (big ? 2.0f : 1.25f));
        if (keyHit(KEY_NINE)) iso = std::max(0.15f, iso - (big ? 0.15f : 0.05f));
        if (keyHit(KEY_ZERO)) iso = std::min(0.85f, iso + (big ? 0.15f : 0.05f));
        if (keyHit(KEY_MINUS)) inertia = std::max(0.0f, inertia - (big ? 0.5f : 0.1f));
        if (keyHit(KEY_EQUAL)) inertia = std::min(2.0f, inertia + (big ? 0.5f : 0.1f));
        if (keyHit(KEY_COMMA)) { fillLevel = std::max(0.05f, fillLevel - (big ? 0.15f : 0.05f)); needRefill = true; }
        if (keyHit(KEY_PERIOD)) { fillLevel = std::min(0.8f, fillLevel + (big ? 0.15f : 0.05f)); needRefill = true; }
        if (IsKeyPressed(KEY_M)) fluid.advMode = (fluid.advMode + 1) % 3;
        if (IsKeyPressed(KEY_G)) fluid.useCG = !fluid.useCG;
        if (IsKeyPressed(KEY_L)) lightingOn = !lightingOn;
        if (IsKeyDown(KEY_LEFT)) lightAzim -= 0.02f;
        if (IsKeyDown(KEY_RIGHT)) lightAzim += 0.02f;
        if (IsKeyDown(KEY_UP)) lightElev = std::min(1.5f, lightElev + 0.02f);
        if (IsKeyDown(KEY_DOWN)) lightElev = std::max(0.05f, lightElev - 0.02f);

        if (!paused) {
            Matrix Ri = QuaternionToMatrix(QuaternionInvert(poseQ));
            if (needRefill) {
                Vector3 upL = Vector3Transform({0, 1, 0}, Ri);
                fluid.refill(upL.x, upL.y, upL.z, fillLevel);
                needRefill = false;
            }

            Vector3 hit;
            Vector2 mpos = GetMousePosition();
            bool hitOk = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
            if (hitOk) {
                if (sliceView) {
                    Ray ray = GetScreenToWorldRay(mpos, cam);
                    Vector3 ro = Vector3Transform(ray.position, invBox);
                    Vector3 rd = Vector3Scale(Vector3Transform(ray.direction, Ri),
                                              1.0f / poseS);
                    if (std::fabs(rd.z) > 1e-6f) {
                        float tt = (0.5f - ro.z) / rd.z;
                        hit = Vector3Add(ro, Vector3Scale(rd, tt));
                        hit.z = 0.5f;
                        hitOk = tt > 0 && hit.x >= 0 && hit.x <= 1 && hit.y >= 0 && hit.y <= 1;
                    } else hitOk = false;
                } else {
                    Vector3 w;
                    hitOk = mouseHitOnPlane(mpos, w);
                    if (hitOk) {
                        hit = Vector3Transform(w, invBox);
                        hitOk = hit.x >= 0 && hit.x <= 1 && hit.y >= 0 && hit.y <= 1 &&
                                hit.z >= 0 && hit.z <= 1;
                    }
                }
            }
            if (hitOk) {
                injectAt(hit, mouseDown, prevHit);
                prevHit = hit;
                mouseDown = true;
            } else {
                mouseDown = false;
            }
            if (testDrag && frame >= 300 && frame <= 600) {
                float a = (frame - 300) * 6.2831853f / 120.0f;
                float ap = (frame - 301) * 6.2831853f / 120.0f;
                if (sliceView) {
                    Vector3 h1{0.5f + 0.25f * std::cos(a), 0.5f + 0.25f * std::sin(a), 0.5f};
                    Vector3 h0{0.5f + 0.25f * std::cos(ap), 0.5f + 0.25f * std::sin(ap), 0.5f};
                    injectAt(h1, frame > 300, h0);
                } else {
                    Vector3 p1{0.5f + 0.25f * std::cos(a), 0.5f, 0.5f + 0.25f * std::sin(a)};
                    Vector3 p0{0.5f + 0.25f * std::cos(ap), 0.5f, 0.5f + 0.25f * std::sin(ap)};
                    injectAt(Vector3Transform(p1, invBox), frame > 300,
                             Vector3Transform(p0, invBox));
                }
            }

            float mv2 = fluid.maxVel;
            for (int si = 0; si < 2; si++) {
                Slot& sl = slots[si];
                if (!sl.active || si == holderSlot) continue;
                for (int k = 0; k < 6; k++) {
                    Vector3 p = Vector3Transform(sl.pos[k], invBox);
                    if (sliceView) p.z = 0.5f;
                    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1 || p.z < 0 || p.z > 1) continue;
                    Vector3 vw = sl.vel[k];
                    float speed = Vector3Length(vw);
                    float s = std::min(std::max((speed - 0.05f) / 0.95f, 0.0f), 1.0f);
                    if (s <= 0) continue;
                    Vector3 vl = Vector3Transform(vw, Ri);
                    float vs = velScale / poseS;
                    float ax = std::min(std::max(vl.x * vs, -mv2), mv2);
                    float ay = std::min(std::max(vl.y * vs, -mv2), mv2);
                    float az = std::min(std::max(vl.z * vs, -mv2), mv2);
                    Vector3 g{p.x * N, p.y * N, p.z * N};
                    if (k == 5) {
                        fluid.blendVelocitySphere(g.x, g.y, g.z, 6.0f, ax, ay, az, 0.5f * s);
                    } else {
                        fluid.blendVelocitySphere(g.x, g.y, g.z, 3.5f, ax, ay, az, 0.6f * s);
                    }
                }
            }

            {
                Vector3 up = Vector3Transform({0, 1, 0}, Ri);
                fluid.upX = up.x; fluid.upY = up.y; fluid.upZ = up.z;
                fluid.buoyancy = -gravity;
                Vector3 VbW = Vector3Scale(Vector3Subtract(poseC, prevPoseC),
                                           1.0f / (poseS * fluid.dt));
                Quaternion qd = QuaternionMultiply(poseQ, QuaternionInvert(prevPoseQ));
                if (qd.w < 0) qd = {-qd.x, -qd.y, -qd.z, -qd.w};
                Vector3 axA; float ang;
                QuaternionToAxisAngle(qd, &axA, &ang);
                if (ang > 0.3f) ang = 0.3f;
                Vector3 w = Vector3Transform(Vector3Scale(axA, ang / fluid.dt), Ri);
                if (forcePrimed) {
                    Vector3 dVw = Vector3Subtract(VbW, VbPrev);
                    float dl = Vector3Length(dVw);
                    if (dl > 0.5f) dVw = Vector3Scale(dVw, 0.5f / dl);
                    Vector3 dV = Vector3Transform(dVw, Ri);
                    Vector3 al = Vector3Scale(Vector3Subtract(w, omegaPrev), 1.0f / fluid.dt);
                    mxW = std::max(mxW, Vector3Length(w));
                    mxDV = std::max(mxDV, Vector3Length(dVw));
                    mxAL = std::max(mxAL, Vector3Length(al));
                    if (Vector3Length(dV) + Vector3Length(w) + Vector3Length(al) > 1e-3f)
                        fluid.addFrameForces(dV.x, dV.y, dV.z, w.x, w.y, w.z,
                                             al.x, al.y, al.z, inertia);
                }
                VbPrev = VbW; omegaPrev = w;
                prevPoseC = poseC; prevPoseQ = poseQ; prevPoseS = poseS;
                forcePrimed = true;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            fluid.step();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            stepMsAccum += ms;
            stepCount++;
            stepMs = (float)(stepMsAccum / stepCount);
            if (soakFrames && frame % 100 == 0) {
                double dsum = 0, dmx = 0, vmx = 0, ke = 0; long nan = 0, clampN = 0, cnt = (long)N * N * N;
                long nFluid = 0, nMush = 0;
                for (long i = 0; i < cnt; i++) {
                    float d = fluid.density[i], a = fluid.Vx[i], b = fluid.Vy[i], c = fluid.Vz[i];
                    if (!std::isfinite(d) || !std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) { nan++; continue; }
                    dsum += d; dmx = std::max(dmx, (double)d);
                    if (d > 0.1f) { nFluid++; if (d < 0.9f) nMush++; }
                    double vm = std::max(std::fabs(a), std::max(std::fabs(b), std::fabs(c)));
                    vmx = std::max(vmx, vm); ke += a * a + b * b + c * c;
                    if (vm >= fluid.maxVel * 0.999f) clampN++;
                }
                double mushy = nFluid ? (double)nMush / nFluid : 0.0;
                double pmn = 1e30, pmx = -1e30;
                for (float pv : fluid.pressB) { pmn = std::min(pmn, (double)pv); pmx = std::max(pmx, (double)pv); }
                int act = (slots[0].active ? 1 : 0) + (slots[1].active ? 1 : 0);
                printf("f=%5d dsum=%.4g dmax=%.4g mushy=%.3f vmax=%.4f ke=%.4g clamped=%ld nan=%ld cg=%d res=%.2e p=[%.3g,%.3g] ms=%.2f frm=%.2f tracker=%s hpps=%.1f slots=%d wmx=%.3g dvmx=%.3g almx=%.3g\n",
                       frame, dsum, dmx, mushy, vmx, ke, clampN, nan, fluid.lastCgIters, fluid.lastCgRelRes, pmn, pmx, ms,
                       GetFrameTime() * 1000.0f,
                       nowT - lastHandPacketTime < 1.0 ? "connected" : "waiting", handPktEma, act,
                       mxW, mxDV, mxAL);
                printf("      hrat=[%.3f,%.3f] open=%s\n",
                       hratMin > 1e8f ? 0.0f : hratMin, hratMax,
                       openT < 0 ? "-" : TextFormat("%.2f", openT));
                mxW = mxDV = mxAL = 0;
                hratMin = 1e9f; hratMax = 0.0f;
                fflush(stdout);
            }
            if (frame > 60 && frame <= 180) {
                metMs += ms;
                metCg += fluid.lastCgIters;
                metRes += fluid.lastCgRelRes;
                metBef += fluid.divRmsBefore;
                metAft += fluid.divRmsAfter;
                metFrm += GetFrameTime() * 1000.0;
                metCount++;
            }
            if (frame > 200) {
                smMs += ms; smCg += fluid.lastCgIters;
                smRes += fluid.lastCgRelRes; smN++;
            }
        } else {
            VbPrev = {0, 0, 0}; omegaPrev = {0, 0, 0}; forcePrimed = false;
            prevPoseC = poseC; prevPoseQ = poseQ; prevPoseS = poseS;
        }

        #pragma omp parallel for
        for (int z = 0; z < N; z++) {
            for (int y = 0; y < N; y++) {
                int ty = z / tilesX, tx = z - ty * tilesX;
                unsigned char* row = atlasBuf.data() + ((ty * N + y) * atlasW + tx * N) * 2;
                const float* src = fluid.density.data() + z * N * N + y * N;
                const float* srch = fluid.heat.data() + z * N * N + y * N;
                for (int x = 0; x < N; x++) {
                    float v = std::min(std::max(src[x], 0.0f), 1.0f);
                    row[x * 2] = (unsigned char)(v * 255.0f);
                    row[x * 2 + 1] = (unsigned char)(std::min(std::max(srch[x], 0.0f), 1.0f) * 255.0f);
                }
            }
        }
        UpdateTexture(atlasTex, atlasBuf.data());

        int midZ = N / 2;
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                int id = IX(x, N - 1 - y, midZ, N);
                float v = std::min(std::max(fluid.density[id], 0.0f), 1.0f);
                Color c = sliceColor(v, colorMode);
                if (colorMode == 1 || colorMode == 3) {
                    float h = std::min(std::max(fluid.heat[id], 0.0f), 1.0f);
                    c.r = (unsigned char)std::min(255.0f, c.r + h * 255.0f);
                    c.g = (unsigned char)std::min(255.0f, c.g + h * 110.0f);
                    c.b = (unsigned char)std::min(255.0f, c.b + h * 30.0f);
                }
                c.a = (unsigned char)(std::min(v * 1.5f, 1.0f) * 255.0f);
                sliceBuf[y * N + x] = c;
            }
        UpdateTexture(sliceTex, sliceBuf.data());

        Rectangle camRect{0, 0, 0, 0};
        if (showWebcam && tf.imgW > 0) {
            if (camTexW != tf.imgW || camTexH != tf.imgH) {
                if (camTex.id) UnloadTexture(camTex);
                Image ci{tf.img.data(), tf.imgW, tf.imgH, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8};
                camTex = LoadTextureFromImage(ci);
                SetTextureFilter(camTex, TEXTURE_FILTER_BILINEAR);
                camTexW = tf.imgW; camTexH = tf.imgH;
            } else if (tf.imgUpdated) {
                UpdateTexture(camTex, tf.img.data());
            }
            camRect.height = (float)GetScreenHeight();
            camRect.width = tf.camAspect * camRect.height;
            camRect.x = (GetScreenWidth() - camRect.width) * 0.5f;
            camRect.y = 0;
        }

        BeginDrawing();
        ClearBackground(Color{18, 20, 26, 255});
        if (camRect.width > 0)
            DrawTexturePro(camTex, {0, 0, (float)camTexW, (float)camTexH}, camRect, {0, 0}, 0.0f,
                           Color{110, 110, 110, 255});

        BeginMode3D(cam);
        rlPushMatrix();
        rlMultMatrixf(MatrixToFloat(boxM));
        DrawCubeWires({0, 0, 0}, 1.0f, 1.0f, 1.0f, Color{170, 200, 255, 150});
        rlPopMatrix();
        if (!sliceView) {
            if (shaderOk) {
                SetShaderValue(volShader, locCamPos, &cam.position, SHADER_UNIFORM_VEC3);
                SetShaderValueMatrix(volShader, locInvBox, invBox);
                SetShaderValue(volShader, locTilesX, &tilesX, SHADER_UNIFORM_INT);
                SetShaderValue(volShader, locTilesY, &tilesY, SHADER_UNIFORM_INT);
                SetShaderValue(volShader, locGridN, &N, SHADER_UNIFORM_INT);
                SetShaderValue(volShader, locColorMode, &colorMode, SHADER_UNIFORM_INT);
                float steps = 64.0f;
                SetShaderValue(volShader, locMarch, &steps, SHADER_UNIFORM_FLOAT);
                SetShaderValueTexture(volShader, locAtlas, atlasTex);
                Vector3 lightDir{std::cos(lightElev) * std::cos(lightAzim),
                                 std::sin(lightElev),
                                 std::cos(lightElev) * std::sin(lightAzim)};
                Vector3 lightColor{1.0f, 0.92f, 0.8f};
                float lightIntensity = 1.0f;
                float ambient = 0.35f;
                int lightingInt = lightingOn ? 1 : 0;
                int shadowSteps = 5;
                float shadowSigma = 60.0f;
                float emission = 3.5f;
                float toneExposure = 1.5f;
                SetShaderValue(volShader, locToneExposure, &toneExposure, SHADER_UNIFORM_FLOAT);
                int orthoInt = 1;
                Vector3 viewDir = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
                SetShaderValue(volShader, locOrtho, &orthoInt, SHADER_UNIFORM_INT);
                SetShaderValue(volShader, locViewDir, &viewDir, SHADER_UNIFORM_VEC3);
                SetShaderValue(volShader, locIso, &iso, SHADER_UNIFORM_FLOAT);
                SetShaderValue(volShader, locSigmaScale, &sigmaScale, SHADER_UNIFORM_FLOAT);
                SetShaderValue(volShader, locLightDir, &lightDir, SHADER_UNIFORM_VEC3);
                SetShaderValue(volShader, locLightColor, &lightColor, SHADER_UNIFORM_VEC3);
                SetShaderValue(volShader, locLightIntensity, &lightIntensity, SHADER_UNIFORM_FLOAT);
                SetShaderValue(volShader, locAmbient, &ambient, SHADER_UNIFORM_FLOAT);
                SetShaderValue(volShader, locLighting, &lightingInt, SHADER_UNIFORM_INT);
                SetShaderValue(volShader, locShadowSteps, &shadowSteps, SHADER_UNIFORM_INT);
                SetShaderValue(volShader, locShadowSigma, &shadowSigma, SHADER_UNIFORM_FLOAT);
                SetShaderValue(volShader, locEmission, &emission, SHADER_UNIFORM_FLOAT);
                SetShaderValue(volShader, locFrame, &frame, SHADER_UNIFORM_INT);
                rlEnableBackfaceCulling();
                rlSetCullFace(RL_CULL_FACE_FRONT);
                rlDisableDepthMask();
                BeginBlendMode(BLEND_ALPHA_PREMULTIPLY);
                DrawModel(cubeModel, {0, 0, 0}, 1.0f, WHITE);
                EndBlendMode();
                rlEnableDepthMask();
                rlSetCullFace(RL_CULL_FACE_BACK);
            }
        } else {
            sliceModel.transform = MatrixMultiply(
                MatrixRotateX(PI * 0.5f), boxM);
            rlDisableBackfaceCulling();
            BeginBlendMode(BLEND_ALPHA);
            DrawModel(sliceModel, {0, 0, 0}, 1.0f, WHITE);
            EndBlendMode();
            rlEnableBackfaceCulling();
            float ak = 0.111f / poseS;
            int mid = N / 2;
            for (int y = 0; y < N; y += 4)
                for (int x = 0; x < N; x += 4) {
                    int id = IX(x, N - 1 - y, mid, N);
                    Vector3 a0{(x + 0.5f) / N - 0.5f, (N - 1 - y + 0.5f) / N - 0.5f, 0.0f};
                    Vector3 a1{a0.x + fluid.Vx[id] * ak, a0.y + fluid.Vy[id] * ak, 0.0f};
                    DrawLine3D(Vector3Transform(a0, boxM), Vector3Transform(a1, boxM),
                               Color{140, 200, 255, 160});
                }
        }
        EndMode3D();

        if (showSkeleton) {
            static const int bones[][2] = {
                {0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{5,9},{9,10},{10,11},
                {11,12},{9,13},{13,14},{14,15},{15,16},{13,17},{0,17},{17,18},{18,19},{19,20}};
            static const Color slotCol[2] = {{255, 120, 80, 255}, {80, 200, 255, 255}};
            float ovh = (float)GetScreenHeight();
            float ovw = tf.camAspect * ovh;
            float ovx = (GetScreenWidth() - ovw) * 0.5f;
            for (int s = 0; s < 2; s++) {
                if (!slots[s].active || slots[s].hand < 0 || slots[s].hand >= tf.numHands) continue;
                const HandData& hd = tf.hands[slots[s].hand];
                auto px = [&](int i) {
                    return Vector2{ovx + hd.lm[i][0] * ovw, hd.lm[i][1] * ovh};
                };
                for (auto& b : bones) {
                    Vector2 a = px(b[0]), c = px(b[1]);
                    DrawLine((int)a.x, (int)a.y, (int)c.x, (int)c.y, slotCol[s]);
                }
                for (int i = 0; i < 21; i++) {
                    Vector2 p = px(i);
                    bool tip = i == 4 || i == 8 || i == 12 || i == 16 || i == 20;
                    DrawCircle((int)p.x, (int)p.y, tip ? 5.0f : 3.0f, slotCol[s]);
                }
            }
        }

        DrawFPS(10, 10);
        static const char* modeNames[4] = {"smoke", "fire", "plasma", "classic"};
        DrawText(TextFormat("N=%d  step=%.2f ms  grav=%.2f  vort=%.2f  opac=%.0f  iso=%.2f  fill=%.2f  mode=%s%s",
                            N, stepMs, gravity, fluid.vorticity,
                            sigmaScale, iso, fillLevel, modeNames[colorMode],
                            paused ? "  [PAUSED]" : ""),
                 10, 34, 16, RAYWHITE);
        const char* adv = fluid.advMode == 0 ? "SL" : (fluid.advMode == 1 ? "MAC(d)" : "MAC(d+v)");
        char pstr[64];
        if (fluid.useCG) snprintf(pstr, sizeof(pstr), "CG(%d it, res %.1e)", fluid.lastCgIters, fluid.lastCgRelRes);
        else snprintf(pstr, sizeof(pstr), "GS(%d)", fluid.iter);
        char obuf[16];
        if (openT >= 0) snprintf(obuf, sizeof(obuf), "%.2f", openT);
        else snprintf(obuf, sizeof(obuf), "-");
        DrawText(TextFormat("adv=%s  p=%s  light=%s  box=%s  inertia=%.1f  open=%s", adv, pstr,
                            lightingOn ? "on" : "off",
                            (boxFollow && holderSlot >= 0) ? "follow" : "rest", inertia, obuf),
                 10, 52, 16, RAYWHITE);
        bool conn = nowT - lastHandPacketTime < 1.0;
        int act = (slots[0].active ? 1 : 0) + (slots[1].active ? 1 : 0);
        DrawText(TextFormat("tracker: %s  hands=%d  %.0f fps   B: webcam  H: skeleton",
                            conn ? "connected" : "waiting (see tracker.log)", act, handPktEma),
                 10, 70, 16, conn ? Color{140, 230, 140, 255} : Color{230, 170, 120, 255});
        if (showHelp) {
            int y = 92;
            const char* lines[] = {
                "Hold a hand: box follows it  Other hand: stirs",
                "LMB drag: stir fluid",
                "X: box follows hand  -/=: inertia  ,/.: fill level",
                "B: webcam background  H: hand skeleton",
                "Space: refill  P: pause  R: reset",
                "C: color mode  V: slice/volume view",
                "1/2: gravity  [/]: iters",
                "5/6: vorticity  7/8: opacity  9/0: surface iso",
                "M: advect mode  G: CG/GS  L: lighting",
                "Arrows: light direction",
                "Hold key to repeat, Shift = bigger steps",
                "F1: toggle help"
            };
            for (auto l : lines) { DrawText(l, 10, y, 14, Color{200, 205, 215, 255}); y += 17; }
        }
        EndDrawing();

        auto printMetrics = [&]() {
            double inv = metCount > 0 ? 1.0 / metCount : 0.0;
            double mb = metBef * inv, ma = metAft * inv;
            printf("avg step=%.3f ms  avg frame=%.3f ms  cgIters=%.1f  cgRelRes=%.2e  divRms before=%.4e after=%.4e  ratio=%.4f\n",
                   metMs * inv, metFrm * inv, metCg * inv, metRes * inv, mb, ma, mb > 0 ? ma / mb : 0.0);
            if (smN > 0)
                printf("soak200+: step=%.2f ms  cgIters=%.1f  cgRelRes=%.2e  n=%d\n",
                       smMs / smN, smCg / smN, smRes / smN, smN);
        };
        if (argTestSlosh) {
            if (frame >= 280 && frame <= 700 && frame % 10 == 0) {
                Vector3 axA; float ang;
                QuaternionToAxisAngle(poseQ, &axA, &ang);
                Matrix Rd = QuaternionToMatrix(QuaternionInvert(poseQ));
                Vector3 dn = Vector3Transform({0, -1, 0}, Rd);
                double ds = 0, cxm = 0, cym = 0, czm = 0, ke = 0;
                for (int z = 1; z < N - 1; z++)
                    for (int y = 1; y < N - 1; y++)
                        for (int x = 1; x < N - 1; x++) {
                            int id = IX(x, y, z, N);
                            float d = fluid.density[id];
                            ds += d;
                            cxm += d * ((x + 0.5) / N - 0.5);
                            cym += d * ((y + 0.5) / N - 0.5);
                            czm += d * ((z + 0.5) / N - 0.5);
                            ke += fluid.Vx[id] * fluid.Vx[id] + fluid.Vy[id] * fluid.Vy[id] +
                                  fluid.Vz[id] * fluid.Vz[id];
                        }
                double fall = 0;
                if (ds > 0)
                    fall = (cxm / ds) * dn.x + (cym / ds) * dn.y + (czm / ds) * dn.z;
                printf("slosh f=%4d cx=%.3f ang=%.1f mass=%.5g fall=%+.3f ke=%.4g\n",
                       frame, poseC.x, ang * (float)(180.0 / 3.14159265), ds, fall, ke);
                fflush(stdout);
            }
            if (frame == 545) TakeScreenshot("testslosh_545.png");
            if (frame == 640) TakeScreenshot("testslosh_640.png");
            if (frame >= 700) break;
        }
        if (soakFrames && frame >= soakFrames) {
            TakeScreenshot(sliceView ? "soak_slice.png" : "soak_hands.png");
            printMetrics();
            break;
        }
        if (shotVolume && frame >= 180) {
            TakeScreenshot("shot.png");
            printMetrics();
            if (lumCheck) {
                float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
                for (int cx = 0; cx < 2; cx++)
                    for (int cy = 0; cy < 2; cy++)
                        for (int cz = 0; cz < 2; cz++) {
                            Vector2 s = GetWorldToScreen({(float)cx, (float)cy, (float)cz}, cam);
                            x0 = std::min(x0, s.x); x1 = std::max(x1, s.x);
                            y0 = std::min(y0, s.y); y1 = std::max(y1, s.y);
                        }
                int ix0 = std::max(0, (int)x0), ix1 = std::min(GetScreenWidth() - 1, (int)std::ceil(x1));
                int iy0 = std::max(200, (int)y0), iy1 = std::min(GetScreenHeight() - 1, (int)std::ceil(y1));
                Image img = LoadImage("shot.png");
                Color* px = LoadImageColors(img);
                double sum = 0; long n = 0;
                for (int y = iy0; y <= iy1; y++)
                    for (int x = ix0; x <= ix1; x++) {
                        Color c = px[y * img.width + x];
                        sum += 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
                        n++;
                    }
                printf("lum=%.2f\n", n ? sum / n : 0.0);
                UnloadImageColors(px);
                UnloadImage(img);
            }
            break;
        }
        if (shotSlice && frame >= 180) {
            TakeScreenshot("shot_slice.png");
            printMetrics();
            break;
        }
    }

    trackerKill();
    netShutdown();
    if (camTex.id) UnloadTexture(camTex);
    UnloadTexture(atlasTex);
    UnloadTexture(sliceTex);
    UnloadModel(cubeModel);
    UnloadShader(volShader);
    CloseWindow();
    return 0;
}
