#ifndef MOSAIC_RENDER_CUH
#define MOSAIC_RENDER_CUH

#include <array>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"
#include <C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include\cuda_runtime.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err_ = call;                                       \
    if (err_ != cudaSuccess) {                                     \
        printf("CUDA error at %s:%d -- %s\n", __FILE__, __LINE__,  \
               cudaGetErrorString(err_));                          \
        return;                                                    \
    }                                                              \
} while(0)

__device__ bool pointInTriangle(int px, int py, int ax, int ay, int bx, int by, int cx, int cy) {
    int sign1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
    int sign2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
    int sign3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
    bool neg = (sign1 < 0) || (sign2 < 0) || (sign3 < 0);
    bool pos = (sign1 > 0) || (sign2 > 0) || (sign3 > 0);
    return !(neg && pos);
}

__global__ void drawTrianglesColored(int w, int h, int channels,
                                            int* points, int triangles,
                                            unsigned char* colors,
                                            unsigned char* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= triangles) return;

    int base = tid * 6;
    int ax = points[base],   ay = points[base+1];
    int bx = points[base+2], by = points[base+3];
    int cx = points[base+4], cy = points[base+5];

    int minX = max(min(min(ax, bx), cx), 0);
    int minY = max(min(min(ay, by), cy), 0);
    int maxX = min(max(max(ax, bx), cx), w-1);
    int maxY = min(max(max(ay, by), cy), h-1);

    unsigned char r = colors[tid*3];
    unsigned char g = colors[tid*3+1];
    unsigned char b = colors[tid*3+2];

    for (int y = minY; y <= maxY; ++y) {
        int row = y * w;
        for (int x = minX; x <= maxX; ++x) {
            if (pointInTriangle(x, y, ax, ay, bx, by, cx, cy)) {
                int idx = (row + x) * channels;
                output[idx]   = r;
                output[idx+1] = g;
                output[idx+2] = b;
            }
        }
    }
}

__global__ void fillBackground(int w, int h, int channels,
                                      unsigned char r, unsigned char g, unsigned char b,
                                      unsigned char* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = (y * w + x) * channels;
    output[idx]   = r;
    output[idx+1] = g;
    output[idx+2] = b;
}

static void renderTrianglesToPNG(int w, int h, int channels,
                                 int* d_triVerts, int numTri,
                                 unsigned char* d_colors,
                                 const std::array<unsigned char,3>& bg_color,
                                 const std::string& outputPng)
{
    size_t img_size = (size_t)w * h * channels;
    unsigned char *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, img_size));

    // Fill background
    dim3 fillBlock(32, 32);
    dim3 fillGrid((w + 31) / 32, (h + 31) / 32);
    fillBackground<<<fillGrid, fillBlock>>>(w, h, channels,
                                            bg_color[0], bg_color[1], bg_color[2],
                                            d_output);
    CUDA_CHECK(cudaGetLastError());

    // Draw triangles
    int threadsPerBlock = 256;
    int blocks = (numTri + threadsPerBlock - 1) / threadsPerBlock;
    drawTrianglesColored<<<blocks, threadsPerBlock>>>(w, h, channels, d_triVerts, numTri,
                                                      d_colors, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Save PNG
    unsigned char* png = (unsigned char*)malloc(img_size);
    CUDA_CHECK(cudaMemcpy(png, d_output, img_size, cudaMemcpyDeviceToHost));
    stbi_write_png(outputPng.c_str(), w, h, channels, png, w * channels);
    printf("Saved %s\n", outputPng.c_str());

    free(png);
    CUDA_CHECK(cudaFree(d_output));
}

static inline long long cross(const IntPoint& a, const IntPoint& b, const IntPoint& c) {
    return (long long)(b.x - a.x) * (c.y - a.y) - (long long)(b.y - a.y) * (c.x - a.x);
}

static bool insetTriangle(const IntPoint& p0, const IntPoint& p1, const IntPoint& p2,
                          double d, IntPoint& q0, IntPoint& q1, IntPoint& q2) {
    if (d == 0.0) {
        q0 = p0; q1 = p1; q2 = p2;
        return true;
    }
    IntPoint a = p0, b = p1, c = p2;
    if (cross(a, b, c) < 0) { std::swap(b, c); }

    auto inwardNormal = [](const IntPoint& start, const IntPoint& end) -> std::pair<double,double> {
        double ex = end.x - start.x, ey = end.y - start.y;
        double len = sqrt(ex*ex + ey*ey);
        if (len == 0) return {0,0};
        return { -ey/len, ex/len };
    };

    auto n_ab = inwardNormal(a, b);
    auto n_bc = inwardNormal(b, c);
    auto n_ca = inwardNormal(c, a);

    auto intersect = [](const std::pair<double,double>& P, const std::pair<double,double>& D1,
                        const std::pair<double,double>& Q, const std::pair<double,double>& D2)
                        -> std::pair<double,double> {
        double crossD = D1.first * D2.second - D1.second * D2.first;
        if (fabs(crossD) < 1e-9) return P;
        double t = ((Q.first - P.first) * D2.second - (Q.second - P.second) * D2.first) / crossD;
        return { P.first + t * D1.first, P.second + t * D1.second };
    };

    auto new_a = intersect(
        {a.x + d * n_ca.first, a.y + d * n_ca.second}, {c.x - a.x, c.y - a.y},
        {a.x + d * n_ab.first, a.y + d * n_ab.second}, {b.x - a.x, b.y - a.y}
    );
    auto new_b = intersect(
        {b.x + d * n_ab.first, b.y + d * n_ab.second}, {a.x - b.x, a.y - b.y},
        {b.x + d * n_bc.first, b.y + d * n_bc.second}, {c.x - b.x, c.y - b.y}
    );
    auto new_c = intersect(
        {c.x + d * n_bc.first, c.y + d * n_bc.second}, {b.x - c.x, b.y - c.y},
        {c.x + d * n_ca.first, c.y + d * n_ca.second}, {a.x - c.x, a.y - c.y}
    );

    q0.x = (int)(new_a.first + 0.5); q0.y = (int)(new_a.second + 0.5);
    q1.x = (int)(new_b.first + 0.5); q1.y = (int)(new_b.second + 0.5);
    q2.x = (int)(new_c.first + 0.5); q2.y = (int)(new_c.second + 0.5);

    return cross(q0, q1, q2) > 10;
}

static std::vector<IntPoint> applyGapToVertices(const std::vector<IntPoint>& originalVerts,
                                                double gap) {
    int numTri = (int)originalVerts.size() / 3;
    std::vector<IntPoint> insetVerts(numTri * 3);
    for (int i = 0; i < numTri; ++i) {
        const IntPoint& p0 = originalVerts[i*3];
        const IntPoint& p1 = originalVerts[i*3+1];
        const IntPoint& p2 = originalVerts[i*3+2];
        IntPoint q0, q1, q2;
        insetTriangle(p0, p1, p2, gap, q0, q1, q2);
        insetVerts[i*3]   = q0;
        insetVerts[i*3+1] = q1;
        insetVerts[i*3+2] = q2;
    }
    return insetVerts;
}

#endif // MOSAIC_RENDER_CUH