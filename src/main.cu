#include <stdlib.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <cmath>

#include <C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include\cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#include "../lib/toml.hpp"
#include "../lib/delaunator.hpp"
#include "jpeg_exif.h"
#include "mosaic_io.h"
#include "mosaic_render.cuh"

// CUDA error checking helper
#define CUDA_CHECK(call) do {                                    \
    cudaError_t err_ = call;                                     \
    if (err_ != cudaSuccess) {                                   \
        printf("CUDA error at %s:%d -- %s\n", __FILE__, __LINE__,  \
               cudaGetErrorString(err_));                        \
        return;                                                  \
    }                                                            \
} while(0)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
__global__
void sobelGradient(int w, int h, int channels, unsigned char* img, unsigned char* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

    int gxR = 0, gyR = 0;
    int gxG = 0, gyG = 0;
    int gxB = 0, gyB = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = max(0, min(w-1, x + dx));
            int ny = max(0, min(h-1, y + dy));
            int idx = (ny * w + nx) * channels;

            unsigned char r = img[idx];
            unsigned char g = img[idx+1];
            unsigned char b = img[idx+2];

            int kw = Gx[dy+1][dx+1];
            gxR += r * kw;  gxG += g * kw;  gxB += b * kw;

            kw = Gy[dy+1][dx+1];
            gyR += r * kw;  gyG += g * kw;  gyB += b * kw;
        }
    }

    float mag = sqrtf((float)(gxR*gxR + gxG*gxG + gxB*gxB +
                              gyR*gyR + gyG*gyG + gyB*gyB));
    mag = min(255.0f, mag);

    int pixel = (y * w + x) * channels;
    output[pixel] = output[pixel+1] = output[pixel+2] = (unsigned char)mag;
}

__global__ void samplePointsKernel(
    const unsigned char* contrast, int w, int h, int channels,
    unsigned char minImportance, unsigned char baseImp,
    float probScale, float exponent, unsigned int seed,
    int* pointBuffer, int* pointCount, int maxPoints)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = (y * w + x) * channels;
    unsigned char imp = contrast[idx] + baseImp;
    if (imp < minImportance) return;

    unsigned int hash = seed + y * w + x;
    hash = (hash ^ 61) ^ (hash >> 16);
    hash = hash + (hash << 3);
    hash = hash ^ (hash >> 4);
    hash = hash * 0x27d4eb2d;
    hash = hash ^ (hash >> 15);
    float randVal = (float)(hash & 0xFFFF) / 65536.0f;

    float excess = (float)(imp - minImportance);
    float prob = powf(excess, exponent) * probScale;   // power-law contrast boost

    if (randVal < prob) {
        int slot = atomicAdd(pointCount, 1);
        if (slot < maxPoints) {
            pointBuffer[slot * 2]     = x;
            pointBuffer[slot * 2 + 1] = y;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU helper: average triangle colour
// ---------------------------------------------------------------------------
static void computeTriangleColor(const unsigned char* img, int w, int h, int channels,
                                 const IntPoint& p0, const IntPoint& p1, const IntPoint& p2,
                                 unsigned char& r, unsigned char& g, unsigned char& b) {
    int ax = p0.x, ay = p0.y, bx = p1.x, by = p1.y, cx = p2.x, cy = p2.y;
    int minX = std::max(0, std::min({ax, bx, cx}));
    int maxX = std::min(w-1, std::max({ax, bx, cx}));
    int minY = std::max(0, std::min({ay, by, cy}));
    int maxY = std::min(h-1, std::max({ay, by, cy}));
    long long sumR = 0, sumG = 0, sumB = 0;
    int count = 0;
    for (int y = minY; y <= maxY; ++y) {
        int row = y * w;
        for (int x = minX; x <= maxX; ++x) {
            int sign1 = (x - bx) * (ay - by) - (ax - bx) * (y - by);
            int sign2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy);
            int sign3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay);
            bool neg = (sign1 < 0) || (sign2 < 0) || (sign3 < 0);
            bool pos = (sign1 > 0) || (sign2 > 0) || (sign3 > 0);
            if (!(neg && pos)) {
                int idx = (row + x) * channels;
                sumR += img[idx];
                sumG += img[idx+1];
                sumB += img[idx+2];
                ++count;
            }
        }
    }
    if (count == 0) { r = g = b = 0; return; }
    r = (unsigned char)(sumR / count);
    g = (unsigned char)(sumG / count);
    b = (unsigned char)(sumB / count);
}

// ---------------------------------------------------------------------------
// Process mode
// ---------------------------------------------------------------------------
static void processImage(const std::string& imgPath,
                         int target_points, double gap_val, bool save_contrast,
                         int min_importance, int base_importance,
                         unsigned int seed_val, double contrast_power,
                         const std::string& customFile, const std::string& outputPng,
                         const std::array<unsigned char,3>& bg_color)
{
    // ----- Load image -----
    int w, h, channels;
    unsigned char* img = stbi_load(imgPath.c_str(), &w, &h, &channels, STBI_rgb);
    if (!img) {
        printf("ERROR: cannot load image '%s'\n", imgPath.c_str());
        printf("  STB failure reason: %s\n", stbi_failure_reason());
        return;
    }
    channels = 3;
    size_t img_size = w * h * channels;

    // ----- Handle EXIF orientation -----
    int orientation = readJpegOrientation(imgPath.c_str());

    if (orientation >= 2 && orientation <= 8) {
        unsigned char* rotated = nullptr;
        int newW = w, newH = h;

        switch (orientation) {
            case 2: // flip horizontally
                rotated = (unsigned char*)malloc(img_size);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        memcpy(rotated + (y * w + (w - 1 - x)) * 3,
                               img + (y * w + x) * 3, 3);
                break;
            case 3: // rotate 180
                rotated = (unsigned char*)malloc(img_size);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        memcpy(rotated + ((h - 1 - y) * w + (w - 1 - x)) * 3,
                               img + (y * w + x) * 3, 3);
                break;
            case 4: // flip vertically
                rotated = (unsigned char*)malloc(img_size);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        memcpy(rotated + ((h - 1 - y) * w + x) * 3,
                               img + (y * w + x) * 3, 3);
                break;
            case 5: // transpose
                newW = h; newH = w;
                rotated = (unsigned char*)malloc((size_t)newW * newH * 3);
                for (int y = 0; y < newH; ++y)
                    for (int x = 0; x < newW; ++x)
                        memcpy(rotated + (y * newW + x) * 3,
                               img + ((newW - 1 - x) * w + (newH - 1 - y)) * 3, 3);
                break;
            case 6: // rotate 90 CW
                newW = h; newH = w;
                rotated = (unsigned char*)malloc((size_t)newW * newH * 3);
                for (int y = 0; y < newH; ++y)
                    for (int x = 0; x < newW; ++x)
                        memcpy(rotated + (y * newW + x) * 3,
                               img + ((newW - 1 - x) * w + y) * 3, 3);
                break;
            case 7: // transverse
                newW = h; newH = w;
                rotated = (unsigned char*)malloc((size_t)newW * newH * 3);
                for (int y = 0; y < newH; ++y)
                    for (int x = 0; x < newW; ++x)
                        memcpy(rotated + (y * newW + x) * 3,
                               img + (x * w + y) * 3, 3);
                break;
            case 8: // rotate 90 CCW
                newW = h; newH = w;
                rotated = (unsigned char*)malloc((size_t)newW * newH * 3);
                for (int y = 0; y < newH; ++y)
                    for (int x = 0; x < newW; ++x)
                        memcpy(rotated + (y * newW + x) * 3,
                               img + (x * w + (newH - 1 - y)) * 3, 3);
                break;
        }

        if (rotated) {
            stbi_image_free(img);
            img = rotated;
            w = newW;
            h = newH;
            img_size = (size_t)w * h * 3;
        }
    }

    // ----- CUDA: copy image to device -----
    unsigned char *gpu_img_in, *gpu_img_out;
    CUDA_CHECK(cudaMalloc(&gpu_img_in, img_size));
    CUDA_CHECK(cudaMalloc(&gpu_img_out, img_size));
    CUDA_CHECK(cudaMemcpy(gpu_img_in, img, img_size, cudaMemcpyHostToDevice));
    unsigned char* host_img = img;

    // ----- Sobel -----
    dim3 blockSizeSobel(32, 32);
    dim3 gridSizeSobel((w + 31) / 32, (h + 31) / 32);
    sobelGradient<<<gridSizeSobel, blockSizeSobel>>>(w, h, channels, gpu_img_in, gpu_img_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----- Copy contrast map to host -----
    unsigned char* contrast_host = (unsigned char*)malloc(img_size);
    CUDA_CHECK(cudaMemcpy(contrast_host, gpu_img_out, img_size, cudaMemcpyDeviceToHost));

    if (save_contrast) {
        std::vector<unsigned char> grey(w * h);
        for (int i = 0; i < w * h; ++i) grey[i] = contrast_host[i * channels];
        stbi_write_png("contrast_map.png", w, h, 1, grey.data(), w);
        printf("Saved contrast_map.png\n");
    }

    // ----- Sampling parameters (power-law aware) -----
    unsigned char minImp = (unsigned char)std::max(min_importance, 1);
    unsigned char baseImp = (unsigned char)base_importance;

    double totalWeighted = 0.0;
    for (int i = 0; i < w * h; ++i) {
        unsigned char v = contrast_host[i * channels] + baseImp;
        if (v >= minImp) {
            double excess = (double)(v - minImp);
            totalWeighted += pow(excess, contrast_power);
        }
    }

    float probScale = 0.0f;
    if (totalWeighted > 0.0 && target_points > 0) {
        probScale = (float)target_points / (float)totalWeighted;
    }

    free(contrast_host);

    // ----- GPU point sampling (pass exponent to kernel) -----
    int maxPoints = w * h;
    int *d_points, *d_pointCount;
    CUDA_CHECK(cudaMalloc(&d_points, maxPoints * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pointCount, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pointCount, 0, sizeof(int)));

    dim3 blockSampler(16, 16);
    dim3 gridSampler((w + 15) / 16, (h + 15) / 16);
    samplePointsKernel<<<gridSampler, blockSampler>>>(
        gpu_img_out, w, h, channels, minImp, baseImp,
        probScale, (float)contrast_power, seed_val,
        d_points, d_pointCount, maxPoints);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int numPoints;
    CUDA_CHECK(cudaMemcpy(&numPoints, d_pointCount, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<IntPoint> hostPoints(numPoints);
    std::vector<int> tmp(numPoints * 2);
    CUDA_CHECK(cudaMemcpy(tmp.data(), d_points, numPoints * 2 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < numPoints; ++i) {
        hostPoints[i].x = tmp[i*2];
        hostPoints[i].y = tmp[i*2+1];
    }
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_pointCount));
    CUDA_CHECK(cudaFree(gpu_img_out));

    // Add mandatory corners and boundary points
    hostPoints.push_back({0,0}); hostPoints.push_back({w-1,0});
    hostPoints.push_back({w-1,h-1}); hostPoints.push_back({0,h-1});

    // ----- Delaunay triangulation -----
    std::vector<double> coords;
    for (const auto& p : hostPoints) {
        coords.push_back((double)p.x);
        coords.push_back((double)p.y);
    }
    delaunator::Delaunator d(coords);
    int numTri = (int)d.triangles.size() / 3;

    // ----- Compute triangle colours -----
    std::vector<unsigned char> triColors(numTri * 3);
    std::vector<IntPoint> triVerts(numTri * 3);
    for (size_t i = 0; i < d.triangles.size(); i += 3) {
        int a = d.triangles[i], b = d.triangles[i+1], c = d.triangles[i+2];
        triVerts[(i/3)*3 + 0] = hostPoints[a];
        triVerts[(i/3)*3 + 1] = hostPoints[b];
        triVerts[(i/3)*3 + 2] = hostPoints[c];
        computeTriangleColor(host_img, w, h, channels,
                             hostPoints[a], hostPoints[b], hostPoints[c],
                             triColors[(i/3)*3 + 0], triColors[(i/3)*3 + 1], triColors[(i/3)*3 + 2]);
    }

    // ----- Optional save .mosaic file -----
    if (!customFile.empty()) {
        saveMosaicFile(customFile.c_str(), w, h, gap_val, triVerts, triColors);
    }

    // ----- Optional PNG rendering -----
    if (!outputPng.empty()) {
        std::vector<IntPoint> finalVerts = applyGapToVertices(triVerts, gap_val);

        int numCoords = numTri * 6;
        std::vector<int> flatVerts(numCoords);
        for (int i = 0; i < numTri; ++i) {
            flatVerts[i*6+0] = finalVerts[i*3].x;   flatVerts[i*6+1] = finalVerts[i*3].y;
            flatVerts[i*6+2] = finalVerts[i*3+1].x; flatVerts[i*6+3] = finalVerts[i*3+1].y;
            flatVerts[i*6+4] = finalVerts[i*3+2].x; flatVerts[i*6+5] = finalVerts[i*3+2].y;
        }

        int *d_triVerts;
        unsigned char *d_colors;
        CUDA_CHECK(cudaMalloc(&d_triVerts, numCoords * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_triVerts, flatVerts.data(), numCoords * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_colors, numTri * 3));
        CUDA_CHECK(cudaMemcpy(d_colors, triColors.data(), numTri * 3, cudaMemcpyHostToDevice));

        renderTrianglesToPNG(w, h, channels, d_triVerts, numTri, d_colors, bg_color, outputPng);

        CUDA_CHECK(cudaFree(d_triVerts));
        CUDA_CHECK(cudaFree(d_colors));
    }

    stbi_image_free(host_img);
    CUDA_CHECK(cudaFree(gpu_img_in));
}

// ---------------------------------------------------------------------------
// Render mode
// ---------------------------------------------------------------------------
static void renderMosaic(const std::string& mosaicFile, const std::string& outputPng,
                         const std::array<unsigned char,3>& bg_color) {
    int w, h;
    double gap;
    std::vector<IntPoint> originalVerts;
    std::vector<unsigned char> colors;

    if (!loadMosaicFile(mosaicFile.c_str(), w, h, gap, originalVerts, colors)) return;

    std::vector<IntPoint> finalVerts = applyGapToVertices(originalVerts, gap);

    int numTri = (int)finalVerts.size() / 3;
    int numCoords = numTri * 6;
    std::vector<int> flatVerts(numCoords);
    for (int i = 0; i < numTri; ++i) {
        flatVerts[i*6+0] = finalVerts[i*3].x;   flatVerts[i*6+1] = finalVerts[i*3].y;
        flatVerts[i*6+2] = finalVerts[i*3+1].x; flatVerts[i*6+3] = finalVerts[i*3+1].y;
        flatVerts[i*6+4] = finalVerts[i*3+2].x; flatVerts[i*6+5] = finalVerts[i*3+2].y;
    }

    int *d_triVerts;
    unsigned char *d_colors;
    CUDA_CHECK(cudaMalloc(&d_triVerts, numCoords * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_triVerts, flatVerts.data(), numCoords * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_colors, numTri * 3));
    CUDA_CHECK(cudaMemcpy(d_colors, colors.data(), numTri * 3, cudaMemcpyHostToDevice));

    renderTrianglesToPNG(w, h, 3, d_triVerts, numTri, d_colors, bg_color, outputPng);

    CUDA_CHECK(cudaFree(d_triVerts));
    CUDA_CHECK(cudaFree(d_colors));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Must include a config file, exiting.\n");
        return 1;
    }

    int target_points = 5000;
    double gap_val = 0.0;
    bool save_contrast = false;
    int min_importance = 5;
    int base_importance = 0;
    unsigned int seed_val = 42;
    double contrast_power = 1.0;
    std::string mode = "process";
    std::string inputFile, customFile, outputPng;
    std::array<unsigned char,3> bg_color = {100, 100, 100};

    try {
        auto config = toml::parse_file(argv[1]);
        mode            = config["mode"].value_or("process");
        inputFile       = config["input_file"].value_or("");
        customFile      = config["custom_file"].value_or("output.mosaic");
        outputPng       = config["output_png"].value_or("output.png");
        target_points   = config["target_points"].value_or(5000);
        gap_val         = config["gap_pixels"].value_or(0.0) / 2.0;
        save_contrast   = config["save_contrast_map"].value_or(false);
        min_importance  = config["min_importance"].value_or(5);
        base_importance = config["base_importance"].value_or(0);
        seed_val        = config["seed"].value_or(42u);
        contrast_power  = config["contrast_power"].value_or(1.0);

        if (config.contains("bg_color") && config["bg_color"].is_array()) {
            auto arr = *config["bg_color"].as_array();
            if (arr.size() == 3 && arr[0].is_integer() && arr[1].is_integer() && arr[2].is_integer()) {
                bg_color[0] = (unsigned char)arr[0].value_or(100);
                bg_color[1] = (unsigned char)arr[1].value_or(100);
                bg_color[2] = (unsigned char)arr[2].value_or(100);
            }
        }
    } catch (...) {
        printf("Could not parse config.toml, using defaults.\n");
    }

    printf("Mode: %s\n", mode.c_str());

    if (mode == "process") {
        if (inputFile.empty()) { printf("ERROR: input_file required\n"); return 1; }
        processImage(inputFile, target_points, gap_val, save_contrast,
                     min_importance, base_importance, seed_val,
                     contrast_power,
                     customFile, outputPng, bg_color);
    } else if (mode == "render") {
        if (customFile.empty()) { printf("ERROR: custom_file required\n"); return 1; }
        if (outputPng.empty()) outputPng = "../output.png";
        renderMosaic(customFile, outputPng, bg_color);
    } else {
        printf("ERROR: unknown mode '%s'. Use 'process' or 'render'.\n", mode.c_str());
        return 1;
    }

    return 0;
}