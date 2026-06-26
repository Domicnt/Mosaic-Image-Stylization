#ifndef MOSAIC_IO_H
#define MOSAIC_IO_H

#include <cstdio>
#include <cstdint>
#include <vector>
#include <map>
#include <utility>

struct IntPoint { int x, y; };

static bool saveMosaicFile(const char* filename,
                           int w, int h, double gap,
                           const std::vector<IntPoint>& vertices,
                           const std::vector<unsigned char>& colors)
{
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("ERROR: cannot write %s\n", filename); return false; }

    std::map<std::pair<int,int>, uint16_t> vertexMap;
    std::vector<uint16_t> uniqueX, uniqueY;
    std::vector<uint16_t> triIndices;
    triIndices.reserve(vertices.size());

    for (const auto& p : vertices) {
        auto key = std::make_pair(p.x, p.y);
        auto it = vertexMap.find(key);
        if (it == vertexMap.end()) {
            uint16_t idx = (uint16_t)uniqueX.size();
            vertexMap[key] = idx;
            uniqueX.push_back((uint16_t)p.x);
            uniqueY.push_back((uint16_t)p.y);
            triIndices.push_back(idx);
        } else {
            triIndices.push_back(it->second);
        }
    }

    const char magic[] = "MOSAIC";
    fwrite(magic, 1, 6, f);
    int version = 1;
    fwrite(&version, sizeof(int), 1, f);
    fwrite(&w, sizeof(int), 1, f);
    fwrite(&h, sizeof(int), 1, f);
    fwrite(&gap, sizeof(double), 1, f);

    int numVertices = (int)uniqueX.size();
    int numTri      = (int)(triIndices.size() / 3);
    fwrite(&numVertices, sizeof(int), 1, f);
    fwrite(&numTri,      sizeof(int), 1, f);

    fwrite(uniqueX.data(), sizeof(uint16_t), numVertices, f);
    fwrite(uniqueY.data(), sizeof(uint16_t), numVertices, f);

    for (int i = 0; i < numTri; ++i) {
        uint16_t idx[3] = { triIndices[i*3], triIndices[i*3+1], triIndices[i*3+2] };
        fwrite(idx, sizeof(uint16_t), 3, f);
        fwrite(&colors[i*3], 1, 3, f);
    }

    fclose(f);
    printf("Saved %s (%d vertices, %d triangles, gap %.1f)\n", filename, numVertices, numTri, gap);
    return true;
}

static bool loadMosaicFile(const char* filename,
                           int& w, int& h, double& gap,
                           std::vector<IntPoint>& vertices,
                           std::vector<unsigned char>& colors)
{
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", filename); return false; }

    char magic[7] = {};
    fread(magic, 1, 6, f);
    if (strncmp(magic, "MOSAIC", 6) != 0) {
        printf("ERROR: invalid mosaic file\n");
        fclose(f);
        return false;
    }
    int version;
    fread(&version, sizeof(int), 1, f);
    if (version != 1) {
        printf("ERROR: unsupported mosaic version %d (expected 1)\n", version);
        fclose(f);
        return false;
    }
    fread(&w, sizeof(int), 1, f);
    fread(&h, sizeof(int), 1, f);
    fread(&gap, sizeof(double), 1, f);

    int numVertices, numTri;
    fread(&numVertices, sizeof(int), 1, f);
    fread(&numTri,      sizeof(int), 1, f);

    std::vector<uint16_t> vx(numVertices), vy(numVertices);
    fread(vx.data(), sizeof(uint16_t), numVertices, f);
    fread(vy.data(), sizeof(uint16_t), numVertices, f);

    vertices.clear();
    colors.clear();
    vertices.reserve(numTri * 3);
    colors.reserve(numTri * 3);

    for (int i = 0; i < numTri; ++i) {
        uint16_t idx[3];
        fread(idx, sizeof(uint16_t), 3, f);
        unsigned char rgb[3];
        fread(rgb, 1, 3, f);

        vertices.push_back({(int)vx[idx[0]], (int)vy[idx[0]]});
        vertices.push_back({(int)vx[idx[1]], (int)vy[idx[1]]});
        vertices.push_back({(int)vx[idx[2]], (int)vy[idx[2]]});
        colors.push_back(rgb[0]);
        colors.push_back(rgb[1]);
        colors.push_back(rgb[2]);
    }

    fclose(f);
    printf("Loaded %s (%d vertices, %d triangles, gap %.1f)\n", filename, numVertices, numTri, gap);
    return true;
}

#endif // MOSAIC_IO_H