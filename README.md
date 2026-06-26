# Mosaic – Low-poly Image Stylization

**Mosaic** is a GPU-accelerated tool that transforms a photo into a stylised low-poly mesh. It analyses image edges, scatters points proportionally to contrast, builds a triangulation from those points, and draws the result as flat‑colored triangles.

Example usage:
| Input:                           | Output:                                        |
|----------------------------------|------------------------------------------------|
|![bird](./test%20images/bird.jpg) |![bird_output](./test%20images/bird_output.png) |

Photo by [Boris  Smokrovi](https://unsplash.com/@borisworkshop) on Unsplash

---

## Features

- **Two work modes**
  - **process:** image → `.mosaic` (+optional PNG)
  - **render:** `.mosaic` → PNG
- **Compact file format** – can save space over `.jpg` or other lossy formats, could be useful in applications with extremely limited memory
- **Extensive Config** – change density, random seed, point placement parameters, etc. without recompiling.

---

## Dependencies

- **CUDA Toolkit** (≥11.0) – `nvcc` compiler and runtime.
- [stb_image + stb_image_write](https://github.com/nothings/stb) – image loading/saving.
- [toml++](https://github.com/marzer/tomlplusplus) – TOML parsing (`toml.hpp`).
- [delaunator-cpp](https://github.com/delfrrr/delaunator-cpp) – Delaunay triangulation (`delaunator.hpp`).

Place the header files in a `../lib/` directory (or adjust paths in the source).

---

## Building

Compile with `nvcc`:
Replace the include path with your local CUDA installation directory and run:

```
nvcc -std=c++17 -O2 -I"[path to CUDA include folder]" -o mosaic src\main.cu
```

Alternatively, update the `makefile` and run:

```
make
```
---

## Usage

    ./mosaic config.toml

Must include a valid `.toml` config file with the following parameters:

### Process mode

    mode = "process"
    input_file = "photo.jpg"        # required
    custom_file = "output.mosaic"   # (empty = skip .mosaic output)
    output_png  = "output.png"      # (empty = skip PNG output)
    target_points = 5000
    gap_pixels = 0.0
    save_contrast_map = false       # save Sobel edge detection map
    min_importance = 5              # min contrast for point placement
    base_importance = 0             # 'virtual contrast' added everywhere
    contrast_power = 2              # edge strength for point placement
    seed = 1
    bg_color = [100, 100, 100]      # RGB background when gap > 0

### Render mode

    mode = "render"
    custom_file = "output.mosaic"
    output_png  = "output.png"
    bg_color = [100, 100, 100]      # optional change background color

---

## .mosaic file format

A compact binary format storing the triangulation and per‑triangle colours.

| Offset | Size | Description |
|--------|------|-------------|
| 0      | 6 B  | File signature |
| 6      | 4 B  | Version (int) |
| 10     | 4 B  | Image width (int) |
| 14     | 4 B  | Image height (int) |
| 18     | 8 B  | Gap value (double) |
| 26     | 4 B  | Number of unique vertices (int) |
| 30     | 4 B  | Number of triangles (int) |
| 34     | V*2  | Vertex array: uint16_t x, y for each vertex |
| …      | T*9  | Triangle data: three uint16_t vertex indices + three uint8_t values for RGB color|

All multi‑byte values are little‑endian.

---

## License

This project is open‑source and provided as‑is. Dependencies retain their own licenses (see their repositories).

---

## Acknowledgements

- [Sean Barrett](https://github.com/nothings) for the STB libraries
- [marzer](https://github.com/marzer) for toml++
- [delfrrr](https://github.com/delfrrr) for delaunator-cpp
- [Boris  Smokrovi](https://unsplash.com/@borisworkshop) for the example image
