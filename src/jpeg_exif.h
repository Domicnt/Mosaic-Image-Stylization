#ifndef JPEG_EXIF_H
#define JPEG_EXIF_H

#include <cstdio>
#include <cstring>
#include <vector>

// Read JPEG EXIF orientation (1‑8). Returns 1 if not found or not JPEG.
static int readJpegOrientation(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 1;

    unsigned char soi[2];
    if (fread(soi, 1, 2, f) != 2 || soi[0] != 0xFF || soi[1] != 0xD8) {
        fclose(f);
        return 1;
    }

    while (1) {
        unsigned char marker[2];
        if (fread(marker, 1, 2, f) != 2) break;
        if (marker[0] != 0xFF) break;
        if (marker[1] == 0xFF) continue;

        if (marker[1] == 0xE1) {
            unsigned short len;
            if (fread(&len, 1, 2, f) != 2) break;
            len = (len << 8) | (len >> 8);
            std::vector<unsigned char> data(len - 2);
            if (fread(data.data(), 1, len - 2, f) != len - 2) break;

            if (len - 2 < 6 || memcmp(data.data(), "Exif\0\0", 6) != 0) {
                fseek(f, (long)(len - 2), SEEK_CUR);
                continue;
            }

            unsigned char* tiff = data.data() + 6;
            bool littleEndian = (tiff[0] == 'I' && tiff[1] == 'I');
            bool bigEndian    = (tiff[0] == 'M' && tiff[1] == 'M');
            if (!littleEndian && !bigEndian) break;

            unsigned int ifdOffset = 0;
            if (littleEndian) {
                ifdOffset = tiff[4] | (tiff[5] << 8) | (tiff[6] << 16) | (tiff[7] << 24);
            } else {
                ifdOffset = (tiff[4] << 24) | (tiff[5] << 16) | (tiff[6] << 8) | tiff[7];
            }
            if (ifdOffset + 2 > (unsigned int)(len - 6)) break;

            unsigned short numEntries = 0;
            if (littleEndian) {
                numEntries = tiff[ifdOffset] | (tiff[ifdOffset+1] << 8);
            } else {
                numEntries = (tiff[ifdOffset] << 8) | tiff[ifdOffset+1];
            }

            for (unsigned short i = 0; i < numEntries; ++i) {
                int entryPos = ifdOffset + 2 + i * 12;
                if (entryPos + 12 > len - 6) break;

                unsigned short tag = 0;
                if (littleEndian) {
                    tag = tiff[entryPos] | (tiff[entryPos+1] << 8);
                } else {
                    tag = (tiff[entryPos] << 8) | tiff[entryPos+1];
                }

                if (tag == 0x0112) {
                    unsigned short type = 0;
                    if (littleEndian) {
                        type = tiff[entryPos+2] | (tiff[entryPos+3] << 8);
                    } else {
                        type = (tiff[entryPos+2] << 8) | tiff[entryPos+3];
                    }
                    if (type == 3) {
                        unsigned short val = 0;
                        if (littleEndian) {
                            val = tiff[entryPos+8] | (tiff[entryPos+9] << 8);
                        } else {
                            val = (tiff[entryPos+8] << 8) | tiff[entryPos+9];
                        }
                        fclose(f);
                        return val;
                    }
                }
            }
            break;
        } else {
            unsigned short len;
            if (fread(&len, 1, 2, f) != 2) break;
            len = (len << 8) | (len >> 8);
            if (len < 2) break;
            fseek(f, len - 2, SEEK_CUR);
        }
    }

    fclose(f);
    return 1;
}

#endif // JPEG_EXIF_H