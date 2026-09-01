#include "se_internal.h"

#include <string.h>

size_t se_utf8_decode(const uint8_t *src, size_t len, uint32_t *out, size_t out_cap, int *ok) {
    size_t i = 0, n = 0;
    *ok = 1;

    while (i < len) {
        if (n >= out_cap) {
            *ok = 0;
            return n;
        }
        uint8_t b0 = src[i];
        uint32_t cp;
        size_t need;

        if (b0 < 0x80) {
            cp = b0;
            need = 0;
        } else if ((b0 & 0xE0) == 0xC0) {
            cp = b0 & 0x1Fu;
            need = 1;
        } else if ((b0 & 0xF0) == 0xE0) {
            cp = b0 & 0x0Fu;
            need = 2;
        } else if ((b0 & 0xF8) == 0xF0) {
            cp = b0 & 0x07u;
            need = 3;
        } else {
            *ok = 0;
            return n;
        }

        if (i + need >= len && need > 0) {
            *ok = 0;
            return n;
        }

        for (size_t k = 1; k <= need; k++) {
            uint8_t bk = src[i + k];
            if ((bk & 0xC0) != 0x80) {
                *ok = 0;
                return n;
            }
            cp = (cp << 6) | (uint32_t)(bk & 0x3Fu);
        }

        if ((need == 1 && cp < 0x80) || (need == 2 && cp < 0x800) || (need == 3 && cp < 0x10000)) {
            *ok = 0;
            return n;
        }
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            *ok = 0;
            return n;
        }

        out[n++] = cp;
        i += need + 1;
    }
    return n;
}

size_t se_utf8_encode(uint32_t cp, uint8_t *dst) {
    if (cp < 0x80) {
        dst[0] = (uint8_t)cp;
        return 1;
    }
    if (cp < 0x800) {
        dst[0] = (uint8_t)(0xC0 | (cp >> 6));
        dst[1] = (uint8_t)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        dst[0] = (uint8_t)(0xE0 | (cp >> 12));
        dst[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        dst[2] = (uint8_t)(0x80 | (cp & 0x3F));
        return 3;
    }
    dst[0] = (uint8_t)(0xF0 | (cp >> 18));
    dst[1] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
    dst[2] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
    dst[3] = (uint8_t)(0x80 | (cp & 0x3F));
    return 4;
}

int se_range_contains(const se_range_t *ranges, uint32_t count, uint32_t cp) {
    uint32_t lo = 0, hi = count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid].lo)
            hi = mid;
        else if (cp > ranges[mid].hi)
            lo = mid + 1;
        else
            return 1;
    }
    return 0;
}

const se_map_entry_t *se_map_lookup(const se_map_entry_t *entries, uint32_t count, uint32_t cp) {
    uint32_t lo = 0, hi = count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (cp < entries[mid].cp)
            hi = mid;
        else if (cp > entries[mid].cp)
            lo = mid + 1;
        else
            return &entries[mid];
    }
    return NULL;
}

int se_is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B920 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}
