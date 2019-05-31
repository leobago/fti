#ifndef __DCP_H__
#define __DCP_H__

#ifdef FTI_NOZLIB
extern const uint32_t crc32_tab[];

static inline uint32_t crc32_raw(const void *buf, size_t size, uint32_t crc)
{
    const uint8_t *p = (const uint8_t *)buf;

    while (size--)
        crc = crc32_tab[(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    return (crc);
}

static inline uint32_t crc32(const void *buf, size_t size)
{
    uint32_t crc;

    crc = crc32_raw(buf, size, ~0U);
    return (crc ^ ~0U);
}
#endif

#endif // __DCP_H__
