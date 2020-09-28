/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   dcp.h
 */

#ifndef FTI_SRC_DCP_H_
#define FTI_SRC_DCP_H_

#include "./interface.h"

#define FTI_DCP_MODE_OFFSET 2000
#define FTI_DCP_MODE_MD5 2001
#define FTI_DCP_MODE_CRC32 2002

#ifdef FTI_NOZLIB
extern const uint32_t crc32_tab[];

static inline uint32_t crc32_raw(const void *buf, size_t size, uint32_t crc) {
    const uint8_t *p = (const uint8_t *)buf;

    while (size--)
        crc = crc32_tab[(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    return (crc);
}

static inline uint32_t crc32(const void *buf, size_t size) {
    uint32_t crc;

    crc = crc32_raw(buf, size, ~0U);
    return (crc ^ ~0U);
}
#endif

void FTI_PrintDcpStats(FTIT_configuration FTI_Conf, FTIT_execution FTI_Exec,
 FTIT_topology FTI_Topo);

#endif  // FTI_SRC_DCP_H_
