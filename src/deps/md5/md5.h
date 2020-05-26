/*
 * This is an OpenSSL-compatible implementation of the RSA Data Security, Inc.
 * MD5 Message-Digest Algorithm (RFC 1321).
 *
 * Homepage:
 * http://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
 *
 * Author:
 * Alexander Peslyak, better known as Solar Designer <solar at openwall.com>
 *
 * This software was written by Alexander Peslyak in 2001.  No copyright is
 * claimed, and the software is hereby placed in the public domain.
 * In case this attempt to disclaim copyright and place the software in the
 * public domain is deemed null and void, then the software is
 * Copyright (c) 2001 Alexander Peslyak and it is hereby released to the
 * general public under the following terms:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * See md5.c for more information.
 */
#ifndef _MD5_H_WRAPPER
#define _MD5_H_WRAPPER

#if HAVE_OPENSSL
#include <openssl/md5.h>
#elif !defined(_MD5_H)
#define _MD5_H
/* Any 32-bit or wider unsigned integer data type will do */
typedef unsigned int MD5_u32plus;

#define     MD5_CBLOCK   64
#define     MD5_LBLOCK   (MD5_CBLOCK/4)
#define     MD5_DIGEST_LENGTH   16

typedef struct {
	MD5_u32plus lo, hi;
	MD5_u32plus a, b, c, d;
	unsigned char buffer[MD5_CBLOCK];
	MD5_u32plus block[MD5_DIGEST_LENGTH];
} MD5_CTX;

extern void MD5_Init(MD5_CTX *ctx);
extern void MD5_Update(MD5_CTX *ctx, const void *data, unsigned long size);
extern void MD5_Final(unsigned char *result, MD5_CTX *ctx);

// openssl provide a function MD5(), we just wrap it out if the library is not used
extern unsigned char * MD5( const unsigned char *pointer, unsigned long pointerLength, unsigned char *md5HashPointer);

#endif

#endif
