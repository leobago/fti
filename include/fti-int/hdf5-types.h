#ifndef HDF5_TYPES_H
#define HDF5_TYPES_H

#include <fti-int/defs.h>

/** @typedef    FTIT_complexType
 *  @brief      Type that consists of other FTI types
 *
 *  This type allows creating complex datatypes.
 */
typedef struct FTIT_complexType FTIT_complexType;

typedef struct FTIT_H5Group FTIT_H5Group;

typedef struct FTIT_H5Group {
    int                 id;                     /**< ID of the group.               */
    char                name[FTI_BUFS];         /**< Name of the group.             */
    int                 childrenNo;             /**< Number of children             */
    int                 childrenID[FTI_BUFS];   /**< IDs of the children groups     */
#ifdef ENABLE_HDF5
    hid_t               h5groupID;              /**< Group hid_t.                   */
#endif
} FTIT_H5Group;

typedef struct FTIT_globalDataset {
    bool                        initialized;    /**< Dataset is initialized         */
    int                         rank;           /**< Rank of dataset                */
    int                         id;             /**< ID of dataset.                 */
    int                         numSubSets;     /**< Number of assigned sub-sets    */
    int*                        varIdx;         /**< FTI_Data index of subset var   */
    FTIT_H5Group*               location;       /**< Dataset location in file.      */
#ifdef ENABLE_HDF5
    hid_t                       hid;            /**< HDF5 id datset.                */
    hid_t                       fileSpace;      /**< HDF5 id dataset filespace      */
    hid_t                       hdf5TypeId;     /**< HDF5 id of assigned FTI type   */
    hsize_t*                    dimension;      /**< num of elements for each dim.  */
#endif
    struct FTIT_globalDataset*  next;           /**< Pointer to next dataset        */
    struct FTIT_type*           type;           /**< corresponding FTI type.        */
    char                        name[FTI_BUFS]; /**< Dataset name.                  */
} FTIT_globalDataset;

typedef struct FTIT_sharedData {
    FTIT_globalDataset* dataset;                /**< Pointer to global dataset.     */
#ifdef ENABLE_HDF5
    hsize_t*            count;                  /**< num of elem in each dim.       */
    hsize_t*            offset;                 /**< coord origin of sub-set.       */
#endif
} FTIT_sharedData;

/** @typedef    FTIT_typeField
 *  @brief      Holds info about field in complex type
 *
 *  This type simplify creating complex datatypes.
 */
typedef struct FTIT_typeField {
    int                 typeID;                 /**< FTI type ID of the field.          */
    int                 offset;                 /**< Offset of the field in structure.  */
    int                 rank;                   /**< Field rank (max. 32)               */
    int                 dimLength[32];          /**< Lenght of each dimention           */
    char                name[FTI_BUFS];         /**< Name of the field                  */
} FTIT_typeField;

/** @typedef    FTIT_complexType
 *  @brief      Type that consists of other FTI types
 *
 *  This type allows creating complex datatypes.
 */
typedef struct FTIT_complexType {
    char                name[FTI_BUFS];         /**< Name of the complex type.          */
    int                 length;                 /**< Number of types in complex type.   */
    FTIT_typeField      field[FTI_BUFS];        /**< Fields of the complex type.        */
} FTIT_complexType;


#endif // HDF5_TYPES_H
