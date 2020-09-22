.. Fault Tolerance Library documentation Compilation file


API Compatibility Notes
===================================================

This section of the user manual explicit differences in API functions among different versions.
An entry will be added in this section whenever FTI adopts changes that are not API compatible with previous versions.


[FTI v1.4] Data-type handling API
---------------------------------------------------

Starting in FTI v1.4.2, the user is no longer required to manage the following FTI data structures:

- **FTIT_type** 
- **FTIT_complexType**

These were previously employed to define custom data-types, being parameters for the following functions:

- **FTI_InitType**
- **FTI_InitComplexType**
- **FTI_AddSimpleField**
- **FTI_AddComplexField**

The new data-type handling API is opaque to the user, meaning that those structures are managed internally by FTI.
The user is granted indirect access to manipulate the data-types through a type identifier, namely **fti_id_t** (i.e a 32-bit integer).
To assist in the transition, the following pre-processor operation is added to FTI:

.. code-block::

   #define FTIT_type fti_id_t

These changes do not impact most existing applications.
However, applications using the following function must be altered:

- **FTI_InitComplexType**
- **FTI_AddSimpleField**
- **FTI_AddComplexField**

Due to the aforementioned changes, these functions had to be restructured.
To reinforce these changes, we renamed these functions to:

- **FTI_InitCompositeType**
- **FTI_AddScalarField**
- **FTI_AddVectorField**

These 3 functions are used to describe user-defined composite data-types.
Up to FTI version 1.4.1, these functions were used as follows:

.. code-block::

   // Actual user structure
   typedef struct Point2D {
       int x,y;
   } Point2D;

   // User-managed FTI data structures
   FTIT_type point_type;
   FTIT_complexType point_format;

   // Declaration of Point2D fields
   FTI_AddSimpleField(&point_format, &FTI_INTG, offsetof(Point2D, x), 0, "x");
   FTI_AddSimpleField(&point_format, &FTI_INTG, offsetof(Point2D, y), 1, "y");

   // Association of data-type to data-type format
   FTI_InitComplexType(&point_type, &point_format, 2, sizeof(Point2D), "Point2D", NULL);

Starting in FTI version 1.4.2, the same objective is achieved with the following snippet:

.. code-block::

   // Actual user structure
   typedef struct Point2D {
       int x,y;
   } Point2D;

   // Identifier to FTI-managed type description
   fti_id_t point_tid;

   // Declaration of an user-defined composite type
   point_tid = FTI_InitCompositeType("Point2D", sizeof(Point2D), NULL);

   // Declaration of Point2D fields
   FTI_AddScalarField(point_tid, "x", FTI_INTG, offsetof(Point2D, x));
   FTI_AddScalarField(point_tid, "y", FTI_INTG, offsetof(Point2D, y));

Please note that the following changes were performed:

- **FTI_InitComplexType** was renamed to **FTI_InitCompositeType**;
- **FTI_AddSimpleField** was renamed to **FTI_AddScalarField**;
- FTI_AddComplexField was renamed to **FTI_AddVectorField**;
- It is no longer necessary to allocate **FTIT_type** and **FTIT_complexType** objects;
- The call to **FTI_InitCompositeType** comes before **FTI_AddScalarField** and **FTI_AddVectorField** function calls;
- The arguments to these functions were re-ordered to be more intuitive.

To see the complete changes, please refer to the API reference page.