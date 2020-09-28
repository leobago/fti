.. Fault Tolerance Library documentation Compilation file


API Compatibility Notes
===================================================

This section of the user manual explicit differences in API functions among different versions.
An entry will be added in this section whenever FTI adopts changes that are not API compatible with previous versions.


[FTI v1.4] Data-type handling API
---------------------------------------------------

Starting in FTI v1.4.1, the user is no longer required to manage the following FTI data structures:

- **FTIT_type** 
- **FTIT_complexType**

These were previously employed to define custom data-types, being parameters for the following functions:

- **FTI_InitType**
- **FTI_InitComplexType**
- **FTI_AddSimpleField**
- **FTI_AddComplexField**

The new data-type handling API is opaque to the user, meaning that those structures are managed internally by FTI.
The user is granted indirect access to manipulate the data-types through a type identifier, namely **fti_id_t** (i.e a 32-bit integer).
Most of the changes are masked by a new pre-processor definition in FTI:

.. code-block::

   #define FTIT_type fti_id_t

As such, these changes will not impact the majority of existing applications.
However, applications using the following functions will need to be adjusted:

- **FTI_InitComplexType**
- **FTI_AddSimpleField**
- **FTI_AddComplexField**

The management of complex types has been completely refactored due to the aforementioned changes.
To reinforce this refactor, we renamed these functions respectively to:

- **FTI_InitCompositeType**
- **FTI_AddScalarField**
- **FTI_AddVectorField**

These functions are used to describe complex data-types to the FTI runtime.
In previous versions of FTI, these functions were used as follows:

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

The same code in FTI v1.4.1 API is slightly different:

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

Please note the following differences between both snippets:

- **FTI_InitComplexType** was renamed to **FTI_InitCompositeType**;
- **FTI_AddSimpleField** was renamed to **FTI_AddScalarField**;
- The user code no longer manages **FTIT_type** and **FTIT_complexType** objects;
- **FTI_InitCompositeType** is called before **FTI_AddScalarField** and **FTI_AddVectorField**;
- The arguments to these functions were re-ordered.

To see the new parameters order in more details, please refer to the API reference page.
