#include "interface.h"

int FTI_PrintDcpStats( FTIT_configuration FTI_Conf, FTIT_execution FTI_Exec, FTIT_topology FTI_Topo )
{

    char str[FTI_BUFS];
    unsigned long pureDataSize = (FTI_Conf.dcpFtiff) ? FTI_Exec.FTIFFMeta.pureDataSize : FTI_Exec.dcpInfoPosix.dataSize;
    unsigned long dcpSize = (FTI_Conf.dcpFtiff) ? FTI_Exec.FTIFFMeta.dcpSize : FTI_Exec.dcpInfoPosix.dcpSize;
    long norder_data, norder_dcp;
    char corder_data[3], corder_dcp[3];
    long DCP_TB = (1024L*1024L*1024L*1024L);
    long DCP_GB = (1024L*1024L*1024L);
    long DCP_MB = (1024L*1024L);
    if ( pureDataSize > DCP_TB ) {
        norder_data = DCP_TB;
        snprintf( corder_data, 3, "TB" );
    } else if ( pureDataSize > DCP_GB ) {
        norder_data = DCP_GB;
        snprintf( corder_data, 3, "GB" );
    } else {
        norder_data = DCP_MB;
        snprintf( corder_data, 3, "MB" );
    }
    if ( dcpSize > DCP_TB ) {
        norder_dcp = DCP_TB;
        snprintf( corder_dcp, 3, "TB" );
    } else if ( dcpSize > DCP_GB ) {
        norder_dcp = DCP_GB;
        snprintf( corder_dcp, 3, "GB" );
    } else {
        norder_dcp = DCP_MB;
        snprintf( corder_dcp, 3, "MB" );
    }

    if ( FTI_Topo.splitRank != 0 ) {
        snprintf( str, FTI_BUFS, "Local CP data: %.2lf %s, Local dCP update: %.2lf %s, dCP share: %.2lf%%",
                (double)pureDataSize/norder_data, corder_data,
                (double)dcpSize/norder_dcp, corder_dcp,
                ((double)dcpSize/pureDataSize)*100 );
        FTI_Print( str, FTI_DBUG );
    } else {
        snprintf( str, FTI_BUFS, "Total CP data: %.2lf %s, Total dCP update: %.2lf %s, dCP share: %.2lf%%",
                (double)pureDataSize/norder_data, corder_data,
                (double)dcpSize/norder_dcp, corder_dcp,
                ((double)dcpSize/pureDataSize)*100 );
    }

    FTI_Print(str, FTI_IDCP);
}
