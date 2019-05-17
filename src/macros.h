#ifndef __MACROS__
#define __MACROS__

void cleanup(char* pattern, ...);

#define MKDIR(a,b) \
	do{				\
		if (mkdir(a,b) == -1) { 																\
			if (errno != EEXIST) {																					\
				char ErrorString[400];																			\
				sprintf(ErrorString,"FILE %s FUNC %s:%d Cannot create directory: %s",__FILE__,__FUNCTION__,__LINE__,a);	\
				FTI_Print(ErrorString, FTI_EROR); 																	\
				return FTI_NSCS; 																					\
			}																										\
		}																											\
	}while(0)

#define FREAD(bytes, buff, size, number, fd, format, ...) \
		do{																										\
	        bytes = fread(buff, size, number, fd);																\
        	if (ferror(fd)) {																					\
				char ErrorString[400];																			\
				sprintf(ErrorString,"FILE %s FUNC %s:%d Error Reading File Bytes Read : %ld",__FILE__,__FUNCTION__,__LINE__,bytes);	\
				FTI_Print(ErrorString, FTI_EROR); 																\
				cleanup(format,__VA_ARGS__,NULL);																\
	            fclose(fd);																						\
            	return FTI_NSCS;																				\
			}																									\
        }while(0)																								

#define FWRITE(bytes, buff, size, number, fd, format, ...) 														\
		do{																										\
	        bytes = fwrite(buff, size, number, fd);																\
        	if (ferror(fd)) {																					\
				char ErrorString[400];																			\
				sprintf(ErrorString,"FILE %s FUNC %s:%d Error Writing File Bytes Written : %ld",__FILE__,__FUNCTION__,__LINE__,bytes);	\
				FTI_Print(ErrorString, FTI_EROR); 																\
				cleanup(format,__VA_ARGS__,NULL);																\
	            fclose(fd);																						\
            	return FTI_NSCS;																				\
			}																									\
        }while(0)	

#endif //__MACROS__
