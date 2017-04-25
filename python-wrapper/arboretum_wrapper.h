#ifndef ARBORETUM_WRAPPER
#define ARBORETUM_WRAPPER

typedef void *VoidPointer;

extern "C" const char *ACreateFromDanseMatrix(const float *data,
                                              const unsigned int *categories,
                                              int nrow, int ncol, int ccol,
                                              float missing, VoidPointer *out);

extern "C" const char *ASetY(VoidPointer data, const float *y);

extern "C" const char *ASetLabel(VoidPointer data, const unsigned char *labels);

extern "C" const char *AInitGarden(const char *configuration,
                                   VoidPointer *garden);

extern "C" const char *AGrowTree(VoidPointer garden, VoidPointer data,
                                 const float *grad);

extern "C" const char *AAppendLastTree(VoidPointer garden, VoidPointer data);

extern "C" const char *AGetY(VoidPointer garden, VoidPointer data,
                             const float **out);

extern "C" const char *APredict(VoidPointer garden, VoidPointer data,
                                const float **out);

extern "C" const char *AFreeDMatrix(VoidPointer ptr);

extern "C" const char *AFreeGarden(VoidPointer ptr);

extern "C" const char *ADeleteArray(float *in);

#endif // ARBORETUM_WRAPPER
