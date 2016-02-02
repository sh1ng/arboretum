#ifndef ARBORETUM_WRAPPER
#define ARBORETUM_WRAPPER

typedef void* VoidPointer;


extern "C" const char* ACreateFromDanseMatrix(const float *data,
                                   int nrow,
                                   int ncol,
                                   float  missing,
                                   VoidPointer *out);

extern "C" const char* ASetY(VoidPointer data,
                      const float *y);


extern "C" const char* AInitGarden(int obj,
                                   int depth,
                                   int min_child_weight,
                                   float colsample_bytree,
                                   float eta,
                                   VoidPointer *garden);

extern "C" const char* AGrowTree(VoidPointer garden,
                                 VoidPointer data,
                                 const float *grad);

extern "C" const char* APredict(VoidPointer garden,
                     VoidPointer data,
                     const float **out);

extern "C" const char* AFreeDMatrix(VoidPointer ptr);

extern "C" const char* AFreeGarden(VoidPointer ptr);

#endif // ARBORETUM_WRAPPER

