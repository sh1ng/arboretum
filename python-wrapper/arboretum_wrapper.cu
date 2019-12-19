#include <memory>
#include "../src/core/garden.h"
#include "../src/core/param.h"
#include "../src/io/io.h"
#include "arboretum_wrapper.h"
#include "stdio.h"

using namespace std;
using namespace arboretum;
using namespace arboretum::io;
using namespace arboretum::core;

namespace arboretum {
namespace wrapper {

extern "C" const char *ACreateFromDenseMatrix(const float *data,
                                              const unsigned int *categories,
                                              int nrow, int ncol, int ccol,
                                              float missing, VoidPointer *out) {
  try {
    DataMatrix *mat = new DataMatrix(nrow, ncol, ccol);
    const size_t size = ncol * nrow;
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
      size_t fidx = i % ncol;
      size_t offset = i / ncol;
      mat->data[fidx][offset] = data[i];
    }

    const size_t size_cat = ccol * nrow;
    for (size_t i = 0; i < size_cat; ++i) {
      mat->data_categories[i % ccol][i / ccol] = categories[i];
    }
    *out = static_cast<VoidPointer>(mat);
    return NULL;
  } catch (const char *error) {
    return error;
  }
}

extern "C" const char *ASetY(VoidPointer data, const float *y) {
  try {
    DataMatrix *data_ptr = static_cast<DataMatrix *>(data);
    data_ptr->y.resize(data_ptr->rows);

#pragma omp parallel for
    for (size_t i = 0; i < data_ptr->rows; ++i) {
      data_ptr->y[i] = y[i];
    }

    return NULL;
  } catch (const char *error) {
    return error;
  }
}

extern "C" const char *ASetLabel(VoidPointer data,
                                 const unsigned char *labels) {
  try {
    DataMatrix *data_ptr = static_cast<DataMatrix *>(data);
    data_ptr->labels.reserve(data_ptr->rows);
#pragma omp parallel for
    for (size_t i = 0; i < data_ptr->rows; ++i) {
      data_ptr->labels[i] = labels[i];
    }

    return NULL;
  } catch (const char *error) {
    return error;
  }
}

extern "C" const char *ASetWeights(VoidPointer data, const float *weights) {
  try {
    DataMatrix *data_ptr = static_cast<DataMatrix *>(data);
    data_ptr->weights.reserve(data_ptr->rows);
#pragma omp parallel for
    for (size_t i = 0; i < data_ptr->rows; ++i) {
      data_ptr->weights[i] = weights[i];
    }

    return NULL;
  } catch (const char *error) {
    return error;
  }
}

extern "C" const char *AInitGarden(const char *configuration,
                                   VoidPointer *out) {
  try {
    const Configuration cfg = Configuration::Parse(configuration);

    Garden *source = new Garden(cfg);
    *out = static_cast<VoidPointer>(source);
    return NULL;
  } catch (const char *error) {
    return error;
  }
}

extern "C" const char *AGrowTree(VoidPointer garden, VoidPointer data,
                                 const float *grad) {
  try {
    io::DataMatrix *data_ptr = static_cast<DataMatrix *>(data);
    Garden *g = static_cast<Garden *>(garden);
    g->GrowTree(data_ptr, (float *)grad);
    return NULL;
  } catch (const char *error) {
    return error;
  }
}

extern "C" const char *APredict(VoidPointer garden, VoidPointer data,
                                const float **out, const int n_rounds) {
  try {
    io::DataMatrix *data_p = static_cast<DataMatrix *>(data);
    Garden *garden_p = static_cast<Garden *>(garden);

    std::vector<float> result;
    garden_p->Predict(data_p, result, n_rounds == -1 ? INT_MAX : n_rounds);

    float *p;
    p = new (nothrow) float[result.size()];
    if (p == nullptr) {
      printf("unable to allocate array \n");
      perror("malloc() failed");
      exit(EXIT_FAILURE);
    }
#pragma omp parallel for
    for (size_t i = 0; i < result.size(); ++i) {
      p[i] = result[i];
    }

    *out = p;
    return NULL;
  } catch (const char *error) {
    return error;
  }
}

extern "C" const char *ADumpModel(const char **model, VoidPointer garden) {
  Garden *garden_p = static_cast<Garden *>(garden);
  *model = garden_p->GetModel();
  return NULL;
}

extern "C" const char *ALoadModel(const char *model, VoidPointer garden) {
  Garden *garden_p = static_cast<Garden *>(garden);
  garden_p->Restore(model);
  return NULL;
}

extern "C" const char *AGetY(VoidPointer garden, VoidPointer data,
                             const float **out) {
  try {
    io::DataMatrix *data_p = static_cast<DataMatrix *>(data);
    Garden *garden_p = static_cast<Garden *>(garden);

    std::vector<float> result;
    garden_p->GetY(data_p, result);

    float *p;
    p = new (nothrow) float[result.size()];
    if (p == nullptr) {
      printf("unable to allocate array \n");
      perror("malloc() failed");
      exit(EXIT_FAILURE);
    }
#pragma omp parallel for
    for (size_t i = 0; i < result.size(); ++i) {
      p[i] = result[i];
    }

    *out = p;
    return NULL;
  } catch (const char *error) {
    return error;
  }
}

extern "C" const char *AAppendLastTree(VoidPointer garden, VoidPointer data) {
  try {
    io::DataMatrix *data_p = static_cast<DataMatrix *>(data);
    Garden *garden_p = static_cast<Garden *>(garden);

    garden_p->UpdateByLastTree(data_p);

    return NULL;
  } catch (const char *error) {
    return error;
  }
}

extern "C" const char *AFreeDMatrix(VoidPointer ptr) {
  delete static_cast<DataMatrix *>(ptr);
  return NULL;
}
extern "C" const char *AFreeGarden(VoidPointer ptr) {
  delete static_cast<Garden *>(ptr);
  return NULL;
}
extern "C" const char *ADeleteArray(float *in) {
  delete[] in;
  return NULL;
}
}  // namespace wrapper
}  // namespace arboretum
