#ifndef IO_H
#define IO_H

#include <vector>
#include <functional>

namespace arboretum {
  namespace io {
    class DataMatrix {
    public:
      std::vector<std::vector<int> > index;
      std::vector<std::vector<float> > data;
      std::vector<float> y_hat;
      std::vector<float> y;
      std::vector<float> grad;
      int rows;
      int columns;
      void Init(const float initial_y, std::function<float const(const float, const float)> func);
      void UpdateGrad();
      DataMatrix(int rows, int columns);
    private:
      std::function<float const(const float, const float)> _gradFunc;
      bool _init;
      std::vector<int> SortedIndex(int column);
    };
  }
}

#endif // IO_H
