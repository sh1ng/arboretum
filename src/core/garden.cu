#define CUB_STDERR

#include <math.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <algorithm>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
#include <limits>
#include <random>
#include "builder.h"
#include "cuda_helpers.h"
#include "cuda_runtime.h"
#include "garden.h"
#include "objective.h"
#include "param.h"

namespace arboretum {
namespace core {
using namespace thrust;
using namespace thrust::cuda;
using thrust::device_vector;
using thrust::host_vector;
using thrust::cuda::experimental::pinned_allocator;

template <typename NODE_T, typename GRAD_T, typename SUM_T>
class ContinuousGardenBuilder : public GardenBuilderBase {
 public:
  ContinuousGardenBuilder(const TreeParam &param, const io::DataMatrix *data,
                          const InternalConfiguration &config,
                          const ApproximatedObjective<GRAD_T> *objective,
                          const bool verbose)
      : verbose(verbose),
        rnd(config.seed),
        overlap_depth(config.overlap),
        param(param),
        gain_param(param.min_leaf_size, param.min_child_weight,
                   param.gamma_absolute, param.gamma_relative, param.lambda,
                   param.alpha),
        objective(objective) {
    grad_d.resize(data->rows);

    active_fids.resize(data->columns);

    const unsigned lenght = 1 << param.depth;

    compute1DInvokeConfig(data->rows, &gridSizeGain, &blockSizeGain,
                          gain_kernel<SUM_T, NODE_T>);

    compute1DInvokeConfig(data->rows, &gridSizeGather, &blockSizeGather,
                          gather_kernel<NODE_T, float>);

    row2Node.resize(data->rows);
    _rowIndex2Node.resize(data->rows, 0);
    _bestSplit.resize(1 << (param.depth - 2));
    _nodeStat.resize(1 << (param.depth - 2));

    parent_node_sum.resize(lenght + 1);
    parent_node_count.resize(lenght + 1);
    parent_node_sum_h.resize(lenght + 1);
    parent_node_count_h.resize(lenght + 1);
    growers = new arboretum::core::ContinuousTreeGrower<NODE_T, GRAD_T, SUM_T>
        *[overlap_depth];

    for (size_t i = 0; i < overlap_depth; ++i) {
      growers[i] =
          new arboretum::core::ContinuousTreeGrower<NODE_T, GRAD_T, SUM_T>(
              data->rows, lenght);
      OK(cudaMallocHost(&(best_split_h[i]), sizeof(NODE_T) * 2));
    }
  }

  virtual ~ContinuousGardenBuilder() {
    for (auto i = 0; i < overlap_depth; ++i) {
      delete growers[i];
      OK(cudaFreeHost(best_split_h[i]));
    }

    delete[] best_split_h;
    delete[] growers;
  }

  virtual void InitGrowingTree(const size_t columns) override {
    int take = (int)(param.colsample_bytree * columns);
    if (take == 0) {
      printf("colsample_bytree is too small %f for %ld columns \n",
             param.colsample_bytree, columns);
      throw "colsample_bytree is too small";
    }
    take = (int)(param.colsample_bytree * param.colsample_bylevel * columns);
    if (take == 0) {
      printf(
          "colsample_bytree and colsample_bylevel are too small %f %f for "
          "%ld columns \n",
          param.colsample_bytree, param.colsample_bylevel, columns);
      throw "colsample_bytree and colsample_bylevel are too small";
    }

    for (size_t i = 0; i < columns; ++i) {
      active_fids[i] = i;
    }

    shuffle(active_fids.begin(), active_fids.end(), rnd);

    std::fill(_rowIndex2Node.begin(), _rowIndex2Node.end(), 0);
    for (size_t i = 0; i < _nodeStat.size(); ++i) {
      _nodeStat[i].Clean();
    }
    for (size_t i = 0; i < _bestSplit.size(); ++i) {
      _bestSplit[i].Clean();
    }
  }

  virtual void InitTreeLevel(const int level, const size_t columns) override {
    int take = (int)(param.colsample_bytree * columns);
    shuffle(active_fids.begin(), active_fids.begin() + take, rnd);
  }

  virtual void GrowTree(RegTree *tree, const io::DataMatrix *data,
                        const unsigned short label) override {
    OK(cudaMemcpyAsync(
        thrust::raw_pointer_cast(grad_d.data()),
        thrust::raw_pointer_cast(objective->grad.data() + label * data->rows),
        data->rows * sizeof(GRAD_T), cudaMemcpyHostToDevice,
        growers[0]->stream));

    grad_slice = const_cast<GRAD_T *>(
        thrust::raw_pointer_cast(objective->grad.data() + label * data->rows));

    InitGrowingTree(data->columns);

    for (unsigned int i = 0; (i + 1) < param.depth; ++i) {
      InitTreeLevel(i, data->columns);
      UpdateNodeStat(i, data, tree);
      FindBestSplits(i, data);
      UpdateTree(i, tree);
      UpdateNodeIndex(i, data, tree);
    }

    UpdateLeafWeight(tree);
  }

  virtual void PredictByGrownTree(RegTree *tree, io::DataMatrix *data,
                                  std::vector<float> &out) const override {
    tree->Predict(data, _rowIndex2Node, out);
  }

 private:
  bool verbose;
  std::default_random_engine rnd;
  std::vector<unsigned int> active_fids;
  const unsigned short overlap_depth;
  const TreeParam param;
  const GainFunctionParameters gain_param;
  GRAD_T *grad_slice;
  const ApproximatedObjective<GRAD_T> *objective;
  host_vector<NODE_T, thrust::cuda::experimental::pinned_allocator<NODE_T>>
      _rowIndex2Node;
  std::vector<NodeStat<SUM_T>> _nodeStat;
  std::vector<Split<SUM_T>> _bestSplit;

  device_vector<GRAD_T> grad_d;
  device_vector<NODE_T> row2Node;
  device_vector<SUM_T> parent_node_sum;
  device_vector<unsigned int> parent_node_count;
  host_vector<SUM_T> parent_node_sum_h;
  host_vector<unsigned int> parent_node_count_h;
  size_t temp_bytes_per_rec = 0;

  NODE_T **best_split_h = new NODE_T *[overlap_depth];

  ContinuousTreeGrower<NODE_T, GRAD_T, SUM_T> **growers;

  int blockSizeGain;
  int gridSizeGain;

  int blockSizeGather;
  int gridSizeGather;

  void FindBestSplits(const unsigned int level, const io::DataMatrix *data) {
    OK(cudaMemcpyAsync(thrust::raw_pointer_cast((row2Node.data())),
                       thrust::raw_pointer_cast(_rowIndex2Node.data()),
                       data->rows * sizeof(NODE_T), cudaMemcpyHostToDevice,
                       growers[0]->stream));

    size_t lenght = 1 << level;

    {
      init(parent_node_sum_h[0]);
      parent_node_count_h[0] = 0;

      for (size_t i = 0; i < lenght; ++i) {
        parent_node_count_h[i + 1] =
            parent_node_count_h[i] + _nodeStat[i].count;
        parent_node_sum_h[i + 1] = parent_node_sum_h[i] + _nodeStat[i].sum_grad;
      }
      parent_node_sum = parent_node_sum_h;
      parent_node_count = parent_node_count_h;
    }

    unsigned int take = (unsigned int)(param.colsample_bylevel *
                                       param.colsample_bytree * data->columns);

    OK(cudaStreamSynchronize(growers[0]->stream));

    for (size_t j = 0; j < take; ++j) {
      for (size_t i = 0; i < overlap_depth && (j + i) < take; ++i) {
        if (j != 0 && (i + 1) < overlap_depth) {
          continue;
        }

        size_t active_fid = active_fids[j + i];
        size_t circular_fid = (j + i) % overlap_depth;

        if (active_fid < data->columns_dense) {
          ProcessDenseFeature(active_fid, circular_fid, level, data);
        } else {
          ProcessCategoryFeature(active_fid - data->columns_dense, circular_fid,
                                 level, data);
        }
      }

      size_t circular_fid = j % overlap_depth;

      cudaStream_t s = growers[circular_fid]->stream;

      OK(cudaStreamSynchronize(s));

      if (active_fids[j] < data->columns_dense) {
        if ((data->reduced_size[active_fids[j]] + level) <
            sizeof(unsigned char) * CHAR_BIT) {
          GetBestSplitForDenseFeature<unsigned char>(
              active_fids[j], circular_fid, lenght, data);
        } else if ((data->reduced_size[active_fids[j]] + level) <
                   sizeof(unsigned short) * CHAR_BIT) {
          GetBestSplitForDenseFeature<unsigned short>(
              active_fids[j], circular_fid, lenght, data);
        } else if ((data->reduced_size[active_fids[j]] + level) <
                   sizeof(unsigned int) * CHAR_BIT) {
          GetBestSplitForDenseFeature<unsigned int>(active_fids[j],
                                                    circular_fid, lenght, data);
        } else {
          GetBestSplitForDenseFeature<NODE_T>(active_fids[j], circular_fid,
                                              lenght, data);
        }
      } else {
        if ((data->category_size[active_fids[j] - data->columns_dense] +
             level) < sizeof(unsigned char) * CHAR_BIT) {
          GetBestSplitForCategoryFeature<unsigned char>(
              active_fids[j] - data->columns_dense, data->columns_dense,
              circular_fid, lenght, data);
        } else if ((data->category_size[active_fids[j] - data->columns_dense] +
                    level) < sizeof(unsigned short) * CHAR_BIT) {
          GetBestSplitForCategoryFeature<unsigned short>(
              active_fids[j] - data->columns_dense, data->columns_dense,
              circular_fid, lenght, data);
        } else if ((data->category_size[active_fids[j] - data->columns_dense] +
                    level) < sizeof(unsigned int) * CHAR_BIT) {
          GetBestSplitForCategoryFeature<unsigned int>(
              active_fids[j] - data->columns_dense, data->columns_dense,
              circular_fid, lenght, data);
        } else {
          GetBestSplitForCategoryFeature<NODE_T>(
              active_fids[j] - data->columns_dense, data->columns_dense,
              circular_fid, lenght, data);
        }
      }
    }

    for (size_t i = 0; i < lenght; ++i) {
      Split<SUM_T> &split = _bestSplit[i];

      if (split.fid < 0) {
        NodeStat<SUM_T> &node_stat = _nodeStat[i];
        _bestSplit[i].gain = 0.0;
        _bestSplit[i].fid = 0;
        _bestSplit[i].split_value = std::numeric_limits<float>::infinity();
        _bestSplit[i].count = node_stat.count;
        _bestSplit[i].sum_grad = node_stat.sum_grad;
      }
    }
  }

  template <typename NODE_VALUE_T>
  inline void GetBestSplitForDenseFeature(const int active_fid,
                                          const size_t circular_fid,
                                          const size_t lenght,
                                          const io::DataMatrix *data) {
    for (size_t i = 0; i < lenght; ++i) {
      if (_nodeStat[i].count <= 0) continue;
      if (growers[circular_fid]->result_h[i].floats[0] > _bestSplit[i].gain) {
        const int index_value = growers[circular_fid]->result_h[i].ints[1];
        const SUM_T s = growers[circular_fid]->sum[index_value];
        if (!_isnan(s)) {
          OK(cudaMemcpyAsync(
              (NODE_VALUE_T *)best_split_h[circular_fid],
              (NODE_VALUE_T *)thrust::raw_pointer_cast(
                  growers[circular_fid]->node_fvalue_sorted.data()) +
                  index_value - 1,
              sizeof(NODE_VALUE_T), cudaMemcpyDeviceToHost,
              growers[circular_fid]->stream));

          OK(cudaMemcpyAsync(
              ((NODE_VALUE_T *)best_split_h[circular_fid]) + 1,
              (NODE_VALUE_T *)thrust::raw_pointer_cast(
                  growers[circular_fid]->node_fvalue_sorted.data()) +
                  index_value,
              sizeof(NODE_VALUE_T), cudaMemcpyDeviceToHost,
              growers[circular_fid]->stream));

          OK(cudaStreamSynchronize(growers[circular_fid]->stream));

          const NODE_VALUE_T segment_fvalue_prev_val =
              ((NODE_VALUE_T *)best_split_h[circular_fid])[0];
          const NODE_VALUE_T segment_fvalue_val =
              ((NODE_VALUE_T *)best_split_h[circular_fid])[1];

          const NODE_VALUE_T mask = (1 << (data->reduced_size[active_fid])) - 1;

          const NODE_VALUE_T fvalue_prev_val = segment_fvalue_prev_val & mask;
          const NODE_VALUE_T fvalue_val = segment_fvalue_val & mask;

          const size_t count_val = growers[circular_fid]->result_h[i].ints[1] -
                                   parent_node_count_h[i];

          const SUM_T sum_val = s - parent_node_sum_h[i];
          _bestSplit[i].fid = active_fid;
          _bestSplit[i].gain = growers[circular_fid]->result_h[i].floats[0];
          _bestSplit[i].split_value =
              (data->data_reduced_mapping[active_fid][fvalue_prev_val] +
               data->data_reduced_mapping[active_fid][fvalue_val]) *
              0.5;
          _bestSplit[i].count = count_val;
          _bestSplit[i].sum_grad = sum_val;
          _bestSplit[i].category = (unsigned int)-1;
        } else {
          if (verbose)
            printf(
                "sum is nan(probably infinity), consider increasing the "
                "accuracy \n");
        }
      }
    }
  }

  template <typename NODE_VALUE_T>
  inline void GetBestSplitForCategoryFeature(const int active_fid,
                                             const size_t columns_dense,
                                             const size_t circular_fid,
                                             const size_t lenght,
                                             const io::DataMatrix *data) {
    for (size_t i = 0; i < lenght; ++i) {
      if (_nodeStat[i].count <= 0) continue;
      if (growers[circular_fid]->result_h[i].floats[0] > _bestSplit[i].gain) {
        const int index_value = growers[circular_fid]->result_h[i].ints[1];
        const SUM_T sum_val = growers[circular_fid]->sum[index_value];
        if (!_isnan(sum_val)) {
          const unsigned int count_val =
              growers[circular_fid]->fvalue[index_value];
          if (count_val == 0) continue;

          OK(cudaMemcpyAsync((NODE_VALUE_T *)best_split_h[circular_fid],
                             (NODE_VALUE_T *)thrust::raw_pointer_cast(
                                 growers[circular_fid]->node_fvalue.data()) +
                                 index_value,
                             sizeof(NODE_VALUE_T), cudaMemcpyDeviceToHost,
                             growers[circular_fid]->stream));

          OK(cudaStreamSynchronize(growers[circular_fid]->stream));

          const NODE_VALUE_T segment_fvalues_val =
              ((NODE_VALUE_T *)best_split_h[circular_fid])[0];

          const NODE_VALUE_T mask =
              (1 << (data->category_size[active_fid])) - 1;

          const NODE_VALUE_T category_val = segment_fvalues_val & mask;

          _bestSplit[i].fid = active_fid + columns_dense;
          _bestSplit[i].gain = growers[circular_fid]->result_h[i].floats[0];
          _bestSplit[i].category = category_val;
          _bestSplit[i].count = count_val;
          _bestSplit[i].sum_grad = sum_val;
          _bestSplit[i].split_value = std::numeric_limits<float>::infinity();
        } else {
          if (verbose)
            printf(
                "sum is nan(probably infinity), consider increasing the "
                "accuracy \n");
        }
      }
    }
  }

  void ProcessDenseFeature(const size_t active_fid, const size_t circular_fid,
                           const size_t level, const io::DataMatrix *data) {
    if ((data->reduced_size[active_fid] + level) <
        sizeof(unsigned char) * CHAR_BIT) {
      growers[circular_fid]->template ProcessDenseFeature<unsigned char>(
          row2Node, grad_slice, data->sorted_data_device[active_fid],
          data->data_reduced[active_fid], parent_node_sum, parent_node_count,
          data->reduced_size[active_fid], level, gain_param);
    } else if ((data->reduced_size[active_fid] + level) <
               sizeof(unsigned short) * CHAR_BIT) {
      growers[circular_fid]->template ProcessDenseFeature<unsigned short>(
          row2Node, grad_slice, data->sorted_data_device[active_fid],
          data->data_reduced[active_fid], parent_node_sum, parent_node_count,
          data->reduced_size[active_fid], level, gain_param);
    } else if ((data->reduced_size[active_fid] + level) <
               sizeof(unsigned int) * CHAR_BIT) {
      growers[circular_fid]->template ProcessDenseFeature<unsigned int>(
          row2Node, grad_slice, data->sorted_data_device[active_fid],
          data->data_reduced[active_fid], parent_node_sum, parent_node_count,
          data->reduced_size[active_fid], level, gain_param);
    } else {
      growers[circular_fid]->template ProcessDenseFeature<NODE_T>(
          row2Node, grad_slice, data->sorted_data_device[active_fid],
          data->data_reduced[active_fid], parent_node_sum, parent_node_count,
          data->reduced_size[active_fid], level, gain_param);
    }
  }

  inline void ProcessCategoryFeature(const size_t active_fid,
                                     const size_t circular_fid,
                                     const size_t level,
                                     const io::DataMatrix *data) {
    if ((data->category_size[active_fid] + level) <
        sizeof(unsigned char) * CHAR_BIT) {
      growers[circular_fid]->template ProcessCategoryFeature<unsigned char>(
          row2Node, grad_slice, data->data_category_device[active_fid],
          data->data_categories[active_fid], parent_node_sum, parent_node_count,
          data->category_size[active_fid], level, gain_param);
    } else if ((data->category_size[active_fid] + level) <
               sizeof(unsigned short) * CHAR_BIT) {
      growers[circular_fid]->template ProcessCategoryFeature<unsigned short>(
          row2Node, grad_slice, data->data_category_device[active_fid],
          data->data_categories[active_fid], parent_node_sum, parent_node_count,
          data->category_size[active_fid], level, gain_param);
    } else if ((data->category_size[active_fid] + level) <
               sizeof(unsigned int) * CHAR_BIT) {
      growers[circular_fid]->template ProcessCategoryFeature<unsigned int>(
          row2Node, grad_slice, data->data_category_device[active_fid],
          data->data_categories[active_fid], parent_node_sum, parent_node_count,
          data->category_size[active_fid], level, gain_param);
    } else {
      growers[circular_fid]->template ProcessCategoryFeature<NODE_T>(
          row2Node, grad_slice, data->data_category_device[active_fid],
          data->data_categories[active_fid], parent_node_sum, parent_node_count,
          data->category_size[active_fid], level, gain_param);
    }
  }

  void UpdateNodeStat(const int level, const io::DataMatrix *data,
                      const RegTree *tree) {
    if (level != 0) {
      const unsigned int offset = Node::HeapOffset(level);
      const unsigned int offset_next = Node::HeapOffset(level + 1);
      std::vector<NodeStat<SUM_T>> tmp(_nodeStat.size());
      std::copy(_nodeStat.begin(), _nodeStat.end(), tmp.begin());

      size_t len = 1 << (level - 1);
      for (size_t i = 0; i < len; ++i) {
        _nodeStat[tree->ChildNode(i + offset, true) - offset_next].count =
            _bestSplit[i].count;
        _nodeStat[tree->ChildNode(i + offset, true) - offset_next].sum_grad =
            _bestSplit[i].sum_grad;

        _nodeStat[tree->ChildNode(i + offset, false) - offset_next].count =
            tmp[i].count - _bestSplit[i].count;

        _nodeStat[tree->ChildNode(i + offset, false) - offset_next].sum_grad =
            tmp[i].sum_grad - _bestSplit[i].sum_grad;
      }
    } else {
      _nodeStat[0].count = data->rows;

      SUM_T sum;
      init(sum);

      CubDebugExit(cub::DeviceReduce::Sum(
          this->growers[0]->temp_bytes, this->growers[0]->temp_bytes_allocated,
          thrust::raw_pointer_cast(grad_d.data()),
          thrust::raw_pointer_cast(this->growers[0]->sum.data()), data->rows));

      OK(cudaMemcpy(&sum,
                    thrust::raw_pointer_cast(this->growers[0]->sum.data()),
                    sizeof(SUM_T), cudaMemcpyDeviceToHost));

      CubDebugExit(cudaDeviceSynchronize());

      _nodeStat[0].sum_grad = sum;
    }

    size_t len = 1 << level;

    for (size_t i = 0; i < len; ++i) {
      _nodeStat[i].gain =
          0.0;  // todo: gain_func(_nodeStat[i].count, _nodeStat[i].sum_grad);
      _bestSplit[i].Clean();
    }
  }

  void UpdateTree(const int level, RegTree *tree) const {
    unsigned int offset = Node::HeapOffset(level);

    const size_t len = 1 << level;

    for (size_t i = 0; i < len; ++i) {
      const Split<SUM_T> &best = _bestSplit[i];
      tree->nodes[i + offset].threshold = best.split_value;
      tree->nodes[i + offset].category = best.category;
      tree->nodes[i + offset].fid = best.fid < 0 ? 0 : best.fid;
    }
  }

  void UpdateNodeIndex(const unsigned int level, const io::DataMatrix *data,
                       RegTree *tree) {
    unsigned int const offset = Node::HeapOffset(level);
    unsigned int const offset_next = Node::HeapOffset(level + 1);

#pragma omp parallel for simd
    for (size_t i = 0; i < data->rows; ++i) {
      const unsigned int node = _rowIndex2Node[i];
      const auto &best = _bestSplit[node];
      const bool isLeft =
          (best.fid < (int)data->columns_dense &&
           data->data[best.fid][i] <= best.split_value) ||
          (best.fid >= (int)data->columns_dense &&
           data->data_categories[best.fid - data->columns_dense][i] ==
               best.category);
      _rowIndex2Node[i] = tree->ChildNode(node + offset, isLeft) - offset_next;
    }
  }

  void UpdateLeafWeight(RegTree *tree) const {
    const unsigned int offset_1 = Node::HeapOffset(tree->depth - 2);
    const unsigned int offset = Node::HeapOffset(tree->depth - 1);
    for (unsigned int i = 0, len = (1 << (tree->depth - 2)); i < len; ++i) {
      const Split<SUM_T> &best = _bestSplit[i];
      const NodeStat<SUM_T> &stat = _nodeStat[i];
      tree->leaf_level[tree->ChildNode(i + offset_1, true) - offset] =
          best.LeafWeight(param) * param.eta;
      tree->leaf_level[tree->ChildNode(i + offset_1, false) - offset] =
          best.LeafWeight(stat, param) * param.eta;
    }
  }
};

Garden::Garden(const TreeParam &param, const Verbose &verbose,
               const InternalConfiguration &cfg)
    : param(param), verbose(verbose), cfg(cfg), _init(false) {}

void Garden::GrowTree(io::DataMatrix *data, float *grad) {
  data->Init(verbose.data);

  if (!_init) {
    switch (param.objective) {
      case LinearRegression: {
        auto obj = new RegressionObjective(data, param.initial_y);

        if (data->max_feature_size + param.depth + 1 <=
            sizeof(unsigned char) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned char, float, double>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder = new ContinuousGardenBuilder<unsigned char, float, float>(
                param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned short) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned short, float, double>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned short, float, float>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned int) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder = new ContinuousGardenBuilder<unsigned int, float, double>(
                param, data, cfg, obj, verbose.booster);
          } else {
            _builder = new ContinuousGardenBuilder<unsigned int, float, float>(
                param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned long) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned long, float, double>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder = new ContinuousGardenBuilder<unsigned long, float, float>(
                param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned long long) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned long long, float, double>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned long long, float, float>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else {
          throw "unsupported depth";
        }
        _objective = obj;
      }

      break;
      case LogisticRegression: {
        auto obj = new LogisticRegressionObjective(data, param.initial_y);

        if (data->max_feature_size + param.depth + 1 <=
            sizeof(unsigned char) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned char, float2, mydouble2>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned char, float2, float2>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned short) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned short, float2, mydouble2>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned short, float2, float2>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned int) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned int, float2, mydouble2>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned int, float2, float2>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned long) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned long, float2, mydouble2>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned long, float2, float2>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned long long) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned long, float2, mydouble2>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned long long, float2, float2>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else {
          throw "unsupported depth";
        }
        _objective = obj;
      } break;
      case SoftMaxOneVsAll: {
        auto obj =
            new SoftMaxObjective(data, param.labels_count, param.initial_y);

        if (data->max_feature_size + param.depth + 1 <=
            sizeof(unsigned char) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned char, float2, mydouble2>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned char, float2, float2>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned short) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned short, float2, mydouble2>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned short, float2, float2>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned int) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned int, float2, mydouble2>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned int, float2, float2>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned long) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder =
                new ContinuousGardenBuilder<unsigned long, float2, mydouble2>(
                    param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned long, float2, float2>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else if (data->max_feature_size + param.depth + 1 <=
                   sizeof(unsigned long long) * CHAR_BIT) {
          if (cfg.double_precision) {
            _builder = new ContinuousGardenBuilder<unsigned long long, float2,
                                                   mydouble2>(
                param, data, cfg, obj, verbose.booster);
          } else {
            _builder =
                new ContinuousGardenBuilder<unsigned long long, float2, float2>(
                    param, data, cfg, obj, verbose.booster);
          }
        } else {
          throw "unsupported depth";
        }
        _objective = obj;
      } break;
      default:
        throw "Unknown objective function " + param.objective;
    }

    // auto mem_per_rec = _builder->MemoryRequirementsPerRecord();
    size_t total;
    size_t free;

    cudaMemGetInfo(&free, &total);

    if (verbose.gpu) {
      printf("Total bytes %ld avaliable %ld \n", total, free);
      //   printf("Memory usage estimation %ld per record %ld in total \n",
      //          mem_per_rec, mem_per_rec * data->rows);
    }

    data->TransferToGPU(free * 9 / 10, verbose.gpu);

    _init = true;
  }

  if (grad == NULL) {
    _objective->UpdateGrad();
  } else {
    //          todo: fix
    //          data->grad = std::vector<float>(grad, grad + data->rows);
  }

  for (unsigned short i = 0; i < param.labels_count; ++i) {
    RegTree *tree = new RegTree(param.depth, i);
    _builder->GrowTree(tree, data, i);
    _trees.push_back(tree);
    if (grad == NULL) {
      _builder->PredictByGrownTree(tree, data, data->y_internal);
    }
  }
}

void Garden::UpdateByLastTree(io::DataMatrix *data) {
  if (data->y_internal.size() == 0)
    data->y_internal.resize(data->rows * param.labels_count,
                            _objective->IntoInternal(param.initial_y));
  for (auto it = _trees.end() - param.labels_count; it != _trees.end(); ++it) {
    (*it)->Predict(data, data->y_internal);
  }
}

void Garden::GetY(arboretum::io::DataMatrix *data,
                  std::vector<float> &out) const {
  out.resize(data->y_internal.size());
  _objective->FromInternal(data->y_internal, out);
}

void Garden::Predict(const arboretum::io::DataMatrix *data,
                     std::vector<float> &out) const {
  out.resize(data->rows * param.labels_count);
  std::vector<float> tmp(data->rows * param.labels_count);

  std::fill(tmp.begin(), tmp.end(), _objective->IntoInternal(param.initial_y));
  for (size_t i = 0; i < _trees.size(); ++i) {
    _trees[i]->Predict(data, tmp);
  }

  _objective->FromInternal(tmp, out);
}
}  // namespace core
}  // namespace arboretum
