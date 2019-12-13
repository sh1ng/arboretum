#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <ctime>
#include <limits>
#include <random>
#include "best_splits.h"
#include "continuous_tree_grower.h"
#include "cub/cub.cuh"
#include "cuda_helpers.h"
#include "garden.h"
#include "hist_tree_grower.h"
#include "histogram.h"
#include "model_helper.h"
#include "objective.h"
#include "param.h"

namespace arboretum {
namespace core {
using namespace thrust;
using namespace thrust::cuda;
using thrust::device_vector;
using thrust::host_vector;

template <typename T>
__global__ void set_segment(T *row2Node, const unsigned *count_prefix_sum,
                            int shift, int level, const size_t n) {
  for (unsigned i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    T leaf = row2Node[i];
    unsigned segment =
      linear_search_2steps(count_prefix_sum, i, unsigned(1 << level) + 1,
                           unsigned(leaf >> (shift + 1)) << 1);
    row2Node[i] = (segment << shift) | (leaf & 1 << (shift - 1));
  }
}

template <typename SUM_T, typename NODE_T>
__global__ void update_by_last_tree(float *y, const SUM_T *best_sum,
                                    const unsigned *best_count,
                                    const SUM_T *sum_prefix_sum,
                                    const unsigned *count_prefix_sum,
                                    const NODE_T *row2Node,
                                    const TreeParam param, const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    NODE_T leaf = row2Node[i];

    unsigned segment = leaf >> 2;

    float delta = 0.0;
    const SUM_T left_sum = best_sum[segment];
    const SUM_T right_sum =
      sum_prefix_sum[segment + 1] - sum_prefix_sum[segment] - left_sum;

    const unsigned left_count = best_count[segment];
    const unsigned right_count =
      count_prefix_sum[segment + 1] - count_prefix_sum[segment] - left_count;

    if (leaf % 2 == 0) {
      delta = Weight(left_sum, left_count, param) * param.eta;
    } else {
      delta = Weight(right_sum, right_count, param) * param.eta;
    }
    assert(isfinite(delta));

    y[i] += delta;
  }
}

template <typename NODE_T, typename GRAD_T, typename SUM_T,
          typename TREE_GROWER>
class ContinuousGardenBuilder : public GardenBuilderBase {
 public:
  ContinuousGardenBuilder(const TreeParam &param, io::DataMatrix *data,
                          const InternalConfiguration &config,
                          ApproximatedObjectiveBase *objective,
                          const bool verbose)
      : verbose(verbose),
        rnd(config.seed),
        overlap_depth(config.overlap),
        param(param),
        gain_param(param.min_leaf_size, param.min_child_weight,
                   param.gamma_absolute, param.gamma_relative, param.lambda,
                   param.alpha),
        objective(static_cast<ApproximatedObjective<GRAD_T> *>(objective)),
        best(1 << param.depth, config.hist_size),
        features_histograms(1 << param.depth, config.hist_size,
                            data->columns_dense) {
    active_fids.resize(data->columns);

    row2Node.resize(data->rows, 0);
    partitioning_index.resize(data->rows, 0);
    _bestSplit.resize(1 << (param.depth - 2));
    _nodeStat.resize(1 << (param.depth - 2));
    grad.resize(data->rows);
    grad_buffer.resize(data->rows);
    data->y_hat.resize(data->rows, objective->IntoInternal(param.initial_y));
    y_hat_d = data->y_hat;
    y_d = data->y;
    y_buffer.resize(data->rows);

    growers = new TREE_GROWER *[overlap_depth];

    for (size_t i = 0; i < overlap_depth; ++i) {
      growers[i] = new TREE_GROWER(data->rows, param.depth, config.hist_size,
                                   &best, &features_histograms, &config);
    }
  }

  virtual ~ContinuousGardenBuilder() {
    for (auto i = 0; i < overlap_depth; ++i) {
      delete growers[i];
    }

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

    thrust::fill(row2Node.begin(), row2Node.end(), 0);
    for (size_t i = 0; i < _nodeStat.size(); ++i) {
      _nodeStat[i].Clean();
    }
    for (size_t i = 0; i < _bestSplit.size(); ++i) {
      _bestSplit[i].Clean();
    }
    this->features_histograms.Clear();
    OK(cudaDeviceSynchronize());
    for (size_t i = 0; i < overlap_depth; i++) {
      OK(cudaStreamSynchronize(growers[i]->stream));
    }
  }

  virtual void InitTreeLevel(const int level, const size_t columns) override {
    int take = (int)(param.colsample_bytree * columns);
    shuffle(active_fids.begin(), active_fids.begin() + take, rnd);
  }

  virtual void GrowTree(RegTree *tree, io::DataMatrix *data,
                        const unsigned short label) override {
    grad_slice = const_cast<GRAD_T *>(
      thrust::raw_pointer_cast(grad.data() + label * data->rows));

    InitGrowingTree(data->columns);

    for (unsigned int i = 0; (i + 1) < param.depth; ++i) {
      InitTreeLevel(i, data->columns);
      UpdateNodeStat(i, data, tree);
      FindBestSplits(i, data);
      UpdateTree(i, tree, data);
    }

    for (size_t i = 0; i < overlap_depth; i++) {
      OK(cudaStreamSynchronize(growers[i]->stream));
    }

    OK(cudaDeviceSynchronize());
    OK(cudaGetLastError());

    UpdateLeafWeight(tree);
    for (size_t i = 0; i < overlap_depth; i++) {
      OK(cudaStreamSynchronize(growers[i]->stream));
    }

    OK(cudaDeviceSynchronize());
    OK(cudaGetLastError());

    UpdateByLastTree(data);
  }

  void UpdateByLastTree(io::DataMatrix *data) {
    int gridSize = 0;
    int blockSize = 0;
    compute1DInvokeConfig(data->rows, &gridSize, &blockSize,
                          update_by_last_tree<SUM_T, NODE_T>);

    update_by_last_tree<SUM_T, NODE_T><<<gridSize, blockSize>>>(
      thrust::raw_pointer_cast(y_hat_d.data()),
      thrust::raw_pointer_cast(this->best.sum.data()),
      thrust::raw_pointer_cast(this->best.count.data()),
      thrust::raw_pointer_cast(this->best.parent_node_sum.data()),
      thrust::raw_pointer_cast(this->best.parent_node_count.data()),
      thrust::raw_pointer_cast(row2Node.data()), param, data->rows);
  }

  virtual void PredictByGrownTree(
    RegTree *tree, io::DataMatrix *data,
    thrust::host_vector<float> &out) const override {
    // tree->Predict(data, _rowIndex2Node, out);
  }

  virtual void UpdateGrad() override {
    objective->UpdateGrad(grad, y_hat_d, y_d);
  }

 private:
  bool verbose;
  std::default_random_engine rnd;
  std::vector<unsigned int> active_fids;
  const unsigned short overlap_depth;
  const TreeParam param;
  const GainFunctionParameters gain_param;
  GRAD_T *grad_slice;
  ApproximatedObjective<GRAD_T> *objective;
  std::vector<NodeStat<SUM_T>> _nodeStat;
  std::vector<Split<SUM_T>> _bestSplit;

  device_vector<NODE_T> row2Node;
  device_vector<unsigned> partitioning_index;
  size_t temp_bytes_per_rec = 0;

  TREE_GROWER **growers;
  BestSplit<SUM_T> best;
  Histogram<SUM_T> features_histograms;
  device_vector<GRAD_T> grad;
  device_vector<GRAD_T> grad_buffer;
  device_vector<float> y_buffer;
  device_vector<float> y_d;
  device_vector<float> y_hat_d;

  void FindBestSplits(const unsigned int level, io::DataMatrix *data) {
    unsigned length = 1 << level;

    unsigned int take = (unsigned int)(param.colsample_bylevel *
                                       param.colsample_bytree * data->columns);

    if (level != 0) {
      this->best.NextLevel(length);

      int gridSize = 0;
      int blockSize = 0;
      compute1DInvokeConfig(data->rows, &gridSize, &blockSize,
                            set_segment<NODE_T>);
      set_segment<NODE_T><<<gridSize, blockSize, 0, growers[0]->stream>>>(
        thrust::raw_pointer_cast(row2Node.data()),
        thrust::raw_pointer_cast(this->best.parent_node_count.data()),
        param.depth - level, level, data->rows);

      growers[0]->CreatePartitioningIndexes(partitioning_index, row2Node,
                                            this->best.parent_node_count, level,
                                            param.depth);

      growers[0]->template PartitionByIndex<GRAD_T>(
        thrust::raw_pointer_cast(grad_buffer.data()),
        thrust::raw_pointer_cast(grad.data()), partitioning_index);

      OK(cudaStreamSynchronize(growers[0]->stream));

      grad.swap(grad_buffer);

      growers[0]->template PartitionByIndex<float>(
        thrust::raw_pointer_cast(y_buffer.data()),
        thrust::raw_pointer_cast(y_hat_d.data()), partitioning_index);

      OK(cudaStreamSynchronize(growers[0]->stream));

      y_buffer.swap(y_hat_d);

      growers[0]->template PartitionByIndex<float>(
        thrust::raw_pointer_cast(y_buffer.data()),
        thrust::raw_pointer_cast(y_d.data()), partitioning_index);

      OK(cudaStreamSynchronize(growers[0]->stream));

      y_buffer.swap(y_d);
    }

    OK(cudaStreamSynchronize(growers[0]->stream));
    OK(cudaStreamSynchronize(growers[0]->copy_d2h_stream));

    for (size_t j = 0; j < data->columns; ++j) {
      for (size_t i = 0; i < overlap_depth && (j + i) < data->columns; ++i) {
        if (j != 0 && (i + 1) < overlap_depth) {
          continue;
        }

        size_t active_fid = active_fids[j + i];
        size_t circular_fid = (j + i) % overlap_depth;

        if (active_fid < data->columns_dense) {
          ProcessDenseFeature(active_fid, circular_fid, level, data,
                              (j + i) >= take);
        } else {
          ProcessCategoryFeature(active_fid - data->columns_dense, circular_fid,
                                 level, data);
        }
      }

      size_t circular_fid = j % overlap_depth;

      if (active_fids[j] < data->columns_dense) {
        // FIXME:
        // if ((data->reduced_size[active_fids[j]] + level) <
        //     sizeof(unsigned char) * CHAR_BIT) {
        //   GetBestSplitForDenseFeature<unsigned char>(
        //       active_fids[j], circular_fid, level,
        //       data->data_reduced_mapping[active_fids[j]],
        //       data->reduced_size[active_fids[j]]);
        // } else if ((data->reduced_size[active_fids[j]] + level) <
        //            sizeof(unsigned short) * CHAR_BIT) {
        //   GetBestSplitForDenseFeature<unsigned short>(
        //       active_fids[j], circular_fid, level,
        //       data->data_reduced_mapping[active_fids[j]],
        //       data->reduced_size[active_fids[j]]);
        // } else if ((data->reduced_size[active_fids[j]] + level) <
        //            sizeof(unsigned int) * CHAR_BIT) {
        GetBestSplitForDenseFeature<unsigned int>(
          active_fids[j], circular_fid, level,
          data->data_reduced_mapping[active_fids[j]],
          data->reduced_size[active_fids[j]], j >= take);
        // } else {
        //   GetBestSplitForDenseFeature<NODE_T>(
        //       active_fids[j], circular_fid, level,
        //       data->data_reduced_mapping[active_fids[j]],
        //       data->reduced_size[active_fids[j]]);
        // }
      } else {
        if ((data->category_size[active_fids[j] - data->columns_dense] +
             level) < sizeof(unsigned char) * CHAR_BIT) {
          GetBestSplitForCategoryFeature<unsigned char>(
            active_fids[j] - data->columns_dense, data->columns_dense,
            circular_fid, length, data);
        } else if ((data->category_size[active_fids[j] - data->columns_dense] +
                    level) < sizeof(unsigned short) * CHAR_BIT) {
          GetBestSplitForCategoryFeature<unsigned short>(
            active_fids[j] - data->columns_dense, data->columns_dense,
            circular_fid, length, data);
        } else if ((data->category_size[active_fids[j] - data->columns_dense] +
                    level) < sizeof(unsigned int) * CHAR_BIT) {
          GetBestSplitForCategoryFeature<unsigned int>(
            active_fids[j] - data->columns_dense, data->columns_dense,
            circular_fid, length, data);
        } else {
          GetBestSplitForCategoryFeature<NODE_T>(
            active_fids[j] - data->columns_dense, data->columns_dense,
            circular_fid, length, data);
        }
      }
    }

    // growers[0]->template Partition<GRAD_T, 1>(
    //   thrust::raw_pointer_cast(grad_d.data()),
    //   thrust::raw_pointer_cast(row2Node.data()), parent_node_count, level,
    //   param.depth);
  }

  // FIXME: use template
  template <typename NODE_VALUE_T>
  inline void GetBestSplitForDenseFeature(
    const int active_fid, const size_t circular_fid, const unsigned level,
    const std::vector<float> &data_reduced_mapping, const unsigned reduced_size,
    const bool partition_only) {
    if (!partition_only) {
      const unsigned length = 1 << level;

      OK(cudaStreamSynchronize(growers[circular_fid]->stream));

      growers[circular_fid]->FindBest(
        this->best, this->row2Node, this->best.parent_node_sum,
        this->best.parent_node_count, active_fid, level, param.depth, length);
    }
    OK(cudaStreamSynchronize(growers[circular_fid]->stream));
    OK(cudaStreamSynchronize(growers[circular_fid]->copy_d2h_stream));
  }

  template <typename NODE_VALUE_T>
  inline void GetBestSplitForCategoryFeature(const int active_fid,
                                             const size_t columns_dense,
                                             const size_t circular_fid,
                                             const size_t lenght,
                                             const io::DataMatrix *data) {}

  void ProcessDenseFeature(const size_t active_fid, const size_t circular_fid,
                           const size_t level, io::DataMatrix *data,
                           const bool partition_only) {
    // if ((data->reduced_size[active_fid]) < sizeof(unsigned char) *
    // CHAR_BIT)
    // {
    //   growers[circular_fid]->template ProcessDenseFeature<unsigned char>(
    //       row2Node, grad_d,
    //       data->sorted_data_device[active_fid].size() > 0
    //           ? thrust::raw_pointer_cast(
    //                 data->sorted_data_device[active_fid].data())
    //           : nullptr,
    //       thrust::raw_pointer_cast(data->data_reduced[active_fid].data()),
    //       parent_node_sum, parent_node_count,
    //       data->reduced_size[active_fid], level, gain_param);
    // } else if ((data->reduced_size[active_fid]) <
    //            sizeof(unsigned short) * CHAR_BIT) {
    //   growers[circular_fid]->template ProcessDenseFeature<unsigned short>(
    //       row2Node, grad_d,
    //       data->sorted_data_device[active_fid].size() > 0
    //           ? thrust::raw_pointer_cast(
    //                 data->sorted_data_device[active_fid].data())
    //           : nullptr,
    //       thrust::raw_pointer_cast(data->data_reduced[active_fid].data()),
    //       parent_node_sum, parent_node_count,
    //       data->reduced_size[active_fid], level, gain_param);
    // } else if ((data->reduced_size[active_fid]) <
    //            sizeof(unsigned int) * CHAR_BIT) {
    //   growers[circular_fid]->template ProcessDenseFeature<unsigned int>(
    //       row2Node, grad_d,
    //       data->sorted_data_device[active_fid].size() > 0
    //           ? thrust::raw_pointer_cast(
    //                 data->sorted_data_device[active_fid].data())
    //           : nullptr,
    //       thrust::raw_pointer_cast(data->data_reduced[active_fid].data()),
    //       parent_node_sum, parent_node_count,
    //       data->reduced_size[active_fid], level, gain_param);
    // } else {
    growers[circular_fid]->template ProcessDenseFeature<NODE_T>(
      partitioning_index, row2Node, grad, data->sorted_data_device[active_fid],
      thrust::raw_pointer_cast(data->data_reduced[active_fid].data()),
      this->best.parent_node_sum, this->best.parent_node_count,
      data->reduced_size[active_fid], level, param.depth, gain_param,
      partition_only, active_fid);
    // }
  }

  inline void ProcessCategoryFeature(const size_t active_fid,
                                     const size_t circular_fid,
                                     const size_t level,
                                     const io::DataMatrix *data) {
    if ((data->category_size[active_fid] + level) <
        sizeof(unsigned char) * CHAR_BIT) {
      growers[circular_fid]->template ProcessCategoryFeature<unsigned char>(
        row2Node, grad, data->data_category_device[active_fid],
        data->data_categories[active_fid], this->best.parent_node_sum,
        this->best.parent_node_count, data->category_size[active_fid], level,
        gain_param);
    } else if ((data->category_size[active_fid] + level) <
               sizeof(unsigned short) * CHAR_BIT) {
      growers[circular_fid]->template ProcessCategoryFeature<unsigned short>(
        row2Node, grad, data->data_category_device[active_fid],
        data->data_categories[active_fid], this->best.parent_node_sum,
        this->best.parent_node_count, data->category_size[active_fid], level,
        gain_param);
    } else if ((data->category_size[active_fid] + level) <
               sizeof(unsigned int) * CHAR_BIT) {
      growers[circular_fid]->template ProcessCategoryFeature<unsigned int>(
        row2Node, grad, data->data_category_device[active_fid],
        data->data_categories[active_fid], this->best.parent_node_sum,
        this->best.parent_node_count, data->category_size[active_fid], level,
        gain_param);
    } else {
      growers[circular_fid]->template ProcessCategoryFeature<NODE_T>(
        row2Node, grad, data->data_category_device[active_fid],
        data->data_categories[active_fid], this->best.parent_node_sum,
        this->best.parent_node_count, data->category_size[active_fid], level,
        gain_param);
    }
  }

  void UpdateNodeStat(const int level, const io::DataMatrix *data,
                      const RegTree *tree) {
    const unsigned len = 1 << level;
    best.Clear(len);

    if (level == 0) {
      SUM_T zero;
      init(zero);

      best.parent_node_count[0] = 0;
      best.parent_node_count[1] = unsigned(data->rows);
      best.parent_node_sum[0] = zero;

      OK(cub::DeviceReduce::Sum(
        this->growers[0]->temp_bytes, this->growers[0]->temp_bytes_allocated,
        thrust::raw_pointer_cast(grad.data()),
        thrust::raw_pointer_cast(&best.parent_node_sum[1]), data->rows));

      CubDebugExit(cudaDeviceSynchronize());
    }

    for (unsigned i = 0; i < len; ++i) {
      _nodeStat[i].gain =
        0.0;  // todo: gain_func(_nodeStat[i].count, _nodeStat[i].sum_grad);
      _bestSplit[i].Clean();
    }
  }

  void UpdateTree(const int level, RegTree *tree, io::DataMatrix *data) {
    const unsigned int offset = Node::HeapOffset(level);
    const size_t len = 1 << level;
    CubDebugExit(cudaDeviceSynchronize());

    best.Sync(1 << level);
    CubDebugExit(cudaDeviceSynchronize());

    for (unsigned i = 0; i < len; ++i) {
      const unsigned quantized = best.split_value_h[i];

      const my_atomics &gain_feature = best.gain_feature_h[i];

      _bestSplit[i].quantized = quantized;
      _bestSplit[i].count = best.count_h[i];
      _bestSplit[i].fid = gain_feature.Feature();
      _bestSplit[i].sum_grad = best.sum_h[i];

      if (gain_feature.Feature() != -1) {
        _bestSplit[i].split_value =
          quantized >= data->data_reduced_mapping[gain_feature.Feature()].size()
            ? std::numeric_limits<float>::infinity()
            : data->data_reduced_mapping[gain_feature.Feature()][quantized];
      } else {
        _bestSplit[i].gain = 0.0;
        _bestSplit[i].fid = 0;
        _bestSplit[i].split_value = std::numeric_limits<float>::infinity();
        _bestSplit[i].count =
          best.parent_node_count_h[i + 1] - best.parent_node_count_h[i];
        _bestSplit[i].sum_grad =
          best.parent_node_sum_h[i + 1] - best.parent_node_sum_h[i];
      }
      const Split<SUM_T> &best = _bestSplit[i];
      tree->nodes[i + offset].threshold = best.split_value;
      tree->nodes[i + offset].category = best.category;
      tree->nodes[i + offset].fid = best.fid < 0 ? 0 : best.fid;
      tree->nodes[i + offset].quantized = best.quantized;
    }
  }

  void UpdateLeafWeight(RegTree *tree) const {
    const unsigned int offset_1 = Node::HeapOffset(tree->depth - 2);
    const unsigned int offset = Node::HeapOffset(tree->depth - 1);

    for (unsigned int i = 0, len = (1 << (tree->depth - 2)); i < len; ++i) {
      const Split<SUM_T> &split = _bestSplit[i];
      tree->weights[tree->ChildNode(i + offset_1, true) - offset] =
        split.LeafWeight(param) * param.eta;
      tree->weights[tree->ChildNode(i + offset_1, false) - offset] =
        split.LeafWeight(
          this->best.parent_node_count_h[i + 1] -
            this->best.parent_node_count_h[i],
          this->best.parent_node_sum_h[i + 1] - this->best.parent_node_sum_h[i],
          param) *
        param.eta;
    }
  }
};

Garden::Garden(const Configuration &cfg) : cfg(cfg), _init(false) {
  switch (cfg.objective) {
    case LinearRegression:
      _objective = new RegressionObjective(cfg.tree_param.initial_y);
      break;
    case LogisticRegression:
      _objective = new LogisticRegressionObjective(cfg.tree_param.initial_y);
      break;
    default:
      throw "Unknown objective function " + cfg.objective;
  }
}

void Garden::GrowTree(io::DataMatrix *data, float *grad) {
  if (cfg.method == Method::Exact)
    data->InitExact(cfg.verbose.data);
  else
    data->InitHist(cfg.internal.hist_size, cfg.verbose.data);

  if (!_init) {
    switch (cfg.objective) {
      case LinearRegression: {
        if (data->max_feature_size + 1 <= sizeof(unsigned char) * CHAR_BIT) {
          if (cfg.internal.double_precision) {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned, float, double,
                ContinuousTreeGrower<unsigned, float, double>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned, float, double,
                HistTreeGrower<unsigned, float, double>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          } else {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned, float, float,
                ContinuousTreeGrower<unsigned, float, float>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned, float, float, HistTreeGrower<unsigned, float, float>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          }
        } else if (data->max_feature_size + 1 <=
                   sizeof(unsigned short) * CHAR_BIT) {
          if (cfg.internal.double_precision) {
            if (cfg.tree_param.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned, float, double,
                ContinuousTreeGrower<unsigned, float, double>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned, float, double,
                HistTreeGrower<unsigned, float, double>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          } else {
            if (cfg.tree_param.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned, float, float,
                ContinuousTreeGrower<unsigned, float, float>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned, float, float, HistTreeGrower<unsigned, float, float>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          }
        } else if (data->max_feature_size + 1 <=
                   sizeof(unsigned int) * CHAR_BIT) {
          if (cfg.internal.double_precision) {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned int, float, double,
                ContinuousTreeGrower<unsigned, float, double>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned int, float, double,
                HistTreeGrower<unsigned, float, double>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);

          } else {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned int, float, float,
                ContinuousTreeGrower<unsigned, float, float>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned int, float, float,
                HistTreeGrower<unsigned, float, float>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          }
        } else if (data->max_feature_size + 1 <=
                   sizeof(unsigned long) * CHAR_BIT) {
          if (cfg.internal.double_precision) {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned long, float, double,
                ContinuousTreeGrower<unsigned long, float, double>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned long, float, double,
                HistTreeGrower<unsigned long, float, double>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          } else {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned long, float, float,
                ContinuousTreeGrower<unsigned long, float, float>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned long, float, float,
                HistTreeGrower<unsigned long, float, float>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          }
        } else if (data->max_feature_size + 1 <=
                   sizeof(unsigned long long) * CHAR_BIT) {
          if (cfg.internal.double_precision) {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned long long, float, double,
                ContinuousTreeGrower<unsigned long long, float, double>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned long long, float, double,
                HistTreeGrower<unsigned long long, float, double>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);

          } else {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned long long, float, float,
                ContinuousTreeGrower<unsigned long long, float, float>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned long long, float, float,
                HistTreeGrower<unsigned long long, float, float>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          }
        } else {
          throw "unsupported dimensionality";
        }
      }

      break;
      case LogisticRegression: {
        if (data->max_feature_size + 1 <= sizeof(unsigned char) * CHAR_BIT) {
          if (cfg.internal.double_precision) {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned, float2, mydouble2,
                ContinuousTreeGrower<unsigned, float2, mydouble2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned, float2, mydouble2,
                HistTreeGrower<unsigned, float2, mydouble2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          } else {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned, float2, float2,
                ContinuousTreeGrower<unsigned, float2, float2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned, float2, float2,
                HistTreeGrower<unsigned, float2, float2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          }
        } else if (data->max_feature_size + 1 <=
                   sizeof(unsigned short) * CHAR_BIT) {
          if (cfg.internal.double_precision) {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned, float2, mydouble2,
                ContinuousTreeGrower<unsigned, float2, mydouble2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned, float2, mydouble2,
                HistTreeGrower<unsigned, float2, mydouble2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          } else {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned, float2, float2,
                ContinuousTreeGrower<unsigned, float2, float2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned, float2, float2,
                HistTreeGrower<unsigned, float2, float2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          }
        } else if (data->max_feature_size + 1 <=
                   sizeof(unsigned int) * CHAR_BIT) {
          if (cfg.internal.double_precision) {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned int, float2, mydouble2,
                ContinuousTreeGrower<unsigned, float2, mydouble2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned int, float2, mydouble2,
                HistTreeGrower<unsigned, float2, mydouble2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          } else {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned int, float2, float2,
                ContinuousTreeGrower<unsigned, float2, float2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned int, float2, float2,
                HistTreeGrower<unsigned, float2, float2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          }
        } else if (data->max_feature_size + 1 <=
                   sizeof(unsigned long) * CHAR_BIT) {
          if (cfg.internal.double_precision) {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned long, float2, mydouble2,
                ContinuousTreeGrower<unsigned long, float2, mydouble2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned long, float2, mydouble2,
                HistTreeGrower<unsigned long, float2, mydouble2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          } else {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned long, float2, float2,
                ContinuousTreeGrower<unsigned long, float2, float2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned long, float2, float2,
                HistTreeGrower<unsigned long, float2, float2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          }
        } else if (data->max_feature_size + 1 <=
                   sizeof(unsigned long long) * CHAR_BIT) {
          if (cfg.internal.double_precision) {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned long, float2, mydouble2,
                ContinuousTreeGrower<unsigned long, float2, mydouble2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned long, float2, mydouble2,
                HistTreeGrower<unsigned long, float2, mydouble2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          } else {
            if (cfg.method == Exact)
              _builder = new ContinuousGardenBuilder<
                unsigned long long, float2, float2,
                ContinuousTreeGrower<unsigned long long, float2, float2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
            else
              _builder = new ContinuousGardenBuilder<
                unsigned long long, float2, float2,
                HistTreeGrower<unsigned long long, float2, float2>>(
                cfg.tree_param, data, cfg.internal, _objective,
                cfg.verbose.booster);
          }
        } else {
          throw "unsupported dimensionality";
        }
      } break;
        //   case SoftMaxOneVsAll: {
        //     auto obj =
        //       new SoftMaxObjective(data, param.labels_count,
        //       param.initial_y);

        //     if (data->max_feature_size + 1 <= sizeof(unsigned char) *
        //     CHAR_BIT) {
        //       if (cfg.double_precision) {
        //         _builder = new ContinuousGardenBuilder<unsigned, float2,
        //         mydouble2>(
        //           param, data, cfg, obj, verbose.booster);
        //       } else {
        //         _builder = new ContinuousGardenBuilder<unsigned, float2,
        //         float2>(
        //           param, data, cfg, obj, verbose.booster);
        //       }
        //     } else if (data->max_feature_size + 1 <=
        //                sizeof(unsigned short) * CHAR_BIT) {
        //       if (cfg.double_precision) {
        //         _builder = new ContinuousGardenBuilder<unsigned, float2,
        //         mydouble2>(
        //           param, data, cfg, obj, verbose.booster);
        //       } else {
        //         _builder = new ContinuousGardenBuilder<unsigned, float2,
        //         float2>(
        //           param, data, cfg, obj, verbose.booster);
        //       }
        //     } else if (data->max_feature_size + 1 <=
        //                sizeof(unsigned int) * CHAR_BIT) {
        //       if (cfg.double_precision) {
        //         _builder =
        //           new ContinuousGardenBuilder<unsigned int, float2,
        //           mydouble2>(
        //             param, data, cfg, obj, verbose.booster);
        //       } else {
        //         _builder =
        //           new ContinuousGardenBuilder<unsigned int, float2, float2>(
        //             param, data, cfg, obj, verbose.booster);
        //       }
        //     } else if (data->max_feature_size + 1 <=
        //                sizeof(unsigned long) * CHAR_BIT) {
        //       if (cfg.double_precision) {
        //         _builder =
        //           new ContinuousGardenBuilder<unsigned long, float2,
        //           mydouble2>(
        //             param, data, cfg, obj, verbose.booster);
        //       } else {
        //         _builder =
        //           new ContinuousGardenBuilder<unsigned long, float2, float2>(
        //             param, data, cfg, obj, verbose.booster);
        //       }
        //     } else if (data->max_feature_size + 1 <=
        //                sizeof(unsigned long long) * CHAR_BIT) {
        //       if (cfg.double_precision) {
        //         _builder = new ContinuousGardenBuilder<unsigned long long,
        //         float2,
        //                                                mydouble2>(
        //           param, data, cfg, obj, verbose.booster);
        //       } else {
        //         _builder =
        //           new ContinuousGardenBuilder<unsigned long long, float2,
        //           float2>(
        //             param, data, cfg, obj, verbose.booster);
        //       }
        //     } else {
        //       throw "unsupported depth";
        //     }
        //     _objective = obj;
        //   } break;
      default:
        throw "Unknown objective function " + cfg.objective;
    }

    // auto mem_per_rec = _builder->MemoryRequirementsPerRecord();
    size_t total;
    size_t free;

    cudaMemGetInfo(&free, &total);

    if (cfg.verbose.gpu) {
      printf("Total bytes %ld avaliable %ld \n", total, free);
      //   printf("Memory usage estimation %ld per record %ld in total \n",
      //          mem_per_rec, mem_per_rec * data->rows);
    }

    if (cfg.internal.upload_features)
      data->TransferToGPU(free * 9 / 10, cfg.verbose.gpu);

    _init = true;
  }

  if (grad == NULL) {
    _builder->UpdateGrad();
  } else {
    //          todo: fix
    //          data->grad = std::vector<float>(grad, grad + data->rows);
  }

  for (unsigned short i = 0; i < cfg.tree_param.labels_count; ++i) {
    RegTree *tree = new RegTree(cfg.tree_param.depth, i);
    _builder->GrowTree(tree, data, i);
    _trees.push_back(tree);
    if (grad == NULL) {
      //   tree->PredictByQuantized(data, data->y_internal);
      //   _builder->PredictByGrownTree(tree, data, data->y_internal);
    }
  }
}

void Garden::UpdateByLastTree(io::DataMatrix *data) {
  if (data->y_hat.size() == 0)
    data->y_hat.resize(data->rows * cfg.tree_param.labels_count,
                       _objective->IntoInternal(cfg.tree_param.initial_y));
  for (auto it = _trees.end() - cfg.tree_param.labels_count; it != _trees.end();
       ++it) {
    (*it)->Predict(data, data->y_hat);
  }
}

void Garden::GetY(arboretum::io::DataMatrix *data,
                  std::vector<float> &out) const {
  //   out.resize(y_internal.size());
  //   _objective->FromInternal(y_internal, out);
}

void Garden::Predict(const arboretum::io::DataMatrix *data,
                     std::vector<float> &out) const {
  out.resize(data->rows * cfg.tree_param.labels_count);
  thrust::host_vector<float> tmp(data->rows * cfg.tree_param.labels_count);

  thrust::fill(tmp.begin(), tmp.end(),
               _objective->IntoInternal(cfg.tree_param.initial_y));
  for (size_t i = 0; i < _trees.size(); ++i) {
    _trees[i]->Predict(data, tmp);
  }

  _objective->FromInternal(tmp, out);
}

const char *Garden::GetModel() const {
  std::vector<DecisionTree> tmp;
  for (size_t i = 0; i < this->_trees.size(); ++i) {
    tmp.push_back(*this->_trees[i]);
  }
  return DumpModel(this->cfg, tmp);
}

void Garden::Restore(const char *json_model) {
  std::vector<DecisionTree> trees = LoadModel(json_model);

  for (auto it = trees.begin(); it != trees.end(); ++it) {
    // TODO: get label
    RegTree *tree = new RegTree(cfg.tree_param.depth, 0);
    tree->weights = it->weights;
    tree->nodes = it->nodes;
    _trees.push_back(tree);
  }
}

Garden::~Garden() {
  if (_builder) delete _builder;
  if (_objective) delete _objective;
  for (size_t i = 0; i < _trees.size(); ++i) {
    delete _trees[i];
  }
}

}  // namespace core
}  // namespace arboretum
