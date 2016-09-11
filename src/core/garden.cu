#define CUB_STDERR

#include <climits>
#include <stdio.h>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
#include "garden.h"
#include "param.h"
#include "objective.h"


namespace arboretum {
  namespace core {
    using namespace std;
    using namespace thrust;
    using namespace thrust::cuda;
    using thrust::host_vector;
    using thrust::device_vector;
    
    struct GainFunctionParameters{
    const unsigned int min_wieght;
    GainFunctionParameters(const unsigned int min_wieght) : min_wieght(min_wieght) {}
   };

    template <class node_type>
    __global__ void gain_kernel(const float *left_sum, const size_t *left_count, float *fvalues, float *fvalue_prevs,
                                node_type *segments, cub::TexObjInputIterator<double> parent_sum_iter,
                                cub::TexObjInputIterator<size_t> parent_count_iter, size_t n, const GainFunctionParameters parameters,
                                double *gain){
      for (int i = blockDim.x * blockIdx.x + threadIdx.x;
               i < n;
               i += gridDim.x * blockDim.x){
          const unsigned int segment = segments[i];
          const double left_sum_value = left_sum[i];
          const size_t left_count_value = left_count[i];
          const double total_sum = parent_sum_iter[segment];
          const size_t total_count = parent_count_iter[segment];
          const float fvalue = fvalues[i];
          const float fvalue_prev = fvalue_prevs[i];
          const size_t right_count = total_count - left_count_value;

          if(left_count_value >= parameters.min_wieght && right_count >= parameters.min_wieght && fvalue != fvalue_prev){
              const size_t d = left_count_value * total_count * (total_count - left_count_value);
              const double top = total_count * left_sum_value - left_count_value * total_sum;
              gain[i] = top*top/d;
            } else {
              gain[i] = 0.0;
            }
          }
    }

    template <class node_type>
    class GardenBuilder : public GardenBuilderBase {
    public:
      GardenBuilder(const TreeParam &param, const io::DataMatrix* data) : overlap_depth(2),
        g_allocator(cub::CachingDeviceAllocator(32, 1, 5)), param(param), gain_param(param.min_child_weight){
        int minGridSize; 
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, gain_kernel<node_type>, 0, 0);
        gridSize = (data->rows + blockSize - 1) / blockSize;
        
        _rowIndex2Node.resize(data->rows, 0);
        _featureNodeSplitStat.resize(data->columns);
        _bestSplit.resize(1 << (param.depth - 2));
        _nodeStat.resize(1 << (param.depth - 2));
        for(size_t fid = 0; fid < data->columns; ++fid){
            _featureNodeSplitStat[fid].resize(1 << param.depth);
          }
        gain = new device_vector<double>[overlap_depth];
        sum = new device_vector<float>[overlap_depth];
        count = new device_vector<size_t>[overlap_depth];
        segments = new device_vector<node_type>[overlap_depth];
        segments_sorted = new device_vector<node_type>[overlap_depth];
        fvalue = new device_vector<float>[overlap_depth];
        position = new device_vector<int>[overlap_depth];
        grad_sorted = new device_vector<float>[overlap_depth];

        for(size_t i = 0; i < overlap_depth; ++i){
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            streams[i] = s;
            gain[i]  = device_vector<double>(data->rows);
            sum[i]   = device_vector<float>(data->rows);
            count[i] = device_vector<size_t>(data->rows);
            segments[i] = device_vector<node_type>(data->rows);
            segments_sorted[i] = device_vector<node_type>(data->rows);
            fvalue[i] = device_vector<float>(data->rows + 1);
            fvalue[i][0] = -std::numeric_limits<float>::infinity();
            fvalue_sorted[i] = device_vector<float>(data->rows + 1);
            fvalue_sorted[i][0] = -std::numeric_limits<float>::infinity();
            position[i] = device_vector<int>(data->rows);
            grad_sorted[i] = device_vector<float>(data->rows);
            grad_sorted_sorted[i] = device_vector<float>(data->rows);
          }

    }
      virtual void InitGrowingTree() override {
        std::fill(_rowIndex2Node.begin(), _rowIndex2Node.end(), 0);
        for(size_t i = 0; i < _featureNodeSplitStat.size(); ++i){
            for(size_t j = 0; j < _featureNodeSplitStat[i].size(); ++j){
                _featureNodeSplitStat[i][j].Clean();
              }
          }
        for(size_t i = 0; i < _nodeStat.size(); ++i){
            _nodeStat[i].Clean();
          }
        for(size_t i = 0; i < _bestSplit.size(); ++i){
            _bestSplit[i].Clean();
          }
      }

      virtual void InitTreeLevel(const int level) override {
        for(size_t i = 0; i < _featureNodeSplitStat.size(); ++i){
            for(size_t j = 0; j < _featureNodeSplitStat[i].size(); ++j){
                _featureNodeSplitStat[i][j].Clean();
              }
          }
      }

      virtual void GrowTree(RegTree *tree, const io::DataMatrix *data, const thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator< float > > &grad) override{
        InitGrowingTree();

        for(int i = 0; i < param.depth - 1; ++i){
          InitTreeLevel(i);
          UpdateNodeStat(i, grad, tree);
          FindBestSplits(i, data, grad);
          UpdateTree(i, tree);
          UpdateNodeIndex(i, data, tree);
        }

        UpdateLeafWeight(tree);
      }

      virtual void PredictByGrownTree(RegTree *tree, const io::DataMatrix *data, std::vector<float> &out) override {
        tree->Predict(data, _rowIndex2Node, out);
      }

    private:
      const unsigned short overlap_depth;
      cub::CachingDeviceAllocator g_allocator;
      const TreeParam param;
      const GainFunctionParameters gain_param;
      host_vector<node_type, thrust::cuda::experimental::pinned_allocator< unsigned int > > _rowIndex2Node;
      std::vector<std::vector<SplitStat> > _featureNodeSplitStat;
      std::vector<NodeStat> _nodeStat;
      std::vector<Split> _bestSplit;


      device_vector<double> *gain = new device_vector<double>[overlap_depth];
      device_vector<float> *sum = new device_vector<float>[overlap_depth];
      device_vector<size_t> *count = new device_vector<size_t>[overlap_depth];
      device_vector<node_type> *segments = new device_vector<node_type>[overlap_depth];
      device_vector<node_type> *segments_sorted = new device_vector<node_type>[overlap_depth];
      device_vector<float> *fvalue = new device_vector<float>[overlap_depth];
      device_vector<float> *fvalue_sorted = new device_vector<float>[overlap_depth];
      device_vector<int> *position = new device_vector<int>[overlap_depth];
      device_vector<float> *grad_sorted = new device_vector<float>[overlap_depth];
      device_vector<float> *grad_sorted_sorted = new device_vector<float>[overlap_depth];
      cudaStream_t *streams = new cudaStream_t[overlap_depth];
      int blockSize;
      int gridSize;

      void FindBestSplits(const int level, const io::DataMatrix *data, const thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator< float > > &grad){
        size_t lenght = 1 << level;

        device_vector<double> parent_node_sum(lenght);
        device_vector<size_t> parent_node_count(lenght);
        {
          host_vector<double> parent_node_sum_h(lenght);
          host_vector<size_t> parent_node_count_h(lenght);

          for(size_t i = 0; i < lenght; ++i){
              parent_node_count_h[i] = _nodeStat[i].count;
              parent_node_sum_h[i] = _nodeStat[i].sum_grad;
            }
          parent_node_sum = parent_node_sum_h;
          parent_node_count = parent_node_count_h;
        }

        cub::TexObjInputIterator<double> parent_sum_iter;
        parent_sum_iter.BindTexture(thrust::raw_pointer_cast(parent_node_sum.data()), sizeof(double) * lenght);

        cub::TexObjInputIterator<size_t> parent_count_iter;
        parent_count_iter.BindTexture(thrust::raw_pointer_cast(parent_node_count.data()), sizeof(size_t) * lenght);


        device_vector<int> *max_key_d = new device_vector<int>[overlap_depth];
        device_vector<thrust::tuple<double, size_t>> *max_value_d = new device_vector<thrust::tuple<double, size_t>>[overlap_depth];
        host_vector<int> *max_key = new host_vector<int>[overlap_depth];
        host_vector<thrust::tuple<double, size_t>> *max_value = new host_vector<thrust::tuple<double, size_t>>[overlap_depth];
        size_t n = 1 << level;
        device_vector<unsigned int> *result = new device_vector<unsigned int>[overlap_depth];
        thrust::device_vector<unsigned int> *data_ = new thrust::device_vector<unsigned int>[overlap_depth];
        for(size_t i = 0; i < overlap_depth; ++i){
            result[i] = device_vector<unsigned int>(1, 0);
            data_[i] = thrust::device_vector<unsigned int>(n, 1);

            max_key_d[i] = device_vector<int>(1 << level, -1);
            max_value_d[i] = device_vector<thrust::tuple<double, size_t>>(1 << level);
            max_key[i] = host_vector<int>(1 << level, -1);
            max_value[i] = host_vector<thrust::tuple<double, size_t>>(1 << level);
          }

        device_vector<node_type> row2Node = _rowIndex2Node;
        device_vector<float> grad_d = data->grad;


                      for(size_t fid = 0; fid < data->columns; ++fid){
                          for(size_t i = 0; i < overlap_depth && (fid + i) < data->columns; ++i){

                              if(fid != 0 && i < overlap_depth - 1){
                                  continue;
                                }

                              size_t active_fid = fid + i;
                              size_t circular_fid = active_fid % overlap_depth;

                              cudaStream_t s = streams[circular_fid];

                              cudaMemcpyAsync(thrust::raw_pointer_cast((&fvalue[circular_fid].data()[1])),
                                              thrust::raw_pointer_cast(data->sorted_data[active_fid].data()),
                                              data->rows * sizeof(float),
                                              cudaMemcpyHostToDevice, s);

                              cudaMemcpyAsync(thrust::raw_pointer_cast(position[circular_fid].data()),
                                              thrust::raw_pointer_cast(data->index[active_fid].data()),
                                              data->rows * sizeof(int),
                                              cudaMemcpyHostToDevice, s);

                              thrust::fill(thrust::cuda::par.on(s),
                                    max_key_d[circular_fid].begin(),
                                    max_key_d[circular_fid].end(),
                                           -1);

                              thrust::gather(thrust::cuda::par.on(s),
                                             position[circular_fid].begin(),
                                             position[circular_fid].end(),
                                             grad_d.begin(),
                                             grad_sorted[circular_fid].begin());

                              thrust::gather(thrust::cuda::par.on(s),
                                             position[circular_fid].begin(),
                                             position[circular_fid].end(),
                                             row2Node.begin(),
                                             segments[circular_fid].begin());

                              size_t  temp_storage_bytes  = 0;
                              void *d_temp_storage = NULL;


                              CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                                                      temp_storage_bytes,
                                                                      thrust::raw_pointer_cast(segments[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(segments_sorted[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(grad_sorted[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(grad_sorted_sorted[circular_fid].data()),
                                                                      data->rows,
                                                                      0,
                                                                      sizeof(unsigned int) * 8,
                                                                      s));

                              CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes, s));

                              CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                                                      temp_storage_bytes,
                                                                      thrust::raw_pointer_cast(segments[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(segments_sorted[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(grad_sorted[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(grad_sorted_sorted[circular_fid].data()),
                                                                      data->rows,
                                                                      0,
                                                                      sizeof(unsigned int) * 8,
                                                                      s));

                              CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                                                      temp_storage_bytes,
                                                                      thrust::raw_pointer_cast(segments[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(segments_sorted[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(fvalue[circular_fid].data() + 1),
                                                                      thrust::raw_pointer_cast(fvalue_sorted[circular_fid].data() + 1),
                                                                      data->rows,
                                                                      0,
                                                                      sizeof(unsigned int) * 8,
                                                                      s));
                              if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
                            }

                          size_t circular_fid = fid % overlap_depth;

                          cudaStream_t s = streams[circular_fid];

                          size_t  temp_storage_bytes  = 0;
                          void *d_temp_storage = NULL;
                          size_t temp_storage_count_bytes = 0;
                          void *d_temp_count_storage = NULL;

                          size_t offset = 0;
                          cub::ConstantInputIterator<size_t> one_iter(1);

                          if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
                          for(size_t i = 0; i < lenght && _nodeStat[i].count > 0;  ++i){
                              d_temp_count_storage = d_temp_storage = NULL;
                              temp_storage_count_bytes = temp_storage_bytes = 0;

                              CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                                                    temp_storage_bytes,
                                                                    thrust::raw_pointer_cast(grad_sorted_sorted[circular_fid].data() + offset),
                                                                    thrust::raw_pointer_cast(sum[circular_fid].data() + offset),
                                                                    _nodeStat[i].count,
                                                                    s));

                              CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes, s));

                              CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                                                    temp_storage_bytes,
                                                                    thrust::raw_pointer_cast(grad_sorted_sorted[circular_fid].data() + offset),
                                                                    thrust::raw_pointer_cast(sum[circular_fid].data() + offset),
                                                                    _nodeStat[i].count,
                                                                    s));

                              CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_count_storage,
                                                                    temp_storage_count_bytes,
                                                                    one_iter,
                                                                    thrust::raw_pointer_cast(count[circular_fid].data() + offset),
                                                                    _nodeStat[i].count,
                                                                    s));

                              CubDebugExit(g_allocator.DeviceAllocate(&d_temp_count_storage, temp_storage_count_bytes, s));

                              CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_count_storage,
                                                                    temp_storage_count_bytes,
                                                                    one_iter,
                                                                    thrust::raw_pointer_cast(count[circular_fid].data() + offset),
                                                                    _nodeStat[i].count,
                                                                    s));

                              offset += _nodeStat[i].count;
                              if (d_temp_count_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_count_storage));
                              if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

                            }

                          gain_kernel<<<gridSize, blockSize, 0, s >>>(thrust::raw_pointer_cast(sum[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(count[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(fvalue_sorted[circular_fid].data() + 1),
                                                                      thrust::raw_pointer_cast(fvalue_sorted[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(segments_sorted[circular_fid].data()),
                                                                      parent_sum_iter,
                                                                      parent_count_iter,
                                                                      data->rows,
                                                                      gain_param,
                                                                      thrust::raw_pointer_cast(gain[circular_fid].data()));


                          size_t offset_max = 0;
                          cub::KeyValuePair<int, double> *max_pair_d;
                          cub::KeyValuePair<int, double> *max_pair_h = new cub::KeyValuePair<int, double>();
                          CubDebugExit(g_allocator.DeviceAllocate((void**)&max_pair_d, sizeof(cub::KeyValuePair<int, double>) * 1, s));

                          for(size_t i = 0; i < lenght && _nodeStat[i].count > 0;  ++i){
                              void     *d_temp_storage_max = NULL;
                              size_t   temp_storage_bytes_max = 0;
                              cub::DeviceReduce::ArgMax(d_temp_storage_max,
                                                        temp_storage_bytes_max,
                                                        thrust::raw_pointer_cast(gain[circular_fid].data() + offset_max),
                                                        max_pair_d,
                                                        _nodeStat[i].count,
                                                        s);

                              CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage_max, temp_storage_bytes_max, s));

                              cub::DeviceReduce::ArgMax(d_temp_storage_max,
                                                        temp_storage_bytes_max,
                                                        thrust::raw_pointer_cast(gain[circular_fid].data() + offset_max),
                                                        max_pair_d,
                                                        _nodeStat[i].count,
                                                        s);
                              cudaMemcpyAsync(max_pair_h,
                                              max_pair_d,
                                              sizeof(cub::KeyValuePair<int, double>),
                                              cudaMemcpyDeviceToHost, s);

                              cudaStreamSynchronize(s);

                              if(max_pair_h->value > _bestSplit[i].gain){
                                const int index_value = max_pair_h->key + offset_max;
                                const float fvalue_prev_val = fvalue_sorted[circular_fid][index_value];
                                const float fvalue_val = fvalue_sorted[circular_fid][index_value + 1];
                                const size_t count_val = count[circular_fid][index_value];
                                const double sum_val = sum[circular_fid][index_value];
                                _bestSplit[i].fid = fid;
                                _bestSplit[i].gain = max_pair_h->value;
                                _bestSplit[i].split_value = (fvalue_prev_val + fvalue_val) * 0.5;
                                _bestSplit[i].count = count_val;
                                _bestSplit[i].sum_grad = sum_val;
                                }
                              if (d_temp_storage_max) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_max));
                              offset_max += _nodeStat[i].count;
                            }
                        }

                      for(size_t i = 0; i < lenght; ++i){
                          NodeStat &node_stat = _nodeStat[i];
                          Split &split = _bestSplit[i];

                          if(split.fid < 0){
                              _bestSplit[i].gain = 0.0;
                              _bestSplit[i].fid = 0;
                              _bestSplit[i].split_value = std::numeric_limits<float>::infinity();
                              _bestSplit[i].count = node_stat.count;
                              _bestSplit[i].sum_grad = node_stat.sum_grad;
                            }
                        }
      }
      void UpdateNodeStat(const int level, const thrust::host_vector<float> &grad, const RegTree *tree){
        if(level != 0){
        const unsigned int offset = Node::HeapOffset(level);
        const unsigned int offset_next = Node::HeapOffset(level + 1);
        std::vector<NodeStat> tmp(_nodeStat.size());
        std::copy(_nodeStat.begin(), _nodeStat.end(), tmp.begin());
        for(size_t i = 0, len = 1 << (level - 1); i < len; ++i){
            _nodeStat[tree->ChildNode(i + offset, true) - offset_next].count = _bestSplit[i].count;
            _nodeStat[tree->ChildNode(i + offset, true) - offset_next].sum_grad = _bestSplit[i].sum_grad;

            _nodeStat[tree->ChildNode(i + offset, false) - offset_next].count =
                tmp[i].count - _bestSplit[i].count;

            _nodeStat[tree->ChildNode(i + offset, false) - offset_next].sum_grad =
                tmp[i].sum_grad - _bestSplit[i].sum_grad;

            _bestSplit[i].Clean();
          }
          } else {
            for(size_t i = 0; i < grad.size(); ++i){
                int node = _rowIndex2Node[i];
                _nodeStat[node].count++;
                _nodeStat[node].sum_grad += grad[i];
              }
          }
        for(size_t i = 0, len = 1 << level; i < len; ++i){
            _nodeStat[i].gain = (_nodeStat[i].sum_grad * _nodeStat[i].sum_grad) / _nodeStat[i].count;
            _bestSplit[i].Clean();
          }
      }

      void UpdateTree(const int level, RegTree *tree) const {
        unsigned int offset = Node::HeapOffset(level);
        for(size_t i = 0, len = 1 << level; i < len; ++i){
            const Split &best = _bestSplit[i];
            tree->nodes[i + offset].threshold = best.split_value;
            tree->nodes[i + offset].fid = best.fid;
            if(tree->nodes[i + offset].fid < 0){
                tree->nodes[i + offset].fid = 0;
              }
          }
      }

      void UpdateNodeIndex(const unsigned int level, const io::DataMatrix *data, RegTree *tree) {
        unsigned int offset = Node::HeapOffset(level);
        unsigned int offset_next = Node::HeapOffset(level + 1);
        unsigned int node;
        for(size_t i = 0; i < data->rows; ++i){
            node = _rowIndex2Node[i];
            Split &best = _bestSplit[node];
            _rowIndex2Node[i] = tree->ChildNode(node + offset, data->data[best.fid][i] <= best.split_value) - offset_next;
          }
      }

      void UpdateLeafWeight(RegTree *tree) const {
        const unsigned int offset_1 = Node::HeapOffset(tree->depth - 2);
        const unsigned int offset = Node::HeapOffset(tree->depth - 1);
        for(unsigned int i = 0, len = (1 << (tree->depth - 2)); i < len; ++i){
            const Split &best = _bestSplit[i];
            const NodeStat &stat = _nodeStat[i];
            tree->leaf_level[tree->ChildNode(i + offset_1, true) - offset] = (best.sum_grad / best.count) * param.eta * (-1);
            tree->leaf_level[tree->ChildNode(i + offset_1, false) - offset] = ((stat.sum_grad - best.sum_grad) / (stat.count - best.count)) * param.eta * (-1);
          }
      }
    };

    Garden::Garden(const TreeParam& param) : param(param), _init(false) {}
    void Garden::GrowTree(io::DataMatrix* data, float *grad){

      if(!_init){
          std::function<float const(float const, float const)> gradFunc;
          switch (param.objective) {
            case LinearRegression:
              gradFunc = GradBuilder::Regression;
              break;
            case LogisticRegression:
              gradFunc = GradBuilder::LogReg;
              break;
            default:
               throw "Unknown objective function";
            }

          data->Init(param.initial_y, gradFunc);

          if(param.depth + 1 <= sizeof(unsigned short) * CHAR_BIT)
            _builder = new GardenBuilder<unsigned short>(param, data);
          else if(param.depth + 1 <= sizeof(unsigned int) * CHAR_BIT)
            _builder = new GardenBuilder<unsigned int>(param, data);
          else if(param.depth + 1 <= sizeof(unsigned long int) * CHAR_BIT)
            _builder = new GardenBuilder<unsigned long int>(param, data);
          else
            throw "unsupported depth";

          _init = true;
        }

      _builder->InitGrowingTree();

      if(grad == NULL){
          SetInitial(data, data->y);
          data->UpdateGrad();
        } else {
          data->grad = std::vector<float>(grad, grad + data->rows);
        }

        RegTree *tree = new RegTree(param.depth);
        _builder->GrowTree(tree, data, data->grad);
        _trees.push_back(tree);
        if(grad == NULL){
            _builder->PredictByGrownTree(tree, data, data->y);
          }
      }

    void Garden::Predict(const arboretum::io::DataMatrix *data, std::vector<float> &out){
      out.resize(data->rows);
      std::fill(out.begin(), out.end(), param.initial_y);
      for(size_t i = 0; i < _trees.size(); ++i){
          _trees[i]->Predict(data, out);
        }
    }

    void Garden::SetInitial(const arboretum::io::DataMatrix *data, std::vector<float> &out){
      if(out.size() != data->rows){
          out.resize(data->rows);
          std::fill(out.begin(), out.end(), param.initial_y);
        }
    }
    }
  }

