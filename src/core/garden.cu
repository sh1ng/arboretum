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
#include "garden.h"
#include "param.h"
#include "objective.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>


namespace arboretum {
  namespace core {
    using namespace std;
    using namespace thrust;
    using namespace thrust::cuda;
    using thrust::host_vector;
    using thrust::device_vector;
    using thrust::cuda::experimental::pinned_allocator;
    
    struct GainFunctionParameters{
    const unsigned int min_wieght;
    GainFunctionParameters(const unsigned int min_wieght) : min_wieght(min_wieght) {}
   };

    template <class type1, class type2>
    __global__ void gather_kernel(const int *position, const type1 *in1, type1 *out1, type2 *in2, type2 *out2, const size_t n){
      for (size_t i = blockDim.x * blockIdx.x + threadIdx.x;
               i < n;
               i += gridDim.x * blockDim.x){
          out1[i] = in1[position[i]];
          out2[i] = in2[position[i]];
        }
    }

    template <class node_type, class float_type>
    __global__ void gain_kernel(const float_type *left_sum, const float *fvalues,
                                const node_type *segments, const cub::TexObjInputIterator<float_type> parent_sum_iter,
                                const cub::TexObjInputIterator<int> parent_count_iter, const size_t n, const GainFunctionParameters parameters,
                                float_type *gain){
      for (size_t i = blockDim.x * blockIdx.x + threadIdx.x;
               i < n;
               i += gridDim.x * blockDim.x){
          const node_type segment = segments[i];

          const float_type left_sum_offset = parent_sum_iter[segment];
          const float_type left_sum_value = left_sum[i] - left_sum_offset;

          const size_t left_count_offset = parent_count_iter[segment];
          const size_t left_count_value = i - left_count_offset;

          const float_type total_sum = parent_sum_iter[segment + 1] - parent_sum_iter[segment];
          const size_t total_count = parent_count_iter[segment + 1] - parent_count_iter[segment];

          const float fvalue = fvalues[i + 1];
          const float fvalue_prev = fvalues[i];
          const size_t right_count = total_count - left_count_value;

          if(left_count_value >= parameters.min_wieght && right_count >= parameters.min_wieght && fvalue != fvalue_prev){
              const size_t d = left_count_value * total_count * (total_count - left_count_value);
              const float_type top = total_count * left_sum_value - left_count_value * total_sum;
              gain[i] = top*top/d;
            } else {
              gain[i] = 0.0;
            }
          }
    }

    template <typename node_type, typename float_type>
    class GardenBuilder : public GardenBuilderBase {
    public:
      GardenBuilder(const TreeParam &param, const io::DataMatrix* data) : overlap_depth(3),
        g_allocator(8, 3, 12, 1024L*1024L*1024L*2L, true), param(param), gain_param(param.min_child_weight){

        const int lenght = 1 << param.depth;

        int minGridSize; 
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSizeGain, gain_kernel<node_type, float_type>, 0, 0);
        gridSizeGain = (data->rows + blockSizeGain - 1) / blockSizeGain;

        minGridSize = 0;

        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSizeGather, gather_kernel<node_type, float_type>, 0, 0);
        gridSizeGather = (data->rows + blockSizeGather - 1) / blockSizeGather;

        row2Node.resize(data->rows);
        _rowIndex2Node.resize(data->rows, 0);
        _featureNodeSplitStat.resize(data->columns);
        _bestSplit.resize(1 << (param.depth - 2));
        _nodeStat.resize(1 << (param.depth - 2));
        for(size_t fid = 0; fid < data->columns; ++fid){
            _featureNodeSplitStat[fid].resize(1 << param.depth);
          }
        gain = new device_vector<float_type>[overlap_depth];
        sum = new device_vector<float_type>[overlap_depth];
        segments = new device_vector<node_type>[overlap_depth];
        segments_sorted = new device_vector<node_type>[overlap_depth];
        fvalue = new device_vector<float>[overlap_depth];
        position = new device_vector<int>[overlap_depth];
        grad_sorted = new device_vector<float_type>[overlap_depth];
        max_d = new device_vector<cub::KeyValuePair<int, float_type> >[overlap_depth];
        max_h = new host_vector<cub::KeyValuePair<int, float_type>, pinned_allocator<cub::KeyValuePair<int, float_type> > >[overlap_depth];

        parent_node_sum.resize(lenght + 1);
        parent_node_count.resize(lenght + 1);
        parent_node_sum_h.resize(lenght + 1);
        parent_node_count_h.resize(lenght + 1);


        for(size_t i = 0; i < overlap_depth; ++i){
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            streams[i] = s;
            gain[i]  = device_vector<float_type>(data->rows);
            sum[i]   = device_vector<float_type>(data->rows);
            segments[i] = device_vector<node_type>(data->rows);
            segments_sorted[i] = device_vector<node_type>(data->rows);
            fvalue[i] = device_vector<float>(data->rows + 1);
            fvalue[i][0] = -std::numeric_limits<float>::infinity();
            fvalue_sorted[i] = device_vector<float>(data->rows + 1);
            fvalue_sorted[i][0] = -std::numeric_limits<float>::infinity();
            position[i] = device_vector<int>(data->rows);
            grad_sorted[i] = device_vector<float_type>(data->rows);
            grad_sorted_sorted[i] = device_vector<float_type>(data->rows);
            max_d[i] = device_vector< cub::KeyValuePair<int, float_type> >(1 << param.depth);
            max_h[i] = thrust::host_vector< cub::KeyValuePair<int, float_type>, thrust::cuda::experimental::pinned_allocator<  cub::KeyValuePair<int, float_type> > >(1 << param.depth);
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
        grad_d = data->grad;

        for(int i = 0; i < param.depth - 1; ++i){
          InitTreeLevel(i);
          UpdateNodeStat(i, data, tree);
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
      host_vector<node_type, thrust::cuda::experimental::pinned_allocator< node_type > > _rowIndex2Node;
      std::vector<std::vector<SplitStat> > _featureNodeSplitStat;
      std::vector<NodeStat> _nodeStat;
      std::vector<Split> _bestSplit;


      device_vector<float_type> *gain = new device_vector<float_type>[overlap_depth];
      device_vector<float_type> *sum = new device_vector<float_type>[overlap_depth];
      device_vector<size_t> *count = new device_vector<size_t>[overlap_depth];
      device_vector<node_type> *segments = new device_vector<node_type>[overlap_depth];
      device_vector<node_type> *segments_sorted = new device_vector<node_type>[overlap_depth];
      device_vector<float> *fvalue = new device_vector<float>[overlap_depth];
      device_vector<float> *fvalue_sorted = new device_vector<float>[overlap_depth];
      device_vector<int> *position = new device_vector<int>[overlap_depth];
      device_vector<float_type> *grad_sorted = new device_vector<float_type>[overlap_depth];
      device_vector<float_type> *grad_sorted_sorted = new device_vector<float_type>[overlap_depth];
      cudaStream_t *streams = new cudaStream_t[overlap_depth];
      device_vector<cub::KeyValuePair<int, float_type> > *max_d;
      host_vector<cub::KeyValuePair<int, float_type>, pinned_allocator<cub::KeyValuePair<int, float_type> > > *max_h;
      device_vector<float_type> grad_d;
      device_vector<node_type> row2Node; 
      device_vector<float_type> parent_node_sum;
      device_vector<int> parent_node_count;
      host_vector<float_type> parent_node_sum_h;
      host_vector<int> parent_node_count_h;

      int blockSizeGain;
      int gridSizeGain;

      int blockSizeGather;
      int gridSizeGather;

      void FindBestSplits(const int level, const io::DataMatrix *data, const thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator< float > > &grad){

        cudaMemcpyAsync(thrust::raw_pointer_cast((row2Node.data())),
                        thrust::raw_pointer_cast(_rowIndex2Node.data()),
                        data->rows * sizeof(node_type),
                        cudaMemcpyHostToDevice, streams[0]);

        size_t lenght = 1 << level;

        {
          parent_node_sum_h[0] = 0.0;
          parent_node_count_h[0] = 0;

          for(size_t i = 0; i < lenght; ++i){
              parent_node_count_h[i + 1] = parent_node_count_h[i] + _nodeStat[i].count;
              parent_node_sum_h[i + 1] = parent_node_sum_h[i] + _nodeStat[i].sum_grad;
            }
          parent_node_sum = parent_node_sum_h;
          parent_node_count = parent_node_count_h;
        }

        cub::TexObjInputIterator<float_type> parent_sum_iter;
        parent_sum_iter.BindTexture(thrust::raw_pointer_cast(parent_node_sum.data()), sizeof(float_type) * (lenght + 1));

        cub::TexObjInputIterator<int> parent_count_iter;
        parent_count_iter.BindTexture(thrust::raw_pointer_cast(parent_node_count.data()), sizeof(int) * (lenght + 1));



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

                              cudaStreamSynchronize(s);

                              gather_kernel<<<gridSizeGather, blockSizeGather, 0, s >>>(thrust::raw_pointer_cast(position[circular_fid].data()),
                                                                          thrust::raw_pointer_cast(row2Node.data()),
                                                                          thrust::raw_pointer_cast(segments[circular_fid].data()),
                                                                          thrust::raw_pointer_cast(grad_d.data()),
                                                                          thrust::raw_pointer_cast(grad_sorted[circular_fid].data()),
                                                                          data->rows);

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
                                                                      level + 1,
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
                                                                      level + 1,
                                                                      s));

                              CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                                                      temp_storage_bytes,
                                                                      thrust::raw_pointer_cast(segments[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(segments_sorted[circular_fid].data()),
                                                                      thrust::raw_pointer_cast(fvalue[circular_fid].data() + 1),
                                                                      thrust::raw_pointer_cast(fvalue_sorted[circular_fid].data() + 1),
                                                                      data->rows,
                                                                      0,
                                                                      level + 1,
                                                                      s));
                              if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
                              d_temp_storage = NULL;
                              temp_storage_bytes = 0;

                              CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                                                    temp_storage_bytes,
                                                                    thrust::raw_pointer_cast(grad_sorted_sorted[circular_fid].data()),
                                                                    thrust::raw_pointer_cast(sum[circular_fid].data()),
                                                                    data->rows,
                                                                    s));

                              CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes, s));

                              CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                                                    temp_storage_bytes,
                                                                    thrust::raw_pointer_cast(grad_sorted_sorted[circular_fid].data()),
                                                                    thrust::raw_pointer_cast(sum[circular_fid].data()),
                                                                    data->rows,
                                                                    s));

                              if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
                              d_temp_storage = NULL;
                              temp_storage_bytes = 0;

                              gain_kernel<<<gridSizeGain, blockSizeGain, 0, s >>>(thrust::raw_pointer_cast(sum[circular_fid].data()),
                                                                          thrust::raw_pointer_cast(fvalue_sorted[circular_fid].data()),
                                                                          thrust::raw_pointer_cast(segments_sorted[circular_fid].data()),
                                                                          parent_sum_iter,
                                                                          parent_count_iter,
                                                                          data->rows,
                                                                          gain_param,
                                                                          thrust::raw_pointer_cast(gain[circular_fid].data()));

                              cub::DeviceSegmentedReduce::ArgMax(d_temp_storage,
                                             temp_storage_bytes,
                                             thrust::raw_pointer_cast(gain[circular_fid].data()),
                                             thrust::raw_pointer_cast(max_d[circular_fid].data()),
                                             lenght,
                                             thrust::raw_pointer_cast(parent_node_count.data()),
                                             thrust::raw_pointer_cast(parent_node_count.data() + 1),
                                             s);

                              CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes, s));

                              cub::DeviceSegmentedReduce::ArgMax(d_temp_storage,
                                             temp_storage_bytes,
                                             thrust::raw_pointer_cast(gain[circular_fid].data()),
                                             thrust::raw_pointer_cast(max_d[circular_fid].data()),
                                             lenght,
                                             thrust::raw_pointer_cast(parent_node_count.data()),
                                             thrust::raw_pointer_cast(parent_node_count.data() + 1),
                                             s);

                              if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

                              cudaMemcpyAsync(thrust::raw_pointer_cast(max_h[circular_fid].data()),
                                              thrust::raw_pointer_cast(max_d[circular_fid].data()),
                                              lenght * sizeof(cub::KeyValuePair<int, float_type>),
                                              cudaMemcpyDeviceToHost, s);
                            }

                          size_t circular_fid = fid % overlap_depth;

                          cudaStream_t s = streams[circular_fid];

                          cudaStreamSynchronize(s);

                          for(size_t i = 0; i < lenght;  ++i){
                              if(_nodeStat[i].count <= 0) continue;

                              if(max_h[circular_fid][i].value > _bestSplit[i].gain){
                                const int index_value = max_h[circular_fid][i].key + parent_node_count_h[i];
                                const float fvalue_prev_val = fvalue_sorted[circular_fid][index_value];
                                const float fvalue_val = fvalue_sorted[circular_fid][index_value + 1];
                                const size_t count_val = max_h[circular_fid][i].key;
                                const float_type sum_val = sum[circular_fid][index_value] - parent_node_sum_h[i];
                                _bestSplit[i].fid = fid;
                                _bestSplit[i].gain = max_h[circular_fid][i].value;
                                _bestSplit[i].split_value = (fvalue_prev_val + fvalue_val) * 0.5;
                                _bestSplit[i].count = count_val;
                                _bestSplit[i].sum_grad = sum_val;
                                }
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
      void UpdateNodeStat(const int level, const io::DataMatrix *data, const RegTree *tree){
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
          }
          } else {
            _nodeStat[0].count = data->rows;
            double sum = 0.0;
            #pragma omp parallel for default(shared) reduction(+:sum)
            for(size_t i = 0; i < data->rows; ++i){
                sum += data->grad[i];
              }
            _nodeStat[0].sum_grad = sum;
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
        #pragma omp parallel for
        for(size_t i = 0; i < data->rows; ++i){
            const unsigned int node = _rowIndex2Node[i];
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
          std::function<float(float const, float const)> gradFunc;
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
            _builder = new GardenBuilder<unsigned short, float>(param, data);
          else if(param.depth + 1 <= sizeof(unsigned int) * CHAR_BIT)
            _builder = new GardenBuilder<unsigned int, float>(param, data);
          else if(param.depth + 1 <= sizeof(unsigned long int) * CHAR_BIT)
            _builder = new GardenBuilder<unsigned long int, float>(param, data);
          else
            throw "unsupported depth";

          _init = true;
        }

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

