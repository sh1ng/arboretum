#include <stdio.h>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
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

    template<typename InputIterator1, typename InputIterator2, typename OutputIterator1, typename OutputIterator2,
             typename BinaryPredicate, typename BinaryFunction>
    __global__ void reduce_by_key(InputIterator1 keys_first, InputIterator1 keys_last,
                                  InputIterator2 values_first, OutputIterator1 keys_output,
                                  OutputIterator2 values_output, BinaryPredicate binary_pred,
                                  BinaryFunction binary_op)
    {
      thrust::reduce_by_key(
                      thrust::cuda::par,
                      keys_first,
                      keys_last,
                      values_first,
                      keys_output,
                      values_output,
                      binary_pred,
                      binary_op
              );
    }


    template <typename T>
    struct max_gain_functor{
      typedef T first_argument_type;
      typedef T second_argument_type;
      typedef T result_type;

      __host__ __device__
      max_gain_functor(){}

      __host__ __device__
      T inline operator()(const T &l, const T &r) const {
        const T _lookup [2] { r, l };
        return _lookup[thrust::get<0>(l) > thrust::get<0>(r)];
      }
    };

    struct gain_functor{
      const int min_wieght;

      __host__ __device__
      gain_functor(const int min_wieght) : min_wieght(min_wieght) {}

      template <typename Tuple>
      __host__ __device__
      void inline operator()(Tuple t)
      {
        const double left_sum = thrust::get<0>(t);
        const size_t left_count = thrust::get<1>(t);
        const double total_sum = thrust::get<2>(t);
        const size_t total_count = thrust::get<3>(t);
        const float fvalue = thrust::get<5>(t);
        const float fvalue_prev = thrust::get<6>(t);
        const size_t right_count = total_count - left_count;

        if(left_count >= min_wieght && right_count >= min_wieght && fvalue != fvalue_prev){
            const size_t d = left_count * total_count * (total_count - left_count);

            const double top = total_count * left_sum - left_count * total_sum;
            thrust::get<4>(t) = top*top/d;
          } else {
            thrust::get<4>(t) = 0.0;
          }
      }
    };

    class GardenBuilder : public GardenBuilderBase {
    public:
      GardenBuilder(const TreeParam &param, const io::DataMatrix* data) : overlap_depth(2), param(param){
        _rowIndex2Node.resize(data->rows, 0);
        _featureNodeSplitStat.resize(data->columns);
        _bestSplit.resize(1 << (param.depth - 2));
        _nodeStat.resize(1 << (param.depth - 2));
        for(size_t fid = 0; fid < data->columns; ++fid){
            _featureNodeSplitStat[fid].resize(1 << param.depth);
          }
        gain = new device_vector<double>[overlap_depth];
        sum = new device_vector<double>[overlap_depth];
        count = new device_vector<size_t>[overlap_depth];
        segments = new device_vector<unsigned int>[overlap_depth];
        fvalue = new device_vector<float>[overlap_depth];
        position = new device_vector<int>[overlap_depth];
        grad_sorted = new device_vector<float>[overlap_depth];

        for(size_t i = 0; i < overlap_depth; ++i){
            cudaStream_t s;
            cudaStreamCreate(&s);
            streams[i] = s;
            gain[i]  = device_vector<double>(data->rows);
            sum[i]   = device_vector<double>(data->rows);
            count[i] = device_vector<size_t>(data->rows);
            segments[i] = device_vector<unsigned int>(data->rows);
            fvalue[i] = device_vector<float>(data->rows + 1);
            fvalue[i][0] = -std::numeric_limits<float>::infinity();
            position[i] = device_vector<int>(data->rows);
            grad_sorted[i] = device_vector<float>(data->rows);
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

      virtual void GrowTree(RegTree *tree, const io::DataMatrix *data, const thrust::host_vector<float> &grad) override{
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
      const TreeParam param;
      host_vector<unsigned int> _rowIndex2Node;
      std::vector<std::vector<SplitStat> > _featureNodeSplitStat;
      std::vector<NodeStat> _nodeStat;
      std::vector<Split> _bestSplit;

      device_vector<double> *gain = new device_vector<double>[overlap_depth];
      device_vector<double> *sum = new device_vector<double>[overlap_depth];
      device_vector<size_t> *count = new device_vector<size_t>[overlap_depth];
      device_vector<unsigned int> *segments = new device_vector<unsigned int>[overlap_depth];
      device_vector<float> *fvalue = new device_vector<float>[overlap_depth];
      device_vector<int> *position = new device_vector<int>[overlap_depth];
      device_vector<float> *grad_sorted = new device_vector<float>[overlap_depth];
      cudaStream_t *streams = new cudaStream_t[overlap_depth];

      void FindBestSplits(const int level, const io::DataMatrix *data, const thrust::host_vector<float> &grad){
        typedef thrust::device_vector<double>::iterator ElementDoubleIterator;
        typedef thrust::device_vector<size_t>::iterator ElementIntIterator;
        typedef thrust::device_vector<unsigned int>::iterator IndexIterator;
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



        device_vector<int> *max_key_d = new device_vector<int>[overlap_depth];
        device_vector<thrust::tuple<double, size_t>> *max_value_d = new device_vector<thrust::tuple<double, size_t>>[overlap_depth];
        host_vector<int> *max_key = new host_vector<int>[overlap_depth];
        host_vector<thrust::tuple<double, size_t>> *max_value = new host_vector<thrust::tuple<double, size_t>>[overlap_depth];

        for(size_t i = 0; i < overlap_depth; ++i){
            max_key_d[i] = device_vector<int>(1 << level, -1);
            max_value_d[i] = device_vector<thrust::tuple<double, size_t>>(1 << level);
            max_key[i] = host_vector<int>(1 << level, -1);
            max_value[i] = host_vector<thrust::tuple<double, size_t>>(1 << level);
          }

        thrust::equal_to<unsigned int> binary_pred;
        max_gain_functor< thrust::tuple<double, size_t> > binary_op;
        device_vector<unsigned int> row2Node = _rowIndex2Node;

                      for(size_t fid = 0; fid < data->columns; ++fid){
                          for(size_t i = 0; i < overlap_depth && (fid + i) < data->columns; ++i){

                              if(fid != 0 && i < overlap_depth - 1){
                                  continue;
                                }
                              size_t active_fid = fid + i;
                              size_t circular_fid = active_fid % overlap_depth;

                              cudaStream_t s = streams[circular_fid];

                              thrust::fill(thrust::cuda::par.on(s),
                                    max_key_d[circular_fid].begin(),
                                    max_key_d[circular_fid].end(),
                                           -1);

                              thrust::copy(thrust::cuda::par.on(s),
                                           data->sorted_grad[active_fid].begin(),
                                           data->sorted_grad[active_fid].end(),
                                           grad_sorted[circular_fid].begin());

                              thrust::copy(thrust::cuda::par.on(s),
                                           data->sorted_data[active_fid].begin(),
                                           data->sorted_data[active_fid].end(),
                                           fvalue[circular_fid].begin() + 1);

                              thrust::copy(thrust::cuda::par.on(s),
                                           data->index[active_fid].begin(),
                                           data->index[active_fid].end(),
                                           position[circular_fid].begin());

                              thrust::gather(thrust::cuda::par.on(s),
                                             position[circular_fid].begin(),
                                             position[circular_fid].end(),
                                             row2Node.begin(),
                                             segments[circular_fid].begin());

                              thrust::stable_sort_by_key(thrust::cuda::par.on(s),
                                                         segments[circular_fid].begin(),
                                                         segments[circular_fid].end(),
                                                         thrust::make_zip_iterator(
                                                           thrust::make_tuple(grad_sorted[circular_fid].begin(),
                                                           fvalue[circular_fid].begin() + 1)
                                                           ));


                              thrust::exclusive_scan_by_key(thrust::cuda::par.on(s),
                                                            segments[circular_fid].begin(),
                                                            segments[circular_fid].end(),
                                                            grad_sorted[circular_fid].begin(),
                                                            sum[circular_fid].begin());

                              thrust::constant_iterator<size_t> one_iter(1);

                              thrust::exclusive_scan_by_key(thrust::cuda::par.on(s),
                                                            segments[circular_fid].begin(),
                                                            segments[circular_fid].end(),
                                                            one_iter,
                                                            count[circular_fid].begin());

                              thrust::permutation_iterator<ElementDoubleIterator, IndexIterator>
                                  parent_node_sum_iter(parent_node_sum.begin(), segments[circular_fid].begin());
                              thrust::permutation_iterator<ElementIntIterator, IndexIterator>
                                  parent_node_count_iter(parent_node_count.begin(), segments[circular_fid].begin());


                              thrust::for_each(thrust::cuda::par.on(s),
                                    thrust::make_zip_iterator(
                                      thrust::make_tuple(sum[circular_fid].begin(),
                                                         count[circular_fid].begin(),
                                                         parent_node_sum_iter,
                                                         parent_node_count_iter,
                                                         gain[circular_fid].begin(),
                                                         fvalue[circular_fid].begin() + 1,
                                                         fvalue[circular_fid].begin())),
                                    thrust::make_zip_iterator(
                                      thrust::make_tuple(sum[circular_fid].end(),
                                                         count[circular_fid].end(),
                                                         parent_node_sum_iter + data->rows,
                                                         parent_node_count_iter + data->rows,
                                                         gain[circular_fid].end(),
                                                         fvalue[circular_fid].end(),
                                                         fvalue[circular_fid].end() - 1)),
                                  gain_functor(param.min_child_weight));


                              thrust::counting_iterator<size_t> iter(0);


                              auto tuple_iterator = thrust::make_zip_iterator(
                                    thrust::make_tuple(gain[circular_fid].begin(),
                                                       iter));

                              reduce_by_key<<<1,1,0,s>>>(
                                                    segments[circular_fid].begin(),
                                                    segments[circular_fid].end(),
                                                    tuple_iterator,
                                                    max_key_d[circular_fid].begin(),
                                                    max_value_d[circular_fid].begin(),
                                                    binary_pred,
                                                    binary_op);

                              thrust::copy(thrust::cuda::par.on(s),
                                           max_key_d[circular_fid].begin(),
                                           max_key_d[circular_fid].end(),
                                           max_key[circular_fid].begin());

                              thrust::copy(thrust::cuda::par.on(s),
                                           max_value_d[circular_fid].begin(),
                                           max_value_d[circular_fid].end(),
                                           max_value[circular_fid].begin());

                            }

                          size_t circular_fid = fid % overlap_depth;
                          cudaStreamSynchronize(streams[circular_fid]);


                          for(size_t i = 0; i < max_key[circular_fid].size(); ++i){
                              const int node_index = max_key[circular_fid][i];
                              const thrust::tuple<double, size_t> t = max_value[circular_fid][i];
                              const double gain_value = thrust::get<0>(t);

                              if(node_index >= 0 && gain_value > _bestSplit[node_index].gain){
                                  const size_t index_value = thrust::get<1>(t);
                                  const float fvalue_prev_val = fvalue[circular_fid][index_value];
                                  const float fvalue_val = fvalue[circular_fid][index_value + 1];
                                  const size_t count_val = count[circular_fid][index_value];
                                  const double sum_val = sum[circular_fid][index_value];
                                  _bestSplit[node_index].fid = fid;
                                  _bestSplit[node_index].gain = gain_value;
                                  _bestSplit[node_index].split_value = (fvalue_prev_val + fvalue_val) * 0.5;
                                  _bestSplit[node_index].count = count_val;
                                  _bestSplit[node_index].sum_grad = sum_val;
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
//                      for(size_t i = 0; i < overlap_depth; ++i){
//                          cudaStreamDestroy(streams[i]);
//                        }


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
              break;
            }

          data->Init(param.initial_y, gradFunc);
          _builder = new GardenBuilder(param, data);
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

