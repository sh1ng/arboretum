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


    template <typename T>
    struct max_gain_functor{
      typedef T first_argument_type;
      typedef T second_argument_type;
      typedef T result_type;

      max_gain_functor(){}

      __host__ __device__
      T operator()(const T &l, const T &r) const {
        if(thrust::get<0>(l) > thrust::get<0>(r))
          return l;
        else
          return r;
      }
    };

    struct gain_functor{
      const int min_wieght;

      gain_functor(int min_wieght) : min_wieght(min_wieght) {}

      template <typename Tuple>
      __host__ __device__
      void operator()(Tuple t)
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
      GardenBuilder(const TreeParam &param, const io::DataMatrix* data) : param(param){
        _rowIndex2Node.resize(data->rows, 0);
        _featureNodeSplitStat.resize(data->columns);
        _bestSplit.resize(1 << (param.depth - 2));
        _nodeStat.resize(1 << (param.depth - 2));
        for(size_t fid = 0; fid < data->columns; ++fid){
            _featureNodeSplitStat[fid].resize(1 << param.depth);
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
      const TreeParam param;
      std::vector<unsigned int> _rowIndex2Node;
      std::vector<std::vector<SplitStat> > _featureNodeSplitStat;
      std::vector<NodeStat> _nodeStat;
      std::vector<Split> _bestSplit;

      void FindBestSplits(const int level, const io::DataMatrix *data, const thrust::host_vector<float> &grad){

                      device_vector<unsigned int> row2Node = _rowIndex2Node;
                      for(size_t fid = 0; fid < data->columns; ++fid){
                          device_vector<unsigned int> segments(data->rows);
                          device_vector<float> grad(data->sorted_grad[fid].begin(), data->sorted_grad[fid].end());

                          device_vector<float> fvalue(data->rows + 1);
                          fvalue[0] = -std::numeric_limits<float>::infinity();
                          thrust::copy(data->sorted_data[fid].begin(), data->sorted_data[fid].end(), fvalue.begin() + 1);
                          device_vector<size_t> position(data->index[fid].begin(), data->index[fid].end());

                          thrust::gather(
                                         position.begin(),
                                         position.end(),
                                         row2Node.begin(),
                                         segments.begin());

                          thrust::stable_sort_by_key(segments.begin(),
                                                     segments.end(),
                                                     thrust::make_zip_iterator(
                                                       thrust::make_tuple(grad.begin(),
                                                                          fvalue.begin() + 1)
                                                       ));

                          device_vector<double> sum(data->rows);
                          device_vector<double> gain(data->rows);
                          device_vector<size_t> count(data->rows, 1);

                          thrust::equal_to<unsigned int> binary_pred;

                          cudaStream_t s1, s2;
                          cudaStreamCreate(&s1);
                          cudaStreamCreate(&s2);

                          thrust::exclusive_scan_by_key(thrust::cuda::par.on(s1),
                                                        segments.begin(),
                                                        segments.end(),
                                                        grad.begin(),
                                                        sum.begin());

                          thrust::exclusive_scan_by_key(thrust::cuda::par.on(s2),
                                                        segments.begin(),
                                                        segments.end(),
                                                        count.begin(),
                                                        count.begin());

                          // synchronize with both streams
                          cudaStreamSynchronize(s1);
                          cudaStreamSynchronize(s2);
                          // destroy streams
                          cudaStreamDestroy(s1);
                          cudaStreamDestroy(s2);



                          size_t lenght = 1 << level;

                          device_vector<double> parent_node_sum(lenght);
                          device_vector<size_t> parent_node_count(lenght);

                          for(size_t i = 0; i < lenght; ++i){
                              parent_node_count[i] = _nodeStat[i].count;
                              parent_node_sum[i] = _nodeStat[i].sum_grad;
                            }

                          device_vector<double> parent_node_sum_vector(data->rows, 0.0);
                          device_vector<size_t> parent_node_count_vector(data->rows, 0);

                          cudaStreamCreate(&s1);
                          cudaStreamCreate(&s2);

                          thrust::gather(thrust::cuda::par.on(s1),
                                         segments.begin(),
                                         segments.end(),
                                         parent_node_sum.begin(),
                                         parent_node_sum_vector.begin());

                          thrust::gather(thrust::cuda::par.on(s2),
                                         segments.begin(),
                                         segments.end(),
                                         parent_node_count.begin(),
                                         parent_node_count_vector.begin());

                          // synchronize with both streams
                          cudaStreamSynchronize(s1);
                          cudaStreamSynchronize(s2);
                          // destroy streams
                          cudaStreamDestroy(s1);
                          cudaStreamDestroy(s2);

                          thrust::for_each(
                                thrust::make_zip_iterator(
                                  thrust::make_tuple(sum.begin(), count.begin(), parent_node_sum_vector.begin(),
                                                     parent_node_count_vector.begin(), gain.begin(),
                                                     fvalue.begin() + 1, fvalue.begin())),
                                thrust::make_zip_iterator(
                                  thrust::make_tuple(sum.end(), count.end(), parent_node_sum_vector.end(),
                                                     parent_node_count_vector.end(), gain.end(),
                                                     fvalue.end(), fvalue.end() - 1)),
                              gain_functor(param.min_child_weight));

                          device_vector<int> max_key(1 << level, -1);
                          device_vector<thrust::tuple<double, size_t>> max_value(1 << level);

                          device_vector<size_t> index(data->rows);
                          thrust::sequence(index.begin(), index.end());

                          auto tuple_iterator = thrust::make_zip_iterator(
                                thrust::make_tuple(gain.begin(),
                                                   index.begin()));

                          max_gain_functor< thrust::tuple<double, size_t> > binary_op;

                          thrust::reduce_by_key(segments.begin(),
                                                segments.end(),
                                                tuple_iterator,
                                                max_key.begin(),
                                                max_value.begin(),
                                                binary_pred,
                                                binary_op);

                          for(size_t i = 0; i < max_key.size(); ++i){
                              const int node_index = max_key[i];
                              const thrust::tuple<double, size_t> t = max_value[i];
                              const double gain_value = thrust::get<0>(t);
                              const size_t index_value = thrust::get<1>(t);

                              if(node_index >= 0){
                                  if(gain_value > _bestSplit[node_index].gain){
                                      _bestSplit[node_index].fid = fid;
                                      _bestSplit[node_index].gain = gain_value;
                                      _bestSplit[node_index].split_value = (fvalue[index_value + 1] + fvalue[index_value]) * 0.5;
                                      _bestSplit[node_index].count = count[index_value];
                                      _bestSplit[node_index].sum_grad = sum[index_value];
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

