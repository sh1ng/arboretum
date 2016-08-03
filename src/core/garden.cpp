#include <stdio.h>
#include <limits>
#include <omp.h>
#include "garden.h"
#include "param.h"
#include "objective.h"


namespace arboretum {
  namespace core {
    using namespace std;

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

      virtual void GrowTree(RegTree *tree, const io::DataMatrix *data, const std::vector<float> &grad) override{
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

      void FindBestSplits(const int level, const io::DataMatrix *data, const std::vector<float> &grad){

                      size_t row_index;
                      size_t node_index;

                      for(size_t fid = 0; fid < data->columns; ++fid){

                          const std::vector<float> &feature_values = data->sorted_data[fid];
                          const std::vector<float> &grad_values = data->sorted_grad[fid];
                          std::vector<SplitStat> &node_split = _featureNodeSplitStat[fid];
                          for(size_t j = 0; j < data->rows; ++j){
                              row_index = data->index[fid][j];
                              node_index = _rowIndex2Node[row_index];
                              const NodeStat &parent_node_stat = _nodeStat[node_index];
                              SplitStat &split = node_split[node_index];
                              const float feature_value = feature_values[j];

                              if(split.count >= param.min_child_weight && split.last_value != feature_value
                                 && (parent_node_stat.count - split.count) >= param.min_child_weight){

                                  const double gain = split.GainForSplit(parent_node_stat);

                                  if(gain > _bestSplit[node_index].gain){

                                      _bestSplit[node_index].fid = fid;
                                      _bestSplit[node_index].gain = gain;
                                      _bestSplit[node_index].split_value = (split.last_value + feature_value) * 0.5;
                                      _bestSplit[node_index].count = split.count;
                                      _bestSplit[node_index].sum_grad = split.sum_grad;
                                    }
                                }

                              split.count +=1;
                              split.sum_grad += grad_values[j];
                              split.last_value = feature_value;
                              }

                          for(size_t i = 0, len = 1 << level; i < len; ++i){
                              NodeStat &node_stat = _nodeStat[i];
                              Split &split = _bestSplit[i];

                              if(split.fid < 0 || split.gain < node_stat.gain){
                                  _bestSplit[i].gain = 0.0;
                                  _bestSplit[i].fid = 0;
                                  _bestSplit[i].split_value = std::numeric_limits<float>::infinity();
                                  _bestSplit[i].count = node_stat.count;
                                  _bestSplit[i].sum_grad = node_stat.sum_grad;
                                }
                            }

                        }

      }
      void UpdateNodeStat(const int level, const std::vector<float> &grad, const RegTree *tree){
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

