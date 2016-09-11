#ifndef BOOSTER_H
#define BOOSTER_H

//#include <omp.h>
#include <cmath>
#include <limits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "param.h"
#include "../io/io.h"

namespace arboretum {
  namespace core {

    using namespace arboretum;

    struct Split {
      float split_value;
      int fid;
      double gain;
      double sum_grad;
      unsigned int count;
      Split(){
        Clean();
      }
      void Clean(){
        fid = -1;
        sum_grad = gain = 0.0;
        count = 0;
        split_value = std::numeric_limits<float>::infinity();
      }
    };

    struct NodeStat {
      int count;
      double sum_grad;
      double gain;
      NodeStat(){
        Clean();
      }

      void Clean(){
        count = 0;
        gain = sum_grad = 0.0;
      }

      inline float LeafWeight() const {
        return sum_grad / count;
      }
    };

    struct Node {
      static inline unsigned int Left(unsigned int parent){
        return 2 * parent + 1;
      }

      static inline unsigned int Right(unsigned int parent){
        return 2 * parent + 2;
      }

      static inline unsigned int HeapOffset(unsigned int level){
        return (1 << level) - 1;
      }

      Node(int id) : id(id), fid(-111){}

      unsigned id;
      float threshold;
      int fid;
    };


    struct SplitStat {
      int count;
      double sum_grad;
      float last_value;
      SplitStat() {
        Clean();
      }
      inline double GainForSplit(const NodeStat& nodeStat){
        const int rigth_count = nodeStat.count - count;
        const double right_sum = nodeStat.sum_grad - sum_grad;
        return right_sum * right_sum/rigth_count + sum_grad * sum_grad/count;
      }

      void Clean(){
        count = 0;
        sum_grad = 0.0;
      }
    };

    struct RegTree{
      static std::vector<int> InitLeft(unsigned int depth){
        std::vector<int> tmp(1 << depth);
        for(int i = 0; i < (1 << depth); ++i){
            tmp[i] = Node::Left(i);
          }
       return tmp;
      }

      static std::vector<int> InitRight(unsigned int depth){
        std::vector<int> tmp(1 << depth);
        for(int i = 0; i < (1 << depth); ++i){
            tmp[i] = Node::Right(i);
          }
        return tmp;
      }

      std::vector<Node> nodes;
      const unsigned int depth;
      const unsigned int offset;
      std::vector<float> leaf_level;
      const std::vector<int> _node_lookup [2];

      RegTree(unsigned int depth) : depth(depth), offset((1 << (depth - 1)) - 1),
        _node_lookup{ RegTree::InitRight(depth), RegTree::InitLeft(depth) }{
        unsigned int nodes_num = (1 << depth) - 1;
        nodes.reserve(nodes_num);

        for(size_t i = 0; i < nodes_num; ++i){
            nodes.push_back(Node(i));
          }
        leaf_level.resize(1 << (depth-1));
      }

      inline int ChildNode(const unsigned int parent, const bool isLeft) const {
        return _node_lookup[isLeft][parent];
      }

      void Predict(const arboretum::io::DataMatrix *data, std::vector<float> &out) const {
        for(size_t i = 0; i < data->rows; ++i){
            unsigned int node_id = 0;
            Node current_node = nodes[node_id];
            for(size_t j = 1, len = depth; j < len; ++j){
                current_node = nodes[node_id];
                node_id = ChildNode(node_id, data->data[current_node.fid][i] <= current_node.threshold);
              }
            out[i] += leaf_level[node_id - offset];
          }
      }

      void Predict(const arboretum::io::DataMatrix *data, const thrust::host_vector<size_t> &row2Node, std::vector<float> &out) const {
        #pragma omp parallel for
        for(size_t i = 0; i < data->rows; ++i){
            out[i] += leaf_level[row2Node[i]];
          }
      }
    };


    class GardenBuilderBase {
    public:
      virtual void InitGrowingTree() = 0;
      virtual void InitTreeLevel(const int level) = 0;
      virtual void GrowTree(RegTree *tree, const io::DataMatrix *data, const thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator< float > > &grad) = 0;
      virtual void PredictByGrownTree(RegTree *tree, const io::DataMatrix *data, std::vector<float> &out) = 0;
    };

    class Garden {
    public:
      Garden(const TreeParam& param);
      const TreeParam param;
      void GrowTree(io::DataMatrix* data, float* grad);
      void Predict(const arboretum::io::DataMatrix *data, std::vector<float> &out);
    private:
      bool _init;
      GardenBuilderBase* _builder;
      std::vector<RegTree*> _trees;
      void SetInitial(const arboretum::io::DataMatrix *data, std::vector<float> &out);
    };

  }
}

#endif // BOOSTER_H
