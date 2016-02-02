#ifndef BOOSTER_H
#define BOOSTER_H

#include <cmath>
#include <limits>
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
      std::vector<Node> nodes;
      unsigned int depth;
      unsigned int offset;
      std::vector<float> leaf_level;

      RegTree(unsigned int depth) : depth(depth){
        unsigned int nodes_num = (1 << depth) - 1;
        offset = (1 << (depth - 1)) - 1;
        nodes.reserve(nodes_num);
        for(size_t i = 0; i < nodes_num; ++i){
            nodes.push_back(Node(i));
          }
        leaf_level.resize(1 << (depth-1));
      }

      void Predict(arboretum::io::DataMatrix *data, std::vector<float> &out){
        for(int i = 0; i < data->rows; ++i){
            unsigned int node_id = 0;
            Node current_node = nodes[node_id];
            while(node_id < offset){
                current_node = nodes[node_id];
                if(data->data[current_node.fid][i] <= current_node.threshold)
                  node_id = Node::Left(node_id);
                else
                  node_id = Node::Right(node_id);
              }
            out[i] += leaf_level[node_id - offset];
          }
      }
    };


    class GardenBuilderBase {
    public:
      virtual void InitGrowingTree() = 0;
      virtual void InitTreeLevel(const int level) = 0;
      virtual void GrowTree(RegTree *tree, const io::DataMatrix *data, std::vector<float> &grad) = 0;
    };

    class Garden {
    public:
      Garden(const TreeParam& param);
      const TreeParam param;
      void GrowTree(io::DataMatrix* data, float* grad);
      void Predict(arboretum::io::DataMatrix *data, std::vector<float> &out);
    private:
      bool _init;
      GardenBuilderBase* _builder;
      std::vector<RegTree*> _trees;
    };

  }
}

#endif // BOOSTER_H
