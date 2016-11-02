#ifndef BOOSTER_H
#define BOOSTER_H

#include <stdio.h>
#include <cmath>
#include <limits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "param.h"
#include "../io/io.h"
#include "cuda_helpers.h"

namespace arboretum {
  namespace core {
    using namespace arboretum;

    template<class grad_type>
    struct Split {
      float split_value;
      int fid;
      double gain;
      grad_type sum_grad;
      unsigned int count;
      Split(){
        Clean();
      }
      void Clean(){
        fid = -1;
        gain = 0.0;
        init(sum_grad);
        count = 0;
        split_value = std::numeric_limits<float>::infinity();
      }
      float LeafWeight() const {
        return Split<grad_type>::LeafWeight(sum_grad, count);
      }
      template<class node_stat>
      float LeafWeight(node_stat& parent) const {
        return Split<grad_type>::LeafWeight(parent.sum_grad - sum_grad, parent.count - count);
      }

    private:
      static inline float LeafWeight(float s, unsigned int c){
        return s/c;
      }
      static inline float LeafWeight(float2 s, unsigned int c){
        return s.x/s.y;
      }
    };

    template<class grad_type>
    struct NodeStat {
      unsigned int count;
      grad_type sum_grad;
      double gain;
      NodeStat(){
        Clean();
      }

      void Clean(){
        count = 0;
        gain = 0.0;
        init(sum_grad);
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
      std::vector<int> _node_lookup [2];

      RegTree(unsigned int depth)
        : depth(depth), offset((1 << (depth - 1)) - 1)
      {
        unsigned int nodes_num = (1 << depth) - 1;
        nodes.reserve(nodes_num);
        _node_lookup[0] = RegTree::InitRight(depth);
        _node_lookup[1] = RegTree::InitLeft(depth);


        for(size_t i = 0; i < nodes_num; ++i){
            nodes.push_back(Node(i));
          }
        leaf_level.resize(1 << (depth-1));
      }

      inline int ChildNode(const unsigned int parent, const bool isLeft) const {
        return _node_lookup[isLeft][parent];
      }

      void Predict(const arboretum::io::DataMatrix *data, std::vector<float> &out) const {
        #pragma omp parallel for simd
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
        #pragma omp parallel for simd
        for(size_t i = 0; i < data->rows; ++i){
            out[i] += leaf_level[row2Node[i]];
          }
      }
    };

    class ApproximatedObjectiveBase {
    public:
      ApproximatedObjectiveBase(io::DataMatrix* data) : data(data) {}
      virtual ~ApproximatedObjectiveBase(){}
      const io::DataMatrix* data;
      virtual void UpdateGrad() = 0;
      virtual float IntoInternal(float v) {
        return v;
      }
      virtual inline void FromInternal(std::vector<float>& in, std::vector<float>& out) {
        in = out;
      }
    };

    template<class grad_type>
    class ApproximatedObjective : public ApproximatedObjectiveBase {
    public:
      ApproximatedObjective(io::DataMatrix* data, float initial_y) : ApproximatedObjectiveBase(data){
        grad.resize(data->rows);
      }
      thrust::host_vector<float2, thrust::cuda::experimental::pinned_allocator< grad_type > > grad;
    };

    class RegressionObjective : public ApproximatedObjective<float> {
    public:
      RegressionObjective(io::DataMatrix* data, float initial_y)
        : ApproximatedObjective<float>(data, initial_y){
        data->y_internal.resize(data->rows, IntoInternal(initial_y));
      }
      virtual void UpdateGrad() override {
        #pragma omp parallel for simd
        for(size_t i = 0; i < data->rows; ++i){
            grad[i] = data->y_hat[i] - data->y_internal[i];
          }
      }
    };

    class LogisticRegressionObjective : public ApproximatedObjective<float2> {
    public:
      LogisticRegressionObjective(io::DataMatrix* data, float initial_y)
        : ApproximatedObjective<float2>(data, initial_y){
        data->y_internal.resize(data->rows, IntoInternal(initial_y));
      }
      virtual void UpdateGrad() override {
        #pragma omp parallel for simd
        for(size_t i = 0; i < data->rows; ++i){
            const float sigmoid = Sigmoid(data->y_internal[i]);
            grad[i].x = data->y_hat[i] - sigmoid;
            grad[i].y = sigmoid * (1.0f - sigmoid);
          }
      }
      virtual inline float IntoInternal(float v) override {
        return std::log(v/(1-v));
      }
      virtual inline void FromInternal(std::vector<float>& in, std::vector<float>& out) override {
        #pragma omp parallel for simd
        for(size_t i = 0; i < out.size(); ++i){
            in[i] = Sigmoid(out[i]);
          }
      }
    private:
      inline float Sigmoid(float x){
        return 1.0/(1.0 + std::exp(-x));
      }
    };


    class GardenBuilderBase {
    public:
      virtual ~GardenBuilderBase(){}
      virtual size_t MemoryRequirementsPerRecord() = 0;
      virtual void InitGrowingTree(const size_t columns) = 0;
      virtual void InitTreeLevel(const int level, const size_t columns) = 0;
      virtual void GrowTree(RegTree *tree, const io::DataMatrix *data) = 0;
      virtual void PredictByGrownTree(RegTree *tree, io::DataMatrix *data, std::vector<float> &out) const = 0;
    };

    class Garden {
    public:
      Garden(const TreeParam& param, const Verbose& verbose, const InternalConfiguration& cfg);
      ~Garden(){
        if(_builder)
          delete _builder;
        if(_objective)
          delete _objective;
      }

      const TreeParam param;
      const Verbose verbose;
      const InternalConfiguration cfg;
      void GrowTree(io::DataMatrix* data, float *grad);
      void Predict(const arboretum::io::DataMatrix *data, std::vector<float> &out) const;
      void UpdateByLastTree(arboretum::io::DataMatrix *data);
      void GetY(arboretum::io::DataMatrix *data, std::vector<float> &out) const;
    private:
      bool _init;
      GardenBuilderBase* _builder;
      ApproximatedObjectiveBase* _objective;
      std::vector<RegTree*> _trees;
    };

  }
}

#endif // BOOSTER_H
