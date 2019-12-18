#ifndef SRC_CORE_JSON_CONVERSION_H
#define SRC_CORE_JSON_CONVERSION_H

#include <string>
#include "json.hpp"
#include "param.h"
#include "reg_tree.h"

namespace arboretum {
namespace core {
using nlohmann::json;

inline void to_json(json &j, const Node &node) {
  const bool leaf = node.id >= unsigned(1 << (node.depth - 1)) - 1;
  j = json{{"id", node.id}, {"leaf", leaf}};
  if (!leaf) {
    j["left"] = 2 * node.id + 1;
    j["right"] = 2 * node.id + 2;
    j["threshold"] = node.threshold;
    j["fid"] = node.fid;
  } else {
    j["idx"] = node.id - (1 << (node.depth - 1)) + 1;
  }
}

inline void from_json(const json &j, Node &node) {
  j.at("id").get_to(node.id);
  if (j.find("threshold") != j.end()) {
    if (j.at("threshold").is_null())
      node.threshold = INFINITY;
    else
      j.at("threshold").get_to(node.threshold);
  }
  if (j.find("fid") != j.end()) j.at("fid").get_to(node.fid);
}

inline void to_json(json &j, const DecisionTree &tree) {
  j = json{
    {"nodes", tree.nodes}, {"weights", tree.weights}, {"depth", tree.depth}};
}

inline void from_json(const json &j, DecisionTree &tree) {
  j.at("nodes").get_to(tree.nodes);
  j.at("weights").get_to(tree.weights);
  j.at("depth").get_to(tree.depth);
  for (unsigned i = 0; i < tree.nodes.size(); ++i) {
    tree.nodes[i].depth = tree.depth;
  }
}

inline void to_json(json &j, const Verbose &cfg) {
  j = json{{"gpu", cfg.gpu}, {"booster", cfg.booster}, {"data", cfg.data}};
}

inline void from_json(const json &j, Verbose &cfg) {
  if (j.find("gpu") != j.end()) j.at("gpu").get_to(cfg.gpu);

  if (j.find("booster") != j.end()) j.at("booster").get_to(cfg.booster);

  if (j.find("data") != j.end()) j.at("data").get_to(cfg.data);
}

inline void to_json(json &j, const InternalConfiguration &cfg) {
  j = json{{"double_precision", cfg.double_precision},
           {"compute_overlap", cfg.overlap},
           {"seed", cfg.seed},
           {"use_hist_subtraction_trick", cfg.use_hist_subtraction_trick},
           {"upload_features", cfg.upload_features},
           {"hist_size", cfg.hist_size}};
}

inline void from_json(const json &j, InternalConfiguration &cfg) {
  if (j.find("double_precision") != j.end())
    j.at("double_precision").get_to(cfg.double_precision);

  if (j.find("compute_overlap") != j.end())
    j.at("compute_overlap").get_to(cfg.overlap);

  if (j.find("seed") != j.end()) j.at("seed").get_to(cfg.seed);

  if (j.find("use_hist_subtraction_trick") != j.end())
    j.at("use_hist_subtraction_trick").get_to(cfg.use_hist_subtraction_trick);

  if (j.find("upload_features") != j.end())
    j.at("upload_features").get_to(cfg.upload_features);

  if (j.find("hist_size") != j.end()) j.at("hist_size").get_to(cfg.hist_size);
}

inline void to_json(json &j, const TreeParam &cfg) {
  j = json{{"max_depth", cfg.depth},
           {"min_child_weight", cfg.min_child_weight},
           {"min_leaf_size", cfg.min_leaf_size},
           {"colsample_bytree", cfg.colsample_bytree},
           {"colsample_bylevel", cfg.colsample_bylevel},
           {"gamma_absolute", cfg.gamma_absolute},
           {"gamma_relative", cfg.gamma_relative},
           {"lambda", cfg.lambda},
           {"alpha", cfg.alpha},
           {"initial_y", cfg.initial_y},
           {"eta", cfg.eta},
           {"max_leaf_weight", cfg.max_leaf_weight},
           {"scale_pos_weight", cfg.scale_pos_weight},
           {"labels_count", cfg.labels_count}};
}

inline void from_json(const json &j, TreeParam &cfg) {
  if (j.find("max_depth") != j.end()) j.at("max_depth").get_to(cfg.depth);

  if (j.find("min_child_weight") != j.end())
    j.at("min_child_weight").get_to(cfg.min_child_weight);

  if (j.find("min_leaf_size") != j.end())
    j.at("min_leaf_size").get_to(cfg.min_leaf_size);

  if (j.find("colsample_bytree") != j.end())
    j.at("colsample_bytree").get_to(cfg.colsample_bytree);

  if (j.find("colsample_bylevel") != j.end())
    j.at("colsample_bylevel").get_to(cfg.colsample_bylevel);

  if (j.find("gamma_absolute") != j.end())
    j.at("gamma_absolute").get_to(cfg.gamma_absolute);

  if (j.find("gamma_relative") != j.end())
    j.at("gamma_relative").get_to(cfg.gamma_relative);

  if (j.find("lambda") != j.end()) j.at("lambda").get_to(cfg.lambda);

  if (j.find("alpha") != j.end()) j.at("alpha").get_to(cfg.alpha);

  if (j.find("initial_y") != j.end()) j.at("initial_y").get_to(cfg.initial_y);

  if (j.find("eta") != j.end()) j.at("eta").get_to(cfg.eta);

  if (j.find("max_leaf_weight") != j.end())
    j.at("max_leaf_weight").get_to(cfg.max_leaf_weight);

  if (j.find("scale_pos_weight") != j.end())
    j.at("scale_pos_weight").get_to(cfg.scale_pos_weight);

  if (j.find("labels_count") != j.end())
    j.at("labels_count").get_to(cfg.labels_count);
}

inline void to_json(json &j, const Configuration &cfg) {
  j = json{{"method", cfg.method},
           {"objective", cfg.objective},
           {"tree", cfg.tree_param},
           {"verbose", cfg.verbose},
           {"internals", cfg.internal}};
}

inline void from_json(const json &j, Configuration &cfg) {
  if (j.find("method") != j.end()) j.at("method").get_to(cfg.method);

  if (j.find("objective") != j.end()) j.at("objective").get_to(cfg.objective);

  if (j.find("tree") != j.end()) j.at("tree").get_to(cfg.tree_param);

  if (j.find("verbose") != j.end()) j.at("verbose").get_to(cfg.verbose);

  if (j.find("internals") != j.end()) j.at("internals").get_to(cfg.internal);
}

}  // namespace core
}  // namespace arboretum

#endif