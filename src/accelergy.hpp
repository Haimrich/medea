#ifndef MEDEA_ACCELERGY_H_
#define MEDEA_ACCELERGY_H_

#include <vector>
#include <map>

#include "compound-config/compound-config.hpp"
#include <yaml-cpp/yaml.h>

namespace medea
{
  class Accelergy
  {
  protected:
    struct Entry {
      YAML::Node energy;
      YAML::Node area;
    };

    std::mutex cache_mutex_;
    std::map<std::string, std::map<size_t, Entry>> cache_;
    std::vector<Entry> invariant_nodes_;
    bool invariant_nodes_inizialized_;

    std::string fast_accelergy_path_;
    config::CompoundConfig *config_;

    std::string Run(const char *cmd)
    {
      std::string result = "";
      char buffer[128];
      FILE *pipe = popen(cmd, "r");
      if (!pipe)
      {
        std::cout << "popen(" << cmd << ") failed" << std::endl;
        exit(0);
      }

      try
      {
        while (fgets(buffer, 128, pipe) != nullptr)
          result += buffer;
      }
      catch (...)
      {
        pclose(pipe);
      }
      pclose(pipe);
      return result;
    }

  public:
    struct RT {
      config::CompoundConfigNode energy;
      config::CompoundConfigNode area;
    };

    Accelergy(config::CompoundConfig *config) :
      invariant_nodes_inizialized_(false)
    {
      config_ = config;
      fast_accelergy_path_ = BUILD_BASE_DIR;
      fast_accelergy_path_ += "/../../scripts/fast_accelergy.py";
      config->getRoot().lookup("medea").lookupValue("fast-accelergy-path", fast_accelergy_path_);
    }


    RT GetReferenceTables(std::map<std::string, uint64_t> &updates, std::string out_prefix)
    {
      YAML::Node ynode = YAML::Node();
      if (!FindInCache(updates, ynode)) {

        std::vector<std::string> input_files = config_->inFiles;

        std::string cmd = "python " + fast_accelergy_path_;

        for (auto input_file : input_files)
          cmd += " " + input_file;

        cmd += " --oprefix " + out_prefix + ". --updates ";

        for (std::map<std::string, uint64_t>::iterator it = updates.begin(); it != updates.end(); it++)
          cmd += " " + it->first + "," + std::to_string(it->second);

        std::string fast_accelergy_out = Run(cmd.c_str());
        if (fast_accelergy_out.length() == 0)
        {
          std::cout << "Failed to run Accelergy. Did you install Accelergy or specify ACCELERGYPATH correctly? Or check accelergy.log to see what went wrong" << std::endl;
          exit(0);
        }
        ynode = YAML::Load(fast_accelergy_out);
        if (!invariant_nodes_inizialized_) InizializeInvariantNodes(ynode, updates);

        UpdateCache(ynode, updates);
      }
      
      auto art_node = config::CompoundConfigNode(nullptr, ynode["ART"], config_);
      auto ert_node = config::CompoundConfigNode(nullptr, ynode["ERT"], config_);

      return (RT) {.energy = ert_node, .area = art_node};
    }


    void InizializeInvariantNodes(YAML::Node &acc_out, std::map<std::string, uint64_t> &updates) {
      std::unique_lock<std::mutex> lock(cache_mutex_);
      if (invariant_nodes_inizialized_) return;

      for (std::size_t i = 0; i < acc_out["ART"]["tables"].size(); i++) {
        auto name = acc_out["ART"]["tables"][i]["name"].as<std::string>();
        name = name.substr(name.find_last_of(".") + 1);

        if (updates.find(name) == updates.end())
          invariant_nodes_.push_back((Entry){.energy = acc_out["ERT"]["tables"][i], .area = acc_out["ART"]["tables"][i]});
      }
      invariant_nodes_inizialized_ = true;
    }


    bool FindInCache(std::map<std::string, uint64_t> &updates, YAML::Node &root) {
      YAML::Node art, ert;

      cache_mutex_.lock();
      try {
        for (auto it = updates.begin(); it != updates.end(); it++)
        {
          Entry &entry = cache_.at(it->first).at(it->second);
          art["tables"].push_back(entry.area);
          ert["tables"].push_back(entry.energy);
        }
      }
      catch (const std::out_of_range& oor) {
        cache_mutex_.unlock();
        return false;
      }
      cache_mutex_.unlock();

      for (auto &e : invariant_nodes_) {
        art["tables"].push_back(e.area);
        ert["tables"].push_back(e.energy);
      }

      art["version"] = 0.3;
      ert["version"] = 0.3;

      root["ART"] = art;
      root["ERT"] = ert;

      //std::cout << "[INFO] Accelergy Cache Hit! " << std::endl;
      return true;
    }


    void UpdateCache(YAML::Node &acc_out, std::map<std::string, uint64_t> &updates) {
      std::unique_lock<std::mutex> lock(cache_mutex_);

      for (std::size_t i = 0; i < acc_out["ART"]["tables"].size(); i++) {
        auto name = acc_out["ART"]["tables"][i]["name"].as<std::string>();
        name = name.substr(name.find_last_of(".") + 1);

        auto size = updates.find(name);
        if (size != updates.end())
          cache_[name][size->second] = (Entry){.energy = acc_out["ERT"]["tables"][i], .area = acc_out["ART"]["tables"][i]};
      }
    }

  };

}

/* maybe useful for caching

YAML::Node node = root_node;
while (node["subtree"]) {
node = node["subtree"][0];
auto local_node = node["local"];

if (local_node) 
    for (auto&& buffer : local_node) 
    if (buffer["name"]) {
        auto buffer_name = buffer["name"].as<std::string>();
        if (buffer_name.compare(0, storage_name.size(), storage_name) == 0 && buffer["attributes"])
        if (buffer["attributes"]["depth"])
            buffer["attributes"]["depth"] = 
    }
    
}
*/

#endif // MEDEA_ACCELERGY_H_