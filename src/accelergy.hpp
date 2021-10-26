#ifndef MEDEA_ACCELERGY_H_
#define MEDEA_ACCELERGY_H_

#include <string>
#include <vector>
#include <map>
#include <mutex>

#include "yaml-cpp/yaml.h"

#include "compound-config/compound-config.hpp"

namespace medea
{
  class Accelergy
  {
  protected:
    struct Entry {
      std::string energy;
      std::string area;
    };

    std::mutex cache_mutex_;
    std::map<std::string, std::map<size_t, Entry>> cache_;

    Entry invariant_nodes_;
    std::once_flag invariant_nodes_inizialized_;

    std::string fast_accelergy_path_;
    config::CompoundConfig *config_;

  public:
  
    struct RT {
      config::CompoundConfigNode energy;
      config::CompoundConfigNode area;
    };


    Accelergy(config::CompoundConfig *config)
    {
      config_ = config;
      fast_accelergy_path_ = BUILD_BASE_DIR;
      fast_accelergy_path_ += "/../../scripts/fast_accelergy.py";
      config->getRoot().lookup("medea").lookupValue("fast-accelergy-path", fast_accelergy_path_);
    }


    RT GetReferenceTables(std::map<std::string, uint64_t> &updates, std::string out_prefix)
    {
      YAML::Node ert_node = YAML::Node();
      YAML::Node art_node = YAML::Node();

      if ( ! FindInCache(updates, ert_node, art_node) ) {

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

        YAML::Node acc_out = YAML::Load(fast_accelergy_out);
        ert_node = acc_out["ERT"];
        art_node = acc_out["ART"];

        std::call_once ( invariant_nodes_inizialized_, [&]{ 
          InizializeInvariantNodes(ert_node, art_node, updates); 
        } );          

        UpdateCache(ert_node, art_node, updates);
      }
      
      auto art_config = config::CompoundConfigNode(nullptr, art_node, config_);
      auto ert_config = config::CompoundConfigNode(nullptr, ert_node, config_);

      return (RT) {.energy = ert_config, .area = art_config};
    }
  

  protected:

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


    void InizializeInvariantNodes(YAML::Node &ert_node, YAML::Node &art_node, std::map<std::string, uint64_t> &updates) {
      YAML::Emitter ytmpe, ytmpa;
      ytmpe << YAML::BeginSeq;
      ytmpa << YAML::BeginSeq;

      for (std::size_t i = 0; i < art_node["tables"].size(); i++) {
        auto name = art_node["tables"][i]["name"].as<std::string>();
        name = name.substr(name.find_last_of(".") + 1);

        if (updates.find(name) == updates.end()) {
          ytmpe << ert_node["tables"][i];
          ytmpa << art_node["tables"][i];
        }
      }

      ytmpe << YAML::EndSeq;
      ytmpa << YAML::EndSeq;

      invariant_nodes_.energy = ytmpe.c_str();
      invariant_nodes_.area = ytmpa.c_str();
    }


    bool FindInCache(std::map<std::string, uint64_t> &updates, YAML::Node &ert_node, YAML::Node &art_node) {
      std::string art, ert;
      art = ert = "version: 0.3\ntables:\n";

      try {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (auto it = updates.begin(); it != updates.end(); it++)
        {
          Entry &entry = cache_.at(it->first).at(it->second);  
          art += entry.area + "\n";
          ert += entry.energy + "\n";
        }
      }
      catch (const std::out_of_range& oor) {
        return false;
      }

      art_node = YAML::Load(art + invariant_nodes_.area);
      ert_node = YAML::Load(ert + invariant_nodes_.energy);

      //std::cout << "[INFO] Accelergy Cache Hit! " << std::endl;
      return true;
    }


    void UpdateCache(YAML::Node &ert_node, YAML::Node &art_node, std::map<std::string, uint64_t> &updates) {
      std::unique_lock<std::mutex> lock(cache_mutex_);

      for (std::size_t i = 0; i < art_node["tables"].size(); i++) {
        auto name = art_node["tables"][i]["name"].as<std::string>();
        name = name.substr(name.find_last_of(".") + 1);
        

        auto size = updates.find(name);
        if (size != updates.end()) {
          YAML::Emitter ytmpe, ytmpa;

          ytmpe << YAML::BeginSeq << ert_node["tables"][i] << YAML::EndSeq;
          ytmpa << YAML::BeginSeq << art_node["tables"][i] << YAML::EndSeq;

          cache_[name][size->second] = (Entry) {
            .energy = ytmpe.c_str(),
            .area = ytmpa.c_str()
          };
        }
        
      }
    }

  };

}

#endif // MEDEA_ACCELERGY_H_