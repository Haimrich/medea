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
    config::CompoundConfigNode art_node;
    config::CompoundConfigNode ert_node;

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

  public:
    Accelergy(std::string fast_accelergy_path, config::CompoundConfig *config, std::map<std::string, uint64_t> &updates, std::string out_prefix)
    {
      std::vector<std::string> input_files = config->inFiles;

      std::string cmd = "python " + fast_accelergy_path;

      for (auto input_file : input_files)
        cmd += " " + input_file;

      cmd += " --oprefix " + out_prefix + ".";
      cmd += " --updates ";

      for (std::map<std::string, uint64_t>::iterator it = updates.begin(); it != updates.end(); it++)
      {
        cmd += " " + it->first + "," + std::to_string(it->second);
      }

      //std::cout << "execute:" << cmd << std::endl;

      std::string fast_accelergy_out = Run(cmd.c_str());
      if (fast_accelergy_out.length() == 0)
      {
        std::cout << "Failed to run Accelergy. Did you install Accelergy or specify ACCELERGYPATH correctly? Or check accelergy.log to see what went wrong" << std::endl;
        exit(0);
      }
      YAML::Node ynode = YAML::Load(fast_accelergy_out);

      art_node = config::CompoundConfigNode(nullptr, ynode["ART"], config);
      ert_node = config::CompoundConfigNode(nullptr, ynode["ERT"], config);
    }

    config::CompoundConfigNode GetART()
    {
      return art_node;
    }

    config::CompoundConfigNode GetERT()
    {
      return ert_node;
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