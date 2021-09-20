#include <vector>
#include <map>

#include "compound-config/compound-config.hpp"
#include <yaml-cpp/yaml.h>

namespace medea
{

    std::string exec(const char *cmd)
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

    config::CompoundConfigNode getARTfromAccelergy(config::CompoundConfig* config, std::map<std::string, uint64_t> &updates, std::string out_prefix)
    {
        std::vector<std::string> input_files = config->inFiles;
        std::string fast_accelergy_path = "/home/enrico/lambda/medea/scripts/fast_accelergy_ART.py";

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

        std::string fast_accelergy_out = exec(cmd.c_str());
        if (fast_accelergy_out.length() == 0)
        {
            std::cout << "Failed to run Accelergy. Did you install Accelergy or specify ACCELERGYPATH correctly? Or check accelergy.log to see what went wrong" << std::endl;
            exit(0);
        }
        YAML::Node ynode = YAML::Load(fast_accelergy_out);
        config::CompoundConfigNode node(nullptr, ynode["ART"], config);
        return node;
    }
}