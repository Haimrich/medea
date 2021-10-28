#include "negotiator.hpp"

#include <boost/filesystem.hpp>
#include "yaml-cpp/yaml.h"

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"
#include "mapping/parser.hpp"

#include "common.hpp"
#include "accelergy.hpp"
#include "mapping-parser-fix.hpp"

namespace fs = boost::filesystem;

namespace medea
{

  MedeaNegotiator::MedeaNegotiator(config::CompoundConfig *config, std::string in_dir, std::vector<unsigned> lookup, std::string out_dir, Accelergy &accelergy) : config_(config),
                                                                                                                                                                  layer_workload_lookup_(lookup),
                                                                                                                                                                  out_dir_(out_dir),
                                                                                                                                                                  accelergy_(accelergy)
  {
    auto rootNode = config->getRoot();
    
    // Architecture configuration.
    auto arch_config = rootNode.lookup("architecture");
    default_arch_ = model::Engine::ParseSpecs(arch_config);
    std::cout << "General architecture configuration complete." << std::endl;

    Accelergy::RT reference_tables = accelergy_.GetReferenceTables("medea_negotiator");
    default_arch_.topology.ParseAccelergyERT(reference_tables.energy);
    default_arch_.topology.ParseAccelergyART(reference_tables.area);
    std::cout << "Accelergy reference tables loaded." << std::endl;

    // For each workload load workload and pareto mappings
    fs::path input_dir(in_dir);
    std::vector<fs::path> workloads_files_paths;
    if ( fs::is_directory(input_dir) )
    {
      for (const auto &p : fs::directory_iterator(input_dir)) {
        fs::path info_file = fs::path(p) / "medea.workload.yaml";
        if ( fs::is_regular_file(info_file) )
          workloads_files_paths.push_back(info_file);
      }
        
      sort(workloads_files_paths.begin(), workloads_files_paths.end());

      if (workloads_files_paths.size())
      {
        std::cout << "Found " << workloads_files_paths.size() << " workloads mapping data files in " << in_dir << std::endl;
      }
      else
      {
        std::cerr << "Error: No input data found in " << in_dir << std::endl;
        exit(1);
      }
    }
    else
    {
      std::cerr << "Error: " << in_dir << " : No such directory." << std::endl;
      exit(1);
    }

    for (auto& workload_file : workloads_files_paths) {

      YAML::Node workload_yaml = YAML::LoadFile(workload_file.string());
      problem::Workload workload;
      std::cout << "qui" << std::endl;
      problem::ParseWorkload(config::CompoundConfigNode(nullptr, workload_yaml, config), workload);
      std::cout << "qua" << std::endl;
      workloads_.push_back(workload);

      fs::path workload_pareto_folder = workload_file.parent_path() / "pareto";
      std::vector<fs::path> pareto_mapping_files;
      std::vector<MedeaMapping> mappings;

      for (const auto& candidate_mapping_file : fs::directory_iterator(workload_pareto_folder)) {
        if ( fs::is_regular_file(candidate_mapping_file) && candidate_mapping_file.path().extension() == ".yaml" ) {
          std::cout << candidate_mapping_file << std::endl;
          //YAML::Node data_yaml = YAML::LoadFile( candidate_mapping_file.path().string() );
          //MedeaMapping mapping(data_yaml, default_arch_, workload, config);
          config::CompoundConfig pareto_config(candidate_mapping_file.path().c_str());
          MedeaMapping mapping(pareto_config, default_arch_, workload);
          mappings.push_back(mapping);
        }
      }
      workload_mappings_.push_back(mappings);

    }
  }


  unsigned MedeaNegotiator::Run()
  {
    return 0;
  }


  MedeaNegotiator::~MedeaNegotiator()
  {
  }

  
  MedeaNegotiator::MedeaMapping::MedeaMapping(config::CompoundConfig &config, model::Engine::Specs &arch_specs, problem::Workload &workload) {
    
    auto root = config.getRoot();
    
    arch = MinimalArchSpecs(root.lookup("arch").getYNode());

    mapping = mapping::ParseAndConstructFixed(root.lookup("mapping"), arch_specs, workload);

    auto stats = root.lookup("stats");
    stats.lookupValue("energy", energy);
    stats.lookupValue("cycles", cycles);
    stats.lookupValue("area", area);
  }
  

}