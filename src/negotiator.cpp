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

  MedeaNegotiator::MedeaNegotiator(config::CompoundConfig *config, std::string in_dir, std::vector<size_t> lookup, std::string out_dir, Accelergy &accelergy) : config_(config),
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

    workload_mappings_.reserve(workloads_files_paths.size());
    for (auto& workload_file : workloads_files_paths) {

      YAML::Node workload_yaml = YAML::LoadFile(workload_file.string());
      problem::Workload workload;
      problem::ParseWorkload(config::CompoundConfigNode(nullptr, workload_yaml, config), workload);
      workloads_.push_back(workload);

      fs::path workload_pareto_folder = workload_file.parent_path() / "pareto";
      std::vector<fs::path> pareto_mapping_files;
      
      for (const auto& candidate_mapping_file : fs::directory_iterator(workload_pareto_folder))
        if ( fs::is_regular_file(candidate_mapping_file) && candidate_mapping_file.path().extension() == ".yaml" )
          pareto_mapping_files.emplace_back(candidate_mapping_file);

      sort(pareto_mapping_files.begin(), pareto_mapping_files.end());

      std::vector<MedeaMapping> mappings;
      mappings.reserve(pareto_mapping_files.size());
      for (size_t i = 0; i < pareto_mapping_files.size(); i++) {
        config::CompoundConfig pareto_config(pareto_mapping_files[i].c_str());
        mappings.emplace_back(i, pareto_config, default_arch_, workload);
      }
      workload_mappings_.push_back(mappings);
    }
    
    auto medea = rootNode.lookup("medea");

    num_threads_ = std::thread::hardware_concurrency();
    if (medea.lookupValue("num-threads", num_threads_))
      std::cout << "Using threads = " << num_threads_ << std::endl;
    else
      std::cout << "Using all available hardware threads = " << num_threads_ << std::endl;
      
    thread_orchestrator_ = new Orchestrator(num_threads_);
  }


  unsigned MedeaNegotiator::Run()
  {

    return 0;
  }


  MedeaNegotiator::~MedeaNegotiator()
  {
    delete thread_orchestrator_;
  }


  MedeaMapping::MedeaMapping(unsigned id, config::CompoundConfig &config, model::Engine::Specs &arch_specs, problem::Workload &workload) : id(id)
  {

    auto root = config.getRoot();

    arch = MinimalArchSpecs(root.lookup("arch").getYNode());

    mapping = mapping::ParseAndConstructFixed(root.lookup("mapping"), arch_specs, workload);

    auto stats = root.lookup("stats");
    stats.lookupValue("energy", energy);
    stats.lookupValue("cycles", cycles);
    stats.lookupValue("area", area);
  }


  NegotiatorIndividual::NegotiatorIndividual(problem::Workload &workload, const std::vector<std::vector<MedeaMapping>> &mappings, 
                                             const std::vector<size_t> &lookup, model::Engine::Specs arch_specs, Accelergy &accelergy) : 
                                             workload_(workload), default_arch_specs_(arch_specs), accelergy_(accelergy)
  {
    num_layers_ = lookup.size();
    mapping_set.reserve(num_layers_);
    need_evaluation_.resize(num_layers_, true);

    std::uniform_int_distribution<size_t> uni_dist(0, mappings[0].size());
    auto m = mappings[0][uni_dist(*rng_)];
    mapping_set.push_back(m);
    negotiated_arch = m.arch;

    for (size_t l = 1; l < num_layers_; l++)
    {
      size_t workload_id = lookup[l];
      uni_dist = std::uniform_int_distribution<size_t>(0, mappings[workload_id].size());
      m = mappings[workload_id][uni_dist(*rng_)];
      mapping_set.push_back(m);
      negotiated_arch = negotiated_arch && m.arch;
    }

    UpdateEngineArchitecture();
    Evaluate();
  }


  bool NegotiatorIndividual::NegotiateArchitecture() 
  {
    auto old_negotiated_arch = negotiated_arch;
    
    negotiated_arch = mapping_set[0].arch;
    for (size_t i = 0; i < mapping_set.size(); i++)
      negotiated_arch = negotiated_arch && mapping_set[i].arch;

    return old_negotiated_arch != negotiated_arch;
  }

  void NegotiatorIndividual::UpdateEngineArchitecture()
  {
    auto new_specs = model::Topology::Specs(default_arch_specs_.topology);

    auto minimal_arithmetic = negotiated_arch.GetLevel(0);
    auto arithmetic = new_specs.GetArithmeticLevel();
    arithmetic->meshX = minimal_arithmetic.mesh_x;
    arithmetic->meshY = minimal_arithmetic.mesh_y;
    arithmetic->instances = minimal_arithmetic.mesh_x * minimal_arithmetic.mesh_y;

    std::map<std::string, uint64_t> updates;

    for (unsigned i = 1; i < default_arch_specs_.topology.NumLevels();  i++)
    {
      auto buffer = new_specs.GetStorageLevel(i - 1);
      if (!buffer->size.IsSpecified()) continue;

      auto minimal_buffer = negotiated_arch.GetLevel(i);
      buffer->meshX = minimal_buffer.mesh_x;
      buffer->meshY = minimal_buffer.mesh_y;
      buffer->instances = minimal_buffer.mesh_x * minimal_buffer.mesh_y;
      buffer->size = minimal_buffer.size;
      buffer->effective_size = static_cast<uint64_t>(std::floor(minimal_buffer.size / buffer->multiple_buffering.Get()));
    
      updates[buffer->name.Get()] = buffer->size.Get() / buffer->block_size.Get();
    }

    //std::string out_prefix = "medea." + std::to_string(thread_id_) + "_tmp";
    std::string out_prefix = "medea.tmp";
    Accelergy::RT rt = accelergy_.GetReferenceTables(updates, out_prefix);

    model::Engine::Specs new_engine_specs;
    new_engine_specs.topology = new_specs;
    new_engine_specs.topology.ParseAccelergyART(rt.area);
    new_engine_specs.topology.ParseAccelergyERT(rt.energy);
    negotiated_arch_specs_ = new_engine_specs;

    for (auto& engine : set_engines_)
      engine.Spec(negotiated_arch_specs_);
  }


  void NegotiatorIndividual::Evaluate(bool evaluate_all) {
    for (size_t i = 0; i < num_layers_; i++)
      if (need_evaluation_[i] || evaluate_all) 
      {
        set_engines_[i].Evaluate(mapping_set[i].mapping, workload_);
        assert(set_engines_[i].IsEvaluated());
        need_evaluation_[i] = false;
      }

    energy = cycles = 0;
    for (size_t i = 0; i < num_layers_; i++) {
      energy += set_engines_[i].Energy();
      cycles += set_engines_[i].Cycles();
    }
    area = set_engines_[0].Area();
  }
}