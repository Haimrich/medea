#include "simple-negotiator.hpp"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "compound-config/compound-config.hpp"
#include "workload/workload.hpp"
#include "model/engine.hpp"

#include "accelergy.hpp"
#include "mapping-parser-fix.hpp"
#include "individual.hpp"


namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char *argv[])
{
  using namespace std;
  using namespace medea;

  cout << R"(
   __  __ ___ ___  ___   _    
  |  \/  | __|   \| __| /_\
  | |\/| | _|| |) | _| / _ \
  |_|  |_|___|___/|___/_/ \_\
  )" << endl;
  
  try
  {

    po::options_description desc(
      "MEDEA: A Multi-objective Evolutionary Approach to DNN Hardware Mapping\n"
      "SIMPLE NEGOTIATOR. Allowed options"
    );
    desc.add_options()
      ("help,h", "Print help information.")
      ("input-files,i", po::value<vector<string>>(), 
        "Input .yaml config files. These should specify paremeters about the architecture, "
        "arch. components, datatype bypass, medea search. In map mode, if an input directory "
        "is not specified, it is possible to provide a single workload file as input."
      )
      ("output,o", po::value<string>()->default_value("."), "Output directory")
      ("mapping-dir,d", po::value<string>(), 
        "The directory containing a mapping for each workload." )
      ("workload-dir,w", po::value<string>(), 
        "The directory containing a workload files." )
      ("lookup,l", po::value<vector<size_t>>()->multitoken(), 
        "Layer workload lookup. Ex. -l 0 1 2 2 3 3 4 . This provide, for each actual network layer "
        "(ex. AlexNet layers) the index of the workload in the input directory (in alphabetic order) "
        "that matches its dimensions (filter sizes, number of channels, ecc.). This because some "
        "layers in networks share the same dimensions and can be mapped once."
      )
    ;

    po::positional_options_description p;
    p.add("input-files", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
      cout << desc << "\n";
      return 0;
    }

    if (!vm.count("input-files"))
    {
      cout << "Missing input files. \n";
      return 0;
    }

    string mapping_dir = vm.count("mapping-dir") ? vm["mapping-dir"].as<string>() : ".";
    string workload_dir = vm.count("workload-dir") ? vm["workload-dir"].as<string>() : ".";

    if (mapping_dir == workload_dir) {
      cerr << "Same workload and mapping input directory." << endl;
      exit(1);
    }

    if (!vm.count("lookup")) {
      cerr << "Missing layer workload lookup." << endl;
      exit(1);
    }
    vector<size_t> layer_workload_lookup = vm["lookup"].as<vector<size_t>>();

    vector<string> input_files = vm["input-files"].as<vector<string>>();
    string out_dir = vm["output"].as<string>();

    auto pre_config = new config::CompoundConfig(input_files);
    Accelergy accelergy(pre_config);
      
    MedeaSimpleNegotiator negotiator(pre_config, workload_dir, mapping_dir, layer_workload_lookup, out_dir, accelergy);
    negotiator.Run();
    
    cout << "[MEDEA] Negotiation completed." << endl;

    delete pre_config;
  }
  catch (exception &e)
  {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }
  catch (...)
  {
    cerr << "Exception of unknown type!" << endl;
    return 1;
  }
  
  return 0;
}

bool gTerminateEval = false;


namespace medea {
  MedeaSimpleNegotiator::MedeaSimpleNegotiator(
    config::CompoundConfig *config, 
    std::string workload_dir, 
    std::string mapping_dir,
    std::vector<size_t> lookup, 
    std::string out_dir, 
    Accelergy &accelergy
  ) : 
    config_(config),
    workload_dir_(workload_dir),
    mapping_dir_(mapping_dir),
    layer_workload_lookup_(lookup),
    out_dir_(out_dir),
    accelergy_(accelergy) 
  {
    num_layers_ = layer_workload_lookup_.size();

    auto rootNode = config->getRoot();
    
    // Architecture configuration.
    auto arch_config = rootNode.lookup("architecture");
    default_arch_ = model::Engine::ParseSpecs(arch_config);
    std::cout << "General architecture configuration complete." << std::endl;

    Accelergy::RT reference_tables = accelergy_.GetReferenceTables("medea_negotiator");
    default_arch_.topology.ParseAccelergyERT(reference_tables.energy);
    default_arch_.topology.ParseAccelergyART(reference_tables.area);
    std::cout << "Accelergy reference tables loaded." << std::endl;
  }


  void MedeaSimpleNegotiator::Run() {

    // Load workloads and mappings
    fs::path workload_path(workload_dir_);
    std::vector<fs::path> workloads_files_paths;
    if ( fs::is_directory(workload_path) )
    {
      for (const auto &p : fs::directory_iterator(workload_path)) {
        if ( fs::is_regular_file(p) && p.path().extension() == ".yaml" )
          workloads_files_paths.push_back(p);
      }
        
      sort(workloads_files_paths.begin(), workloads_files_paths.end());

    } else {
      std::cerr << "Error: No such directory " << workload_dir_ << std::endl;
      exit(1);
    }

    fs::path mapping_path(mapping_dir_);
    std::vector<fs::path> mappings_files_paths;
    if ( fs::is_directory(mapping_path) )
    {
      for (const auto &p : fs::directory_iterator(mapping_path)) {
        fs::path mapping_file = p.path() / "map_16.yaml";
        if ( fs::is_regular_file(mapping_file) )
          mappings_files_paths.push_back(mapping_file);
      }
        
      sort(mappings_files_paths.begin(), mappings_files_paths.end());

    } else {
      std::cerr << "Error: No such directory " << mapping_dir_ << std::endl;
      exit(1);
    }

    if (workloads_files_paths.size())
    {
      std::cout << "Found " << workloads_files_paths.size() << " workloads data files in " << workload_dir_ << std::endl;
    }
    else
    {
      std::cerr << "Error: No input data found in " << workload_dir_ << std::endl;
      exit(1);
    }

    if (mappings_files_paths.size())
    {
      std::cout << "Found " << mappings_files_paths.size() << " mapping data files in " << mapping_dir_ << std::endl;
    }
    else
    {
      std::cerr << "Error: No input data found in " << mapping_dir_ << std::endl;
      exit(1);
    }

    if (workloads_files_paths.size() != mappings_files_paths.size()) 
    {
      std::cerr << "Error: Different number of workloads and mappings." << std::endl;
      exit(1);
    }

    workloads_.reserve(workloads_files_paths.size());
    for (auto& workload_file : workloads_files_paths) 
    {
      problem::Workload workload;
      config::CompoundConfig workload_config(workload_file.c_str());
      problem::ParseWorkload(workload_config.getRoot().lookup("problem"), workload);

      workloads_.push_back(workload);
    }

    mappings_.reserve(mappings_files_paths.size());
    for (size_t i = 0; i < workloads_.size(); i++) 
    {
      config::CompoundConfig mapping_config(mappings_files_paths[i].c_str());
      Mapping mapping;
      mapping = mapping::ParseAndConstructFixed(mapping_config.getRoot().lookup("mapping"), default_arch_, workloads_[i]);
      mappings_.push_back(mapping);
    }

    // First evaluation for minimal archs
    num_workloads_ = workloads_.size();

    engines_.resize(num_workloads_);
    for (size_t w = 0; w < num_workloads_; w++) 
    {
      engines_[w].Spec(default_arch_);
      auto status = engines_[w].Evaluate(mappings_[w], workloads_[w]);

      if (!engines_[w].IsEvaluated())
      {
        std::cout << "Error in workload " << w << std::endl;
        std::cout << "Mapping: " << mappings_[w].PrintCompact() << std::endl;

        for (auto &s : status) std::cout << s.success << " " << s.fail_reason << std::endl;
        exit(1);
      }
    }

    // Negotiate archh among engines

    auto new_specs = NegotiateArchitecture();

    // Output
    double energy = 0, cycles = 0, area;
    fs::create_directories(out_dir_);
    int max_digits_layers = std::to_string(num_layers_).length();

    for (size_t l = 0; l < num_layers_; l++) 
    {
      size_t w = layer_workload_lookup_[l];
      model::Engine engine;
      engine.Spec(new_specs);
      auto status = engine.Evaluate(mappings_[w], workloads_[w]);

      if (!engine.IsEvaluated())
      {
        std::cout << "Error in workload " << w << std::endl;
        std::cout << "Mapping: " << mappings_[w].PrintCompact() << std::endl;

        for (auto &s : status) std::cout << s.success << " " << s.fail_reason << std::endl;
        exit(1);
      }
      
      std::string layer_id = std::to_string(l);
      std::string stats_filename = out_dir_ + "/medea.negotiator.stats." + std::string(max_digits_layers - layer_id.length(), '0') + layer_id + ".txt";
      std::ofstream stats_file(stats_filename);
      stats_file << engine << std::endl;
      stats_file.close();

      energy += engine.Energy();
      cycles += engine.Cycles();
      area = engine.Area();
    }

    std::string g_stats_filename = out_dir_ + "/medea.negotiator.stats.global.txt";
    std::ofstream g_stats_file(g_stats_filename);
    g_stats_file << "stats:" << std::endl; 
    g_stats_file << "  energy: " << energy << std::endl;
    g_stats_file << "  cycles: " << cycles << std::endl;
    g_stats_file << "  area: " << area << std::endl;
    g_stats_file.close();
  }


  model::Engine::Specs MedeaSimpleNegotiator::NegotiateArchitecture() 
  {
    unsigned buffer_update_granularity = 16;

    struct Level {
      unsigned parx, pary, size;
    };
    unsigned num_arch_levels = default_arch_.topology.NumLevels();
    std::vector<std::vector<Level>> minimal_archs(num_workloads_, std::vector<Level>(num_arch_levels));

    for (size_t w = 0; w < num_workloads_; w++) {
      // Buffers
      for (size_t a = num_arch_levels - 2; a > 0; a--) 
      {
        auto buffer = default_arch_.topology.GetStorageLevel(a - 1);
        if ( !buffer->block_size.IsSpecified() || !buffer->size.IsSpecified() ) continue;
        auto block_size = buffer->block_size.Get();
        auto tile_sizes = engines_[w].GetTopology().GetStats().tile_sizes.at(a-1);
        auto utilized_capacity = std::accumulate(tile_sizes.begin(), tile_sizes.end(), 0);
        unsigned needed_depth = (utilized_capacity / block_size) + 1;
        unsigned remainder = needed_depth % buffer_update_granularity;
        unsigned new_depth = remainder ? needed_depth + buffer_update_granularity - remainder : needed_depth;

        minimal_archs[w][a].size = new_depth * block_size;
        minimal_archs[w][a].parx = GetParallelAtLevel(mappings_[w], spacetime::Dimension::SpaceX, a);
        minimal_archs[w][a].pary = GetParallelAtLevel(mappings_[w], spacetime::Dimension::SpaceY, a);
      }

      // Arithmetic
      minimal_archs[w][0].size = 0;
      minimal_archs[w][0].parx = GetParallelAtLevel(mappings_[w], spacetime::Dimension::SpaceX, 0);
      minimal_archs[w][0].pary = GetParallelAtLevel(mappings_[w], spacetime::Dimension::SpaceY, 0);
    }

    // Negotiation
    std::vector<Level> negotiated_arch(num_arch_levels, {0,0,0});
    
    for (size_t l = 0; l < num_layers_; l++) {
      size_t w = layer_workload_lookup_[l];
      for (size_t a = 0; a < num_arch_levels; a++) 
      {
        negotiated_arch[a].size = std::max(negotiated_arch[a].size, minimal_archs[w][a].size);
        negotiated_arch[a].parx = std::max(negotiated_arch[a].parx, minimal_archs[w][a].parx);
        negotiated_arch[a].pary = std::max(negotiated_arch[a].pary, minimal_archs[w][a].pary);
      }
    }

    std::map<std::string, uint64_t> updates;

    auto new_specs = model::Topology::Specs(default_arch_.topology);
    bool first_level_set = false;
    for (size_t a = num_arch_levels - 2; a > 0; a--) 
    {
      auto buffer = new_specs.GetStorageLevel(a-1);
      if ( !buffer->block_size.IsSpecified() || !buffer->size.IsSpecified() ) continue;
      if (first_level_set) 
      {
        buffer->meshX = negotiated_arch[a].parx * new_specs.GetStorageLevel(a)->meshX.Get();
        buffer->meshY = negotiated_arch[a].pary * new_specs.GetStorageLevel(a)->meshY.Get();
      } 
      else 
      {
        buffer->meshX = negotiated_arch[a].parx;
        buffer->meshY = negotiated_arch[a].pary;
      }
      buffer->instances = buffer->meshX.Get() * buffer->meshY.Get();
      buffer->size = negotiated_arch[a].size;
      buffer->effective_size = static_cast<uint64_t>(std::floor(buffer->size.Get() / buffer->multiple_buffering.Get()));
      
      updates[buffer->name.Get()] = buffer->size.Get() / buffer->block_size.Get();
      first_level_set = true;
    }
    
    std::string out_prefix = "medea.simple_negotiator_tmp";
    Accelergy::RT rt = accelergy_.GetReferenceTables(updates, out_prefix);

    model::Engine::Specs new_engine_specs;
    new_engine_specs.topology = new_specs;
    new_engine_specs.topology.ParseAccelergyART(rt.area);
    new_engine_specs.topology.ParseAccelergyERT(rt.energy);
    return new_engine_specs;
  }


  uint64_t MedeaSimpleNegotiator::GetParallelAtLevel(const Mapping &mapping, spacetime::Dimension dim, uint64_t level)
  {
    uint64_t result = 1;
    size_t start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level - 1) + 1 : 0;
    size_t end = mapping.loop_nest.storage_tiling_boundaries.at(level) + 1;
    for (size_t l = start; l < end; l++)
      if (mapping.loop_nest.loops[l].spacetime_dimension == dim)
        result *= mapping.loop_nest.loops[l].end;
    return result;
  }

  MedeaSimpleNegotiator::~MedeaSimpleNegotiator() 
  {
  }

}

