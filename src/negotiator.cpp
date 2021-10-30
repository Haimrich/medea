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
    std::random_device rd;
    rng_ = new std::mt19937(rd());

    unsigned population_size = 100;
    medea.lookupValue("negotiator-population-size", population_size);
    population_size_ = population_size + population_size % 2;

    parent_population_ = std::vector<NegotiatorIndividual>(population_size_, {rng_, workloads_, layer_workload_lookup_, workload_mappings_, default_arch_, &accelergy_});
    merged_population_.resize(2*population_size_);

    num_generations_ = 20;
    medea.lookupValue("negotiator-num-generations", num_generations_);

    mutation_prob_ = 0.4;
    medea.lookupValue("negotiator-mutation-prob", mutation_prob_);
    individual_mutation_prob_ = 0.4;
    medea.lookupValue("negotiator-individual-mutation-prob", individual_mutation_prob_);
    crossover_prob_ = 0.4;
    medea.lookupValue("negotiator-crossover-prob", crossover_prob_);
    individual_crossover_prob_ = 0.4;
    medea.lookupValue("negotiator-idnividual-crossover-prob", individual_crossover_prob_);
  }


  unsigned MedeaNegotiator::Run()
  {
    //AssignRankAndCrowdingDistance(parent_population_);

    for (unsigned g = 0; g < num_generations_; g++) 
    {
      population_ = parent_population_;

      std::uniform_real_distribution<double> dice(0, 1);
      for (size_t p = 0; p < population_size_; p += 2)
      {
        if (dice(*rng_) < crossover_prob_)
          NegotiatorIndividual::Crossover(individual_crossover_prob_, population_[p], population_[p+1]);

        if (dice(*rng_) < mutation_prob_) population_[p].Mutate(individual_mutation_prob_, workload_mappings_, layer_workload_lookup_);
        if (dice(*rng_) < mutation_prob_) population_[p+1].Mutate(individual_mutation_prob_, workload_mappings_, layer_workload_lookup_);

        population_[p].Evaluate(workloads_, layer_workload_lookup_);
        population_[p+1].Evaluate(workloads_, layer_workload_lookup_);
      }
      
      Merging();
      AssignRankAndCrowdingDistance(merged_population_);
      Survival();
      
      std::cout << "[INFO] Generation " << g << " completed." << std::endl;
    }
    return 0;
  }


  void MedeaNegotiator::Merging()
  {
    std::copy(
        population_.begin(),
        population_.end(),
        merged_population_.begin()
    );
    std::copy(
        parent_population_.begin(),
        parent_population_.end(),
        merged_population_.begin() + population_size_
    );
  }


  void MedeaNegotiator::AssignRankAndCrowdingDistance(NegotiatorPopulation &population)
  {
    std::vector<size_t> pareto_front;
    std::vector<std::vector<size_t>> dominated_by(population.size(), std::vector<size_t>());
    std::vector<size_t> num_dominating(population.size(), 0);

    for (uint64_t i = 0; i < population.size(); i++)
    {
      for (uint64_t j = i + 1; j < population.size(); j++)
      {
        switch (population[i].CheckDominance(population[j]))
        {
        case Dominance::DOMINATING:
          dominated_by[i].push_back(j);
          num_dominating[j]++;
          break;
        case Dominance::DOMINATED:
          dominated_by[j].push_back(i);
          num_dominating[i]++;
          break;
        case Dominance::FRONTIER:
          break;
        }
      }

      if (num_dominating[i] == 0)
      {
        population[i].rank = 0;
        pareto_front.push_back(i);
      }
    }

    size_t total_debug = 0;
    for (size_t f = 0; !pareto_front.empty(); f++)
    {
      total_debug += pareto_front.size();

      AssignCrowdingDistance(population, pareto_front);
      std::vector<size_t> new_pareto_front;
      
      for (size_t p : pareto_front)
        for (size_t q : dominated_by[p])
        {
          num_dominating[q]--;

          if (num_dominating[q] == 0)
          {
            population[q].rank = f + 1;
            new_pareto_front.push_back(q);
          }
        }
      
      pareto_front = new_pareto_front;
    }

    assert(total_debug == population.size());
  }


  void MedeaNegotiator::AssignCrowdingDistance(NegotiatorPopulation &population, std::vector<size_t> &pareto_front)
  {
    for (auto p : pareto_front)
      population[p].crowding_distance = 0.0;

    for (unsigned i = 0; i < 3; i++)
    {
      std::sort(pareto_front.begin(), pareto_front.end(), [&](const uint64_t a, const uint64_t b) -> bool
                { return population[a].objectives[i] < population[b].objectives[i]; });

      population[pareto_front.front()].crowding_distance = 10e14;
      population[pareto_front.back()].crowding_distance = 10e14;

      double range = population[pareto_front.back()].objectives[i] - population[pareto_front.front()].objectives[i];
      assert(range >= 0);

      for (uint64_t j = 1; j < pareto_front.size() - 1; j++)
      {
        uint64_t r_prev = pareto_front[j - 1];
        uint64_t r_next = pareto_front[j + 1];
        uint64_t r_this = pareto_front[j];
        population[r_this].crowding_distance += std::abs(population[r_next].objectives[i] - population[r_prev].objectives[i]) / range;
      }
    }
  }


  void MedeaNegotiator::Survival()
  {
    // Sort by rank and crowding distance and select population_size_
    std::partial_sort_copy(
        merged_population_.begin(), merged_population_.end(),
        parent_population_.begin(), parent_population_.end(),
        [&](const NegotiatorIndividual &a, const NegotiatorIndividual &b) -> bool
        {
          return a.rank < b.rank || (a.rank == b.rank && a.crowding_distance > b.crowding_distance);
        });

    // Shuffle
    //if (!use_tournament_)
    std::shuffle(std::begin(parent_population_), std::end(parent_population_), *rng_);
  }


  MedeaNegotiator::~MedeaNegotiator()
  {
    delete rng_;
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


  NegotiatorIndividual::NegotiatorIndividual(std::mt19937* rng, std::vector<problem::Workload> &workloads, const std::vector<size_t> &lookup, 
                                             const std::vector<std::vector<MedeaMapping>> &mappings, model::Engine::Specs arch_specs, Accelergy* accelergy) : 
                                             rng_(rng), default_arch_specs_(arch_specs)
  {
    accelergy_ = accelergy;
    num_layers_ = lookup.size();
    mapping_set_.reserve(num_layers_);
    need_evaluation_.resize(num_layers_, true);
    set_engines_.resize(num_layers_);

    std::uniform_int_distribution<size_t> uni_dist(0, mappings[0].size()-1);
    auto m = mappings[0][uni_dist(*rng_)];
    mapping_set_.push_back(m);

    for (size_t l = 1; l < num_layers_; l++)
    {
      size_t workload_id = lookup[l];
      uni_dist = std::uniform_int_distribution<size_t>(0, mappings[workload_id].size()-1);
      m = mappings[workload_id][uni_dist(*rng_)];
      mapping_set_.push_back(m);
    }

    Evaluate(workloads, lookup);
  }


  bool NegotiatorIndividual::NegotiateArchitecture() 
  {
    auto old_negotiated_arch = negotiated_arch_;
    
    negotiated_arch_ = mapping_set_[0].arch;
    for (size_t i = 0; i < num_layers_; i++)
      negotiated_arch_ &= mapping_set_[i].arch;

    if (old_negotiated_arch == negotiated_arch_) 
      return false;

    auto new_specs = model::Topology::Specs(default_arch_specs_.topology);

    auto minimal_arithmetic = negotiated_arch_.GetLevel(0);
    auto arithmetic = new_specs.GetArithmeticLevel();
    arithmetic->meshX = minimal_arithmetic.mesh_x;
    arithmetic->meshY = minimal_arithmetic.mesh_y;
    arithmetic->instances = minimal_arithmetic.mesh_x * minimal_arithmetic.mesh_y;

    std::map<std::string, uint64_t> updates;

    for (unsigned i = 1; i < default_arch_specs_.topology.NumLevels();  i++)
    {
      auto buffer = new_specs.GetStorageLevel(i - 1);
      if (!buffer->size.IsSpecified()) continue;

      auto minimal_buffer = negotiated_arch_.GetLevel(i);
      buffer->meshX = minimal_buffer.mesh_x;
      buffer->meshY = minimal_buffer.mesh_y;
      buffer->instances = minimal_buffer.mesh_x * minimal_buffer.mesh_y;
      buffer->size = minimal_buffer.size;
      buffer->effective_size = static_cast<uint64_t>(std::floor(minimal_buffer.size / buffer->multiple_buffering.Get()));
    
      updates[buffer->name.Get()] = buffer->size.Get() / buffer->block_size.Get();
    }

    //std::string out_prefix = "medea." + std::to_string(thread_id_) + "_tmp";
    std::string out_prefix = "medea.tmp";
    Accelergy::RT rt = accelergy_->GetReferenceTables(updates, out_prefix);

    negotiated_arch_specs_.topology = new_specs;
    negotiated_arch_specs_.topology.ParseAccelergyART(rt.area);
    negotiated_arch_specs_.topology.ParseAccelergyERT(rt.energy);

    return true;
  }


  void NegotiatorIndividual::Evaluate(std::vector<problem::Workload> &workloads, const std::vector<size_t> &lookup) {
    bool arch_changed = NegotiateArchitecture();
    
    for (size_t i = 0; i < num_layers_; i++)
      if (need_evaluation_[i] || arch_changed) 
      {
        set_engines_[i].Spec(negotiated_arch_specs_);
        auto status = set_engines_[i].Evaluate(mapping_set_[i].mapping, workloads[lookup[i]]);
        if (!set_engines_[i].IsEvaluated()) {
          std::cout << "Error in workload " << lookup[i] << " with mapping " << mapping_set_[i].mapping.id << std::endl;
          for (auto& s : status)
            std::cout << s.success << " " << s.fail_reason << std::endl;
          YAML::Emitter yout;
          yout << negotiated_arch_;
          std::cout << yout.c_str() << std::endl;
          exit(1);
        }
        need_evaluation_[i] = false;
      }

    energy = 0;
    cycles = 0;
    area = set_engines_[0].Area();

    for (size_t i = 0; i < num_layers_; i++) 
    {
      energy += set_engines_[i].Energy();
      cycles += set_engines_[i].Cycles();
    }
  }


  void NegotiatorIndividual::Mutate(double mutation_prob, const std::vector<std::vector<MedeaMapping>> &mappings, const std::vector<size_t> &lookup) 
  {
    std::uniform_real_distribution<double> dice(0, 1);

    for (size_t l = 0; l < num_layers_; l++) 
      if (dice(*rng_) < mutation_prob) 
      {
        size_t workload_id = lookup[l];
        std::uniform_int_distribution<size_t> uni_dist(0, mappings[workload_id].size()-1);
        mapping_set_[l] =  mappings[workload_id][uni_dist(*rng_)];
        need_evaluation_[l] = true;
      }
  }


  void NegotiatorIndividual::Crossover(double crossover_prob, NegotiatorIndividual& offspring_a, NegotiatorIndividual& offspring_b)
  {
    std::uniform_real_distribution<double> dice(0, 1);

    for (size_t l = 0; l < offspring_a.num_layers_; l++) 
      if (dice(*(offspring_a.rng_)) < crossover_prob) 
      {
        std::swap(offspring_a.mapping_set_[l], offspring_b.mapping_set_[l]);
        
        offspring_a.need_evaluation_[l] = true;
        offspring_b.need_evaluation_[l] = true;
      }
  }


  Dominance NegotiatorIndividual::CheckDominance(const NegotiatorIndividual &other)
  {
    bool all_a_less_or_equal_than_b = true;
    bool any_a_less_than_b = false;
    bool all_b_less_or_equal_than_a = true;
    bool any_b_less_than_a = false;

    for (unsigned i = 0; i < 3; i++)
    {
      if (objectives[i] > other.objectives[i])
      {
        all_a_less_or_equal_than_b = false;
        any_b_less_than_a = true;
      }
      else if (other.objectives[i] > objectives[i])
      {
        any_a_less_than_b = true;
        all_b_less_or_equal_than_a = false;
      }
    }

    if (all_a_less_or_equal_than_b && any_a_less_than_b)
      return Dominance::DOMINATING;
    if (all_b_less_or_equal_than_a && any_b_less_than_a)
      return Dominance::DOMINATED;

    return Dominance::FRONTIER;
  }


  NegotiatorIndividual& NegotiatorIndividual::operator=(const NegotiatorIndividual& other) {
    mapping_set_ = other.mapping_set_;
    negotiated_arch_ = other.negotiated_arch_;
    rng_ = other.rng_;
    set_engines_ = other.set_engines_;
    need_evaluation_ = other.need_evaluation_;
    default_arch_specs_ = other.default_arch_specs_;
    negotiated_arch_specs_ = other.negotiated_arch_specs_;
    accelergy_ = other.accelergy_;

    objectives[0] = other.objectives[0];
    objectives[1] = other.objectives[1];
    objectives[2] = other.objectives[2];

    return *this;
  }
}