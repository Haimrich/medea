#include "negotiator.hpp"

#include <boost/filesystem.hpp>
#include "yaml-cpp/yaml.h"

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"

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
      problem::Workload workload;
      config::CompoundConfig workload_config(workload_file.c_str());
      problem::ParseWorkload(workload_config.getRoot(), workload);

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

    num_generations_ = 20;
    medea.lookupValue("negotiator-num-generations", num_generations_);

    mutation_prob_ = 0.4;
    medea.lookupValue("negotiator-mutation-prob", mutation_prob_);
    individual_mutation_prob_ = 0.4;
    medea.lookupValue("negotiator-individual-mutation-prob", individual_mutation_prob_);
    crossover_prob_ = 0.4;
    medea.lookupValue("negotiator-crossover-prob", crossover_prob_);
    individual_crossover_prob_ = 0.4;
    medea.lookupValue("negotiator-indvidual-crossover-prob", individual_crossover_prob_);

    num_layers_ = lookup.size();

    parent_population_.reserve(population_size_);
    for (size_t p = 0; p < population_size_; p++) {
      NegotiatorIndividual ind = RandomIndividual();
      EvaluateIndividual(ind);
      parent_population_.push_back(ind);   
    }

    merged_population_.resize(2*population_size_);
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
          Crossover(population_[p], population_[p+1]);

        if (dice(*rng_) < mutation_prob_) 
          MutateIndividual(population_[p]);
        if (dice(*rng_) < mutation_prob_) 
          MutateIndividual(population_[p+1]);

        EvaluateIndividual(population_[p]);
        EvaluateIndividual(population_[p+1]);
      }
      
      Merging();
      AssignRankAndCrowdingDistance(merged_population_);
      Survival();
      
      std::cout << "[INFO] Generation " << g << " completed." << std::endl;
      std::cout << "[RANK]" << std::endl;
      for (auto& ind : parent_population_) 
        std::cout << ind.rank << "\t" << ind.objectives[0] << "\t" << ind.objectives[1] << "\t" << ind.objectives[2] << std::endl;
      std::cout << "[SIZE] " << parent_population_.size() << std::endl;
    }

    return OutputParetoFrontFiles();
  }



  unsigned MedeaNegotiator::OutputParetoFrontFiles()
  {
    std::string dir = out_dir_ + "/negotiated_pareto";
    int max_digits_inds = std::to_string(population_size_).length();
    int max_digits_layers = std::to_string(layer_workload_lookup_.size()).length();

    unsigned p = 0;
    for (auto& ind : parent_population_)
    {
      if (ind.rank) continue;

      std::string ind_id = std::to_string(p+1);
      std::string ind_out_path = dir + "/" + std::string(max_digits_inds - ind_id.length(), '0') + ind_id + "/";
      fs::create_directories(ind_out_path);

      model::Engine engine;
      engine.Spec(ind.negotiated_arch_specs);

      for (size_t l = 0; l < layer_workload_lookup_.size(); l++) 
      {
        engine.Evaluate(workload_mappings_[layer_workload_lookup_[l]][ind.mapping_id_set[l]].mapping, workloads_[layer_workload_lookup_[l]]);
        assert(engine.IsEvaluated());
        
        std::string layer_id = std::to_string(l);
        std::string stats_filename = ind_out_path + "/medea.negotiator.stats." + std::string(max_digits_layers - layer_id.length(), '0') + layer_id + ".txt";
        std::ofstream stats_file(stats_filename);
        stats_file << engine << std::endl;
        stats_file.close();
      }

      p++;
    }
    return p;
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
        switch (CheckDominance(population[i], population[j]))
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
    for (size_t p : pareto_front)
      population[p].crowding_distance = 0.0;

    for (unsigned i = 0; i < 3; i++)
    {
      std::sort(pareto_front.begin(), pareto_front.end(), [&](const size_t a, const size_t b) -> bool
                { return population[a].objectives[i] < population[b].objectives[i]; });

      population[pareto_front.front()].crowding_distance = 10e14;
      population[pareto_front.back()].crowding_distance = 10e14;

      double range = population[pareto_front.back()].objectives[i] - population[pareto_front.front()].objectives[i];
      assert(range >= 0);

      for (size_t j = 1; j < pareto_front.size() - 1; j++)
      {
        size_t r_prev = pareto_front[j - 1];
        size_t r_next = pareto_front[j + 1];
        size_t r_this = pareto_front[j];
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



  NegotiatorIndividual MedeaNegotiator::RandomIndividual()
  {
    NegotiatorIndividual out;
    out.mapping_id_set.reserve(num_layers_);
    out.mapping_evaluations.resize(num_layers_);

    for (size_t l = 0; l < num_layers_; l++)
    {
      size_t workload_id = layer_workload_lookup_[l];
      std::uniform_int_distribution<size_t> uni_dist(0, workload_mappings_[workload_id].size() - 1);
      out.mapping_id_set.push_back(uni_dist(*rng_));
      out.mapping_evaluations[l].need_evaluation = true;
    }

    return out;
  }

  bool MedeaNegotiator::NegotiateArchitecture(NegotiatorIndividual& ind)
  {
    auto old_negotiated_arch = ind.negotiated_arch;

    ind.negotiated_arch = workload_mappings_[layer_workload_lookup_[0]][ind.mapping_id_set[0]].arch;
    for (size_t l = 0; l < num_layers_; l++)
      ind.negotiated_arch &= workload_mappings_[layer_workload_lookup_[l]][ind.mapping_id_set[l]].arch;

    if (old_negotiated_arch == ind.negotiated_arch)
      return false;

    auto new_specs = model::Topology::Specs(default_arch_.topology);

    auto minimal_arithmetic = ind.negotiated_arch.GetLevel(0);
    auto arithmetic = new_specs.GetArithmeticLevel();
    arithmetic->meshX = minimal_arithmetic.mesh_x;
    arithmetic->meshY = minimal_arithmetic.mesh_y;
    arithmetic->instances = minimal_arithmetic.mesh_x * minimal_arithmetic.mesh_y;

    std::map<std::string, uint64_t> updates;

    for (unsigned i = 1; i < default_arch_.topology.NumLevels(); i++)
    {
      auto buffer = new_specs.GetStorageLevel(i - 1);
      if (!buffer->size.IsSpecified())
        continue;

      auto minimal_buffer = ind.negotiated_arch.GetLevel(i);
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

    ind.negotiated_arch_specs.topology = new_specs;
    ind.negotiated_arch_specs.topology.ParseAccelergyART(rt.area);
    ind.negotiated_arch_specs.topology.ParseAccelergyERT(rt.energy);

    return true;
  }

  void MedeaNegotiator::EvaluateIndividual(NegotiatorIndividual &ind)
  {
    bool arch_changed = NegotiateArchitecture(ind);
    model::Engine engine;
    engine.Spec(ind.negotiated_arch_specs);

    for (size_t i = 0; i < num_layers_; i++)
      if (ind.mapping_evaluations[i].need_evaluation || arch_changed)
      {
        auto status = engine.Evaluate(workload_mappings_[layer_workload_lookup_[i]][ind.mapping_id_set[i]].mapping, workloads_[layer_workload_lookup_[i]]);

        if (!engine.IsEvaluated())
        {
          std::cout << "Error in workload " << layer_workload_lookup_[i] << " with mapping ";
          std::cout << workload_mappings_[layer_workload_lookup_[i]][ind.mapping_id_set[i]].mapping.id << std::endl;

          for (auto &s : status)
            std::cout << s.success << " " << s.fail_reason << std::endl;
          YAML::Emitter yout;
          yout << ind.negotiated_arch;
          std::cout << yout.c_str() << std::endl;
          exit(1);
        }

        ind.mapping_evaluations[i].energy = engine.Energy();
        ind.mapping_evaluations[i].cycles = engine.Cycles();
        ind.mapping_evaluations[i].need_evaluation = false;
      }

    ind.objectives[0] = 0;
    ind.objectives[1] = 0;
    ind.objectives[2] = engine.Area();

    for (size_t i = 0; i < num_layers_; i++)
    {
      ind.objectives[0] += ind.mapping_evaluations[i].energy;
      ind.objectives[1] += ind.mapping_evaluations[i].cycles;
    }
  }

  void MedeaNegotiator::MutateIndividual(NegotiatorIndividual &ind)
  {
    std::uniform_real_distribution<double> dice(0, 1);

    for (size_t l = 0; l < num_layers_; l++)
      if (dice(*rng_) < individual_mutation_prob_)
      {
        size_t workload_id = layer_workload_lookup_[l];
        std::uniform_int_distribution<size_t> uni_dist(0, workload_mappings_[workload_id].size() - 1);
        ind.mapping_id_set[l] = uni_dist(*rng_);
        ind.mapping_evaluations[l].need_evaluation = true;
      }
  }

  void MedeaNegotiator::Crossover(NegotiatorIndividual &offspring_a, NegotiatorIndividual &offspring_b)
  {
    std::uniform_real_distribution<double> dice(0, 1);

    for (size_t l = 0; l < num_layers_; l++)
      if (dice(*rng_) < individual_crossover_prob_)
      {
        std::swap(offspring_a.mapping_id_set[l], offspring_b.mapping_id_set[l]);

        offspring_a.mapping_evaluations[l].need_evaluation = true;
        offspring_b.mapping_evaluations[l].need_evaluation = true;
      }
  }

  Dominance MedeaNegotiator::CheckDominance(const NegotiatorIndividual &a, const NegotiatorIndividual &b)
  {
    bool all_a_less_or_equal_than_b = true;
    bool any_a_less_than_b = false;
    bool all_b_less_or_equal_than_a = true;
    bool any_b_less_than_a = false;

    for (unsigned i = 0; i < 3; i++)
    {
      if (a.objectives[i] > b.objectives[i])
      {
        all_a_less_or_equal_than_b = false;
        any_b_less_than_a = true;
      }
      else if (b.objectives[i] > a.objectives[i])
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

/*
  NegotiatorIndividual &NegotiatorIndividual::operator=(const NegotiatorIndividual &other)
  {
    num_layers_ = other.num_layers_;
    mapping_set_ = other.mapping_set_;
    negotiated_arch_ = other.negotiated_arch_;
    rng_ = other.rng_;
    set_engines_ = other.set_engines_;
    need_evaluation_ = other.need_evaluation_;
    default_arch_specs_ = other.default_arch_specs_;
    negotiated_arch_specs_ = other.negotiated_arch_specs_;
    accelergy_ = other.accelergy_;

    rank = other.rank;
    crowding_distance = other.crowding_distance;

    objectives = other.objectives;

    return *this;
  }
*/

  MedeaNegotiator::~MedeaNegotiator()
  {
    delete rng_;
    delete thread_orchestrator_;
  }
  

  MedeaMapping::MedeaMapping(unsigned id, config::CompoundConfig &config, model::Engine::Specs &arch_specs, problem::Workload &workload) : id(id)
  {
    auto root = config.getRoot();
    auto mapping_config = root.lookup("mapping");

    arch = MinimalArchSpecs(root.lookup("arch").getYNode());

    mapping = mapping::ParseAndConstructFixed(mapping_config, arch_specs, workload);

    auto stats = root.lookup("stats");
    stats.lookupValue("energy", energy);
    stats.lookupValue("cycles", cycles);
    stats.lookupValue("area", area);
  }
}