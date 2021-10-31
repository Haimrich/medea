#include "negotiator.hpp"

#include <boost/filesystem.hpp>
#include "yaml-cpp/yaml.h"

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"

#include "common.hpp"
#include "accelergy.hpp"
#include "mapping-parser-fix.hpp"
#include "negotiator-thread.hpp"

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

    auto medea = rootNode.lookup("medea");

    double pareto_sieving = -1;
    medea.lookupValue("pareto-sieving", pareto_sieving);

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
      
      if (0 <= pareto_sieving || pareto_sieving >= 1) 
      {
        workload_mappings_.push_back(mappings);
      } 
      else // Sieving 
      {
        size_t num_selected = static_cast<size_t>(std::ceil(pareto_sieving * mappings.size()));
        std::vector<MedeaMapping> sieved_mappings(2 * num_selected);

        std::partial_sort_copy(mappings.begin(), mappings.end(), sieved_mappings.begin(), sieved_mappings.begin() + num_selected,
                               [&](const MedeaMapping a, const MedeaMapping b) -> bool
                               { return a.cycles < b.cycles || (a.cycles == b.cycles && a.energy < b.energy); });
        std::partial_sort_copy(mappings.begin(), mappings.end(), sieved_mappings.begin() + num_selected, sieved_mappings.end(),
                               [&](const MedeaMapping a, const MedeaMapping b) -> bool
                               { return a.energy < b.energy || (a.energy == b.energy && a.cycles < b.cycles); });

        workload_mappings_.push_back(sieved_mappings);
      }
    }
    
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

    parent_population_.resize(population_size_);
    merged_population_.resize(2*population_size_);
  }


  unsigned MedeaNegotiator::Run()
  {
    auto chrono_start = std::chrono::high_resolution_clock::now();

    std::vector<MedeaNegotiatorThread *> threads_;
    for (unsigned t = 0; t < num_threads_; t++)
    {
      threads_.push_back(
        new MedeaNegotiatorThread(
            t,
            config_,
            default_arch_,
            accelergy_,
            thread_orchestrator_,
            num_threads_,
            rng_,
            layer_workload_lookup_,
            workloads_,
            workload_mappings_,
            num_generations_,
            population_size_,
            parent_population_,
            population_,
            mutation_prob_,
            individual_mutation_prob_,
            crossover_prob_,
            individual_crossover_prob_
        )
      );
    }

    for (unsigned t = 0; t < num_threads_; t++)
      threads_.at(t)->Start();

    thread_orchestrator_->LeaderDone();
    thread_orchestrator_->LeaderWait();

    //AssignRankAndCrowdingDistance(parent_population_);
    population_ = parent_population_;
    std::cout << "[INFO] Initial Population Done." << std::endl;

    for (unsigned g = 0; g < num_generations_; g++) 
    {
      std::cout << "[THRD] [" << std::string(num_threads_, '-') << "]\r[THRD] [" << std::flush;
      thread_orchestrator_->LeaderDone();

      // Crossover, mutation in threads.

      thread_orchestrator_->LeaderWait();
      std::cout << "]" << std::endl;
      
      Merging();
      AssignRankAndCrowdingDistance(merged_population_);
      Survival();
      
      population_ = parent_population_;
      std::cout << "[INFO] Generation " << g << " completed." << std::endl;
    }

    thread_orchestrator_->LeaderDone();

    for (unsigned t = 0; t < num_threads_; t++)
      threads_.at(t)->Join();

    auto chrono_end = std::chrono::high_resolution_clock::now();
    auto chrono_duration = std::chrono::duration_cast<std::chrono::seconds>(chrono_end - chrono_start).count();

    std::cout << std::endl;
    std::cout << "[MEDEA] Elapsed time: " << chrono_duration << " seconds. Saving output..." << std::endl;

    return OutputParetoFrontFiles();
  }


  unsigned MedeaNegotiator::OutputParetoFrontFiles()
  {
    std::string dir = out_dir_ + "/negotiated_pareto";
    int max_digits_inds = std::to_string(population_size_).length();
    int max_digits_layers = std::to_string(layer_workload_lookup_.size()).length();
    
    std::string global_stats_filename = dir + "/medea.stats.txt";
    std::ofstream global_stats_file(global_stats_filename);

    unsigned p = 0;
    for (auto& ind : parent_population_)
    {
      if (ind.rank) continue;
      global_stats_file << ind.objectives[0] << ", " << ind.objectives[1] << ", " << ind.objectives[2] << std::endl;

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

    global_stats_file.close();
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