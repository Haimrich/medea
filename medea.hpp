#pragma once

#include "applications/medea/medea-common.hpp"
#include "applications/medea/medea-thread.hpp"

#include <fstream>
#include <iomanip>
#include <cmath> 

#include "util/accelergy_interface.hpp"
#include "compound-config/compound-config.hpp"
#include "util/numeric.hpp"

#include "mapping/loop.hpp"


//--------------------------------------------//
//                Application                 //
//--------------------------------------------//
namespace medea {

class Medea
{
 protected:

  problem::Workload workload_;
  model::Engine::Specs arch_specs_;
  mapspace::MapSpace* mapspace_;
  mapping::Constraints* constraints_;

  std::string out_dir_ = ".";
  std::string out_prefix_ = "medea";

  uint32_t num_generations_;
  uint32_t population_size_;
  uint32_t immigrant_population_size_;

  Population population_, parent_population_, immigrant_population_, merged_population_;

  Orchestrator* thread_orchestrator_;
  std::mutex best_mutex_;

  Individual best_individual_;

  uint32_t num_threads_;

  RandomGenerator128 *if_rng_, *lp_rng_, *db_rng_, *sp_rng_, *crossover_rng_;

  std::random_device rand_dev;
  std::mt19937_64 rng;
  //std::default_random_engine rng;
  std::uniform_real_distribution<> proba;

  enum Dominance { DOMINATING, DOMINATED, FRONTIER };

 public:

  Medea(config::CompoundConfig* config, std::string out_dir) :
    out_dir_(out_dir),
    rng(rand_dev()),
    proba(0, 1)
  {
    auto rootNode = config->getRoot();

    // Problem configuration.
    auto problem = rootNode.lookup("problem");
    problem::ParseWorkload(problem, workload_);
    std::cout << "Problem configuration complete." << std::endl;

    // Architecture configuration.
    config::CompoundConfigNode arch;
    arch = rootNode.lookup("architecture");
    arch_specs_ = model::Engine::ParseSpecs(arch);

#ifdef USE_ACCELERGY
    // Call accelergy ERT with all input files
    if (arch.exists("subtree") || arch.exists("local")) {
      accelergy::invokeAccelergy(config->inFiles, out_prefix_, out_dir_);
      std::string ertPath = out_prefix_ + ".ERT.yaml";
      auto ertConfig = new config::CompoundConfig(ertPath.c_str());
      auto ert = ertConfig->getRoot().lookup("ERT");
      std::cout << "Generate Accelergy ERT (energy reference table) to replace internal energy model." << std::endl;
      arch_specs_.topology.ParseAccelergyERT(ert);
      
      std::string artPath = out_prefix_ + ".ART.yaml";
      auto artConfig = new config::CompoundConfig(artPath.c_str());
      auto art = artConfig->getRoot().lookup("ART");
      std::cout << "Generate Accelergy ART (area reference table) to replace internal area model." << std::endl;
      arch_specs_.topology.ParseAccelergyART(art);
    }
#endif
    std::cout << "Architecture configuration complete." << std::endl;

    // MapSpace configuration.
    config::CompoundConfigNode arch_constraints;
    config::CompoundConfigNode mapspace;

   // Architecture constraints.
    if (arch.exists("constraints"))
      arch_constraints = arch.lookup("constraints");
    else if (rootNode.exists("arch_constraints"))
      arch_constraints = rootNode.lookup("arch_constraints");
    else if (rootNode.exists("architecture_constraints"))
      arch_constraints = rootNode.lookup("architecture_constraints");

    // Mapspace constraints.
    if (rootNode.exists("mapspace"))
      mapspace = rootNode.lookup("mapspace");
    else if (rootNode.exists("mapspace_constraints"))
      mapspace = rootNode.lookup("mapspace_constraints");


    auto tmp_mapspace = mapspace::ParseAndConstruct(mapspace, arch_constraints, arch_specs_, workload_);
    auto split_mapspaces = tmp_mapspace->Split(1);
    mapspace_ = split_mapspaces[0];

    // Constraints Object
    ArchProperties arch_props(arch_specs_);
    constraints_ = new mapping::Constraints(arch_props, workload_);
    constraints_->Parse(mapspace);
    constraints_->Parse(arch_constraints);

    std::cout << "Mapspace construction complete." << std::endl;

   
    // Mapper
    auto mapper = rootNode.lookup("mapper");

    num_generations_ = 30;
    mapper.lookupValue("num-generations", num_generations_);
    population_size_ = 100;
    mapper.lookupValue("population-size", population_size_);
    immigrant_population_size_ = 40;
    mapper.lookupValue("immigrant-population-size", immigrant_population_size_);

    population_.resize(population_size_);
    parent_population_.resize(population_size_);
    immigrant_population_.resize(immigrant_population_size_);
    merged_population_.resize(2*population_size_+immigrant_population_size_);

    std::cout << "Num. generations: " << num_generations_ << " - Pop. size: " << population_size_ << " - Immigrant pop. size: " << immigrant_population_size_ << std::endl;
  
    num_threads_ = std::thread::hardware_concurrency();
    if (mapper.lookupValue("num-threads", num_threads_))
      std::cout << "Using threads = " << num_threads_ << std::endl;
    else
      std::cout << "Using all available hardware threads = " << num_threads_ << std::endl;

  
    // Thread Orchestrator
    thread_orchestrator_ = new Orchestrator(num_threads_);

    // Random gen
    if_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::IndexFactorization));
    lp_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::LoopPermutation));
    db_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::DatatypeBypass));
    sp_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::Spatial));
    crossover_rng_ = new RandomGenerator128(10000);
  }

  ~Medea()
  {
    if (mapspace_) delete mapspace_;
    if (constraints_) delete constraints_;

    if (thread_orchestrator_) delete thread_orchestrator_;

    if (if_rng_) delete if_rng_;
    if (lp_rng_) delete lp_rng_;
    if (db_rng_) delete db_rng_;
    if (sp_rng_) delete sp_rng_;

    if (crossover_rng_) delete crossover_rng_;
    
  }

  void Survival() {
    // Sort by rank and crowding distance and select population_size_
    std::partial_sort_copy(
      merged_population_.begin(), merged_population_.end(),
      parent_population_.begin(), parent_population_.end(),
      [&](const Individual & a, const Individual & b) -> bool { 
        return a.rank < b.rank || (a.rank == b.rank && a.crowding_distance > b.crowding_distance); 
    });

    // Shuffle
    //std::shuffle(std::begin(parent_population_), std::end(parent_population_), rng);
  }

  Dominance CheckDominance(const Individual& a, const Individual& b) {
    if ( (a.energy <= b.energy && a.latency < b.latency) || (a.energy < b.energy && a.latency <= b.latency) )
      return Dominance::DOMINATING;
    
    if ( (b.energy <= a.energy && b.latency < a.latency) || (b.energy < a.energy && b.latency <= a.latency) )
      return Dominance::DOMINATED;
    
    return Dominance::FRONTIER;
  }

  void AssignCrowdingDistance(Population& population, std::vector<uint64_t>& pareto_front) {
    std::sort(pareto_front.begin(), pareto_front.end(), [&](const uint64_t & a, const uint64_t & b) -> bool { return population[a].energy < population[b].energy; });
    
    population[pareto_front.front()].crowding_distance = 10e14;
    population[pareto_front.back()].crowding_distance = 10e14;

    double range = population[pareto_front.back()].energy - population[pareto_front.front()].energy;
    assert(range >= 0);

    for (uint64_t i = 1; i < pareto_front.size() - 1; i++) {
      uint64_t r_prev = pareto_front[i-1];
      uint64_t r_next = pareto_front[i+1];
      uint64_t r_this = pareto_front[i];
      population[r_this].crowding_distance = std::abs(population[r_next].energy - population[r_prev].energy) / range;
    }

    std::sort(pareto_front.begin(), pareto_front.end(), [&](const uint64_t & a, const uint64_t & b) -> bool { return population[a].latency < population[b].latency; });
    
    population[pareto_front.front()].crowding_distance = 10e14;
    population[pareto_front.back()].crowding_distance = 10e14;

    range = population[pareto_front.back()].latency - population[pareto_front.front()].latency;

    for (uint64_t i = 1; i < pareto_front.size() - 1; i++) {
      uint64_t r_prev = pareto_front[i-1];
      uint64_t r_next = pareto_front[i+1];
      uint64_t r_this = pareto_front[i];
      population[r_this].crowding_distance += std::abs(population[r_next].latency - population[r_prev].latency) / range;
    }
  
  }

  void AssignRankAndCrowdingDistance(Population& population) {
    std::vector<uint64_t> pareto_front;
    std::vector<std::vector<uint64_t>> dominated_by(population.size(), std::vector<uint64_t>());
    std::vector<uint64_t> num_dominating(population.size(), 0);

    for (uint64_t i = 0; i < population.size(); i++) {

      for (uint64_t j = i + 1; j < population.size(); j++) {

        switch ( CheckDominance(population[i], population[j]) ) {
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

      if (num_dominating[i] == 0) {
        population[i].rank = 0;
        pareto_front.push_back(i);
      }

    }

    uint64_t total_debug = 0;
    for (uint64_t f = 0; !pareto_front.empty(); f++) {
      total_debug += pareto_front.size();

      AssignCrowdingDistance(population, pareto_front);

      std::vector<uint64_t> new_pareto_front;

      for (uint64_t p : pareto_front) {

        for (uint64_t q : dominated_by[p]) {

          num_dominating[q]--;

          if (num_dominating[q] == 0) {
            population[q].rank = f + 1;
            new_pareto_front.push_back(q);
          }

        }
      }
      pareto_front = new_pareto_front;
    }
    std::cout << "TDB: " << total_debug << " POS: " << population.size() << std::endl;
  }

  void Merging() {
    std::copy(
      population_.begin(),
      population_.end(),
      merged_population_.begin()
    );
    std::copy(
      parent_population_.begin(),
      parent_population_.end(),
      merged_population_.begin()+population_size_
    );
    std::copy(
      immigrant_population_.begin(),
      immigrant_population_.end(),
      merged_population_.begin()+2*population_size_
    );
  }

  // ---------------
  // Run the mapper.
  // ---------------
  void Run()
  {
    // Output file names.
    const std::string stats_file_name = out_dir_ + "/" + out_prefix_ + ".stats.txt";
    const std::string map_txt_file_name = out_dir_ + "/" + out_prefix_ + ".map.txt";

    // =====
    // Thread Start.
    // =====

    std::vector<MedeaThread*> threads_;
    for (unsigned t = 0; t < num_threads_; t++)
    {
      threads_.push_back(
        new MedeaThread(
          t, 
          workload_,
          arch_specs_,
          mapspace_,
          constraints_,
          &best_individual_,
          immigrant_population_,
          parent_population_,
          population_,
          thread_orchestrator_,
          &best_mutex_,
          num_threads_,
          population_size_,
          immigrant_population_size_,
          num_generations_,
          if_rng_,
          lp_rng_,
          db_rng_,
          sp_rng_,
          crossover_rng_
        )
      );
    }

    for (unsigned t = 0; t < num_threads_; t++)
      threads_.at(t)->Start();

    thread_orchestrator_->LeaderDone();
    thread_orchestrator_->LeaderWait();
    
    std::cout << "[INFO] Initial Population Done." << std::endl;

    AssignRankAndCrowdingDistance(parent_population_);

    std::ofstream init_population_file(out_dir_ + "/" + out_prefix_ + ".initial_population.txt");
    for (Individual& ind : parent_population_) {
      init_population_file <<  ind.rank << "," << ind.crowding_distance << "," << ind.energy << "," << ind.latency << std::endl;
    }
    init_population_file.close();

    thread_orchestrator_->LeaderDone();

    for (uint32_t g = 0; g < num_generations_; g++) {
      
      // Wait Crossover, Mutation and Immigration
      thread_orchestrator_->LeaderWait();

      // Select for next generation
      Merging();
      AssignRankAndCrowdingDistance(merged_population_);
      std::cout << "[RANKS] ";
      for (Individual& ind : merged_population_) std::cout << ind.rank;
      std::cout << std::endl;
      Survival();
      std::cout << "[RANKS] ";
      for (Individual& ind : parent_population_) {
        std::cout << ind.rank;
      }
      std::cout << std::endl;
      

      double mean = 0.0;
      for (auto& i : parent_population_) mean += i.fitness;
      mean /= population_size_;
      std::cout << "[INFO] Generation " << g << " done. Average Fitness: " << mean << std::endl;

      thread_orchestrator_->LeaderDone();
    }


    // ============
    // Termination.
    // ============

    
    if (best_individual_.engine.IsEvaluated())
    {
      std::ofstream map_txt_file(map_txt_file_name);
      best_individual_.genome.PrettyPrint(map_txt_file, arch_specs_.topology.StorageLevelNames(),
                               best_individual_.engine.GetTopology().TileSizes());
      map_txt_file.close();

      std::ofstream stats_file(stats_file_name);
      stats_file << best_individual_.engine << std::endl;
      stats_file.close();

      std::cout << std::endl;
      std::cout << "Summary stats for best mapping found by mapper:" << std::endl; 
      std::cout << "  Utilization = " << std::setw(4) << std::fixed << std::setprecision(2)
                << best_individual_.engine.Utilization() << " | pJ/MACC = " << std::setw(8)
                << std::fixed << std::setprecision(3) << best_individual_.engine.Energy() /
        best_individual_.engine.GetTopology().MACCs() << std::endl;

      std::ofstream population_file(out_dir_ + "/" + out_prefix_ + ".population.txt");
      for (Individual& ind : parent_population_) {
        population_file <<  ind.rank << "," << ind.crowding_distance << "," << ind.energy << "," << ind.latency << std::endl;
      }
      population_file.close();
    }
    else
    {
      std::cout << "MESSAGE: no valid mappings found within search criteria." << std::endl;
    }
  }
};

}