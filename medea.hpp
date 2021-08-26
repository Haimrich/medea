#pragma once

#include "applications/medea/medea-common.hpp"
#include "applications/medea/medea-thread.hpp"

#include <fstream>
#include <iomanip>

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

  uint32_t generations_;
  uint32_t population_size_;
  uint32_t elite_population_size_;

  Population population_, parent_population_, merged_population_;

  Orchestrator* thread_orchestrator_;
  std::mutex best_mutex_;

  Individual best_individual_;

  uint32_t num_threads_;

  RandomGenerator128 *if_rng_, *lp_rng_, *db_rng_, *sp_rng_, *crossover_rng_;

  std::random_device rand_dev;
  std::mt19937_64 rng;
  //std::default_random_engine rng;
  std::uniform_real_distribution<> proba;

 public:

  Medea(config::CompoundConfig* config, std::string out_dir_) :
    out_dir_(out_dir_),
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

    generations_ = 30;
    mapper.lookupValue("num-generations", generations_);
    population_size_ = 100;
    mapper.lookupValue("population-size", population_size_);
    elite_population_size_ = 60;
    mapper.lookupValue("elite-population-size", elite_population_size_);

    population_.resize(population_size_);
    parent_population_.resize(population_size_);
    merged_population_.resize(2*population_size_);

    std::cout << "Num. generations: " << generations_ << " - Pop. size: " << population_size_ << " - Elite pop. size: " << elite_population_size_ << std::endl;
  
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

  // Stochastic Universal Sampling Selection
  void SUS() {
    double fitness_sum = 0.0;
    double min_fitness = 0.0;
    for (auto& i : merged_population_) if (i.fitness < min_fitness) min_fitness = i.fitness;
    for (auto& i : merged_population_) fitness_sum += i.fitness - min_fitness;
    double intsize = fitness_sum / population_size_;
    
    double ptr = proba(rng) * intsize;

    for (uint64_t i = 0; i < population_size_; i++) {
      unsigned j = 0;
      double fsum = 0.0;

      while (fsum <= ptr) {
        fsum += merged_population_[j].fitness - min_fitness;
        j++;
      }
      parent_population_[i] = merged_population_[j-1];

      ptr += intsize;
    }
  }

  // Fitness proportional roulette wheel
  void RWS() {
    double fitness_sum = 0.0;
    double min_fitness = 0.0;
    for (auto& i : merged_population_) if (i.fitness < min_fitness) min_fitness = i.fitness;
    for (auto& i : merged_population_) fitness_sum += i.fitness - min_fitness;
    
    
    for (uint64_t i = 0; i < population_size_; i++) {
      unsigned old_j = population_size_;
      unsigned j;

      // Avoiding consecutive repeating genomes because crossover would be useless
      do { 
        j = 0;
        double fsum = proba(rng) * fitness_sum;

        while (fsum >= 0.0) {
          fsum -= merged_population_[j].fitness - min_fitness;
          j++;
        }
      } while ( (j == old_j) && (i % 2) );
      
      old_j = j;
      parent_population_[i] = merged_population_[j-1];
    }
  }

  // Custom Selection
  void Selection() {
    // Select best ones as new parent population
    std::copy(merged_population_.begin(), merged_population_.begin()+population_size_, parent_population_.begin());

    // Shuffle
    std::shuffle(std::begin(parent_population_), std::end(parent_population_), rng);
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
          parent_population_,
          population_,
          thread_orchestrator_,
          &best_mutex_,
          num_threads_,
          population_size_,
          elite_population_size_,
          generations_,
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

    thread_orchestrator_->LeaderDone();

    for (uint32_t g = 0; g < generations_; g++) {
      
      // Wait Crossover
      thread_orchestrator_->LeaderWait();

      // Order by fitness
      std::sort(population_.begin(), population_.end(), 
          [](const Individual & a, const Individual & b) -> bool
      { 
          return a.fitness > b.fitness; 
      });

      // Start mutation
      thread_orchestrator_->LeaderDone();

      // Wait for Mutation
      thread_orchestrator_->LeaderWait();

      // Merge parent and offspring pop
      std::merge(population_.begin(), population_.end(), parent_population_.begin(), parent_population_.end(), merged_population_.begin(),
      [](const Individual & a, const Individual & b) -> bool
      { 
          return a.fitness > b.fitness; 
      });

      Selection();
      //SUS();
      //RWS();

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
    }
    else
    {
      std::cout << "MESSAGE: no valid mappings found within search criteria." << std::endl;
    }
  }
};

}