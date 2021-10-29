#include "mapper.hpp"

#include <iomanip>
#include <cmath>
#include <chrono>

#include "compound-config/compound-config.hpp"
#include "util/numeric.hpp"
#include "mapping/loop.hpp"
#include "mapping/parser.hpp"
#include "util/accelergy_interface.hpp"
#include "mapspaces/mapspace-factory.hpp"

#include "common.hpp"
#include "individual.hpp"

namespace medea
{

  MedeaMapper::MedeaMapper(config::CompoundConfig *config, std::string out_dir, Accelergy &accelergy) : config_(config),
                                                                      out_dir_(out_dir),
                                                                      accelergy_(accelergy),
                                                                      rng(rand_dev()),
                                                                      proba(0, 1)
  {
    auto rootNode = config->getRoot();

    // Problem configuration.
    auto problem = rootNode.lookup("problem");
    problem::ParseWorkload(problem, workload_);
    std::cout << "Problem configuration complete." << std::endl;

    // Architecture configuration.
    arch_config_ = rootNode.lookup("architecture");
    arch_specs_ = model::Engine::ParseSpecs(arch_config_);

    // Call accelergy ERT with all input files
    if (arch_config_.exists("subtree") || arch_config_.exists("local"))
    {
      accelergy::invokeAccelergy(config->inFiles, out_prefix_, out_dir_);
      std::string ertPath = out_dir_ + "/" + out_prefix_ + ".ERT.yaml";
      auto ertConfig = new config::CompoundConfig(ertPath.c_str());
      auto ert = ertConfig->getRoot().lookup("ERT");
      std::cout << "Generate Accelergy ERT (energy reference table) to replace internal energy model." << std::endl;
      arch_specs_.topology.ParseAccelergyERT(ert);

      std::string artPath = out_dir_ + "/" + out_prefix_ + ".ART.yaml";
      auto artConfig = new config::CompoundConfig(artPath.c_str());
      auto art = artConfig->getRoot().lookup("ART");
      std::cout << "Generate Accelergy ART (area reference table) to replace internal area model." << std::endl;
      arch_specs_.topology.ParseAccelergyART(art);
    }

    std::cout << "Architecture configuration complete." << std::endl;

    // MapSpace configuration.
    config::CompoundConfigNode arch_constraints;
    config::CompoundConfigNode mapspace;

    // Architecture constraints.
    if (arch_config_.exists("constraints"))
      arch_constraints = arch_config_.lookup("constraints");
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

    // Medea
    auto medea = rootNode.lookup("medea");

    num_generations_ = 30;
    medea.lookupValue("num-generations", num_generations_);
    population_size_ = 100;
    medea.lookupValue("population-size", population_size_);
    immigrant_population_size_ = 40;
    medea.lookupValue("immigrant-population-size", immigrant_population_size_);

    population_.resize(population_size_);
    parent_population_.resize(population_size_);
    immigrant_population_.resize(immigrant_population_size_);
    merged_population_.resize(2 * population_size_ + immigrant_population_size_);

    std::cout << "Num. generations: " << num_generations_ << " - Pop. size: " << population_size_ << " - Immigrant pop. size: " << immigrant_population_size_ << std::endl;

    fill_mutation_prob_ = 0.3;
    medea.lookupValue("fill-mutation-prob", fill_mutation_prob_);
    parallel_mutation_prob_ = 0.5;
    medea.lookupValue("parallel-mutation-prob", parallel_mutation_prob_);
    random_mutation_prob_ = 0.5;
    medea.lookupValue("random-mutation-prob", random_mutation_prob_);
    use_tournament_ = false;
    medea.lookupValue("use-tournament", use_tournament_);

    std::cout << std::setprecision(3) << "Fill Mut. Prob.: " << fill_mutation_prob_ << " - Parallel Mut. Prob.: " << parallel_mutation_prob_ << " - Random Mut. Prob.: " << random_mutation_prob_ << " - Using tournament: " << use_tournament_ << std::endl;

    num_threads_ = std::thread::hardware_concurrency();
    if (medea.lookupValue("num-threads", num_threads_))
      std::cout << "Using threads = " << num_threads_ << std::endl;
    else
      std::cout << "Using all available hardware threads = " << num_threads_ << std::endl;

    // Mapping injection
    user_mapping_defined_ = rootNode.exists("mapping");
    if (user_mapping_defined_)
    {
      auto mapping_conf = rootNode.lookup("mapping");
      user_mapping_ = mapping::ParseAndConstruct(mapping_conf, arch_specs_, workload_);
    }

    // Thread Orchestrator
    thread_orchestrator_ = new Orchestrator(num_threads_);

    // Random gen
    if_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::IndexFactorization));
    lp_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::LoopPermutation));
    db_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::DatatypeBypass));
    sp_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::Spatial));
    crossover_rng_ = new RandomGenerator128(10000);
  }

  MedeaMapper::~MedeaMapper()
  {
    if (mapspace_)
      delete mapspace_;
    if (constraints_)
      delete constraints_;

    if (thread_orchestrator_)
      delete thread_orchestrator_;

    if (if_rng_)
      delete if_rng_;
    if (lp_rng_)
      delete lp_rng_;
    if (db_rng_)
      delete db_rng_;
    if (sp_rng_)
      delete sp_rng_;

    if (crossover_rng_)
      delete crossover_rng_;
  }

  void MedeaMapper::Survival()
  {
    // Sort by rank and crowding distance and select population_size_
    std::partial_sort_copy(
        merged_population_.begin(), merged_population_.end(),
        parent_population_.begin(), parent_population_.end(),
        [&](const Individual &a, const Individual &b) -> bool
        {
          return a.rank < b.rank || (a.rank == b.rank && a.crowding_distance > b.crowding_distance);
        });

    // Shuffle
    if (!use_tournament_)
      std::shuffle(std::begin(parent_population_), std::end(parent_population_), rng);
  }

  Dominance MedeaMapper::CheckDominance(const Individual &a, const Individual &b)
  {
    bool all_a_less_or_equal_than_b = true;
    bool any_a_less_than_b = false;
    bool all_b_less_or_equal_than_a = true;
    bool any_b_less_than_a = false;

    for (unsigned i = 0; i < std::tuple_size<decltype(Individual::objectives)>::value; i++)
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

  void MedeaMapper::AssignCrowdingDistance(Population &population, std::vector<uint64_t> &pareto_front)
  {
    for (auto p : pareto_front)
      //population[p].crowding_distance = -std::accumulate(population[p].objectives.begin(), population[p].objectives.end(), 1, std::multiplies<double>());
      population[p].crowding_distance = 0.0;

    for (unsigned i = 0; i < std::tuple_size<decltype(Individual::objectives)>::value; i++)
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

  void MedeaMapper::AssignRankAndCrowdingDistance(Population &population)
  {
    std::vector<uint64_t> pareto_front;
    std::vector<std::vector<uint64_t>> dominated_by(population.size(), std::vector<uint64_t>());
    std::vector<uint64_t> num_dominating(population.size(), 0);

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

    uint64_t total_debug = 0;
    for (uint64_t f = 0; !pareto_front.empty(); f++)
    {
      total_debug += pareto_front.size();

      AssignCrowdingDistance(population, pareto_front);

      std::vector<uint64_t> new_pareto_front;

      for (uint64_t p : pareto_front)
      {

        for (uint64_t q : dominated_by[p])
        {

          num_dominating[q]--;

          if (num_dominating[q] == 0)
          {
            population[q].rank = f + 1;
            new_pareto_front.push_back(q);
          }
        }
      }
      pareto_front = new_pareto_front;
    }

    assert(total_debug == population.size());
  }

  void MedeaMapper::Merging()
  {
    std::copy(
        population_.begin(),
        population_.end(),
        merged_population_.begin());
    std::copy(
        parent_population_.begin(),
        parent_population_.end(),
        merged_population_.begin() + population_size_);
    std::copy(
        immigrant_population_.begin(),
        immigrant_population_.end(),
        merged_population_.begin() + 2 * population_size_);
  }

  void MedeaMapper::PrintGeneration(std::ofstream &out, Population &pop, uint64_t gen_id)
  {
    out << "GEN " << gen_id << std::endl;

    for (Individual &ind : pop)
    {
      out << ind.rank << "," << ind.crowding_distance << "," << ind.objectives[0] << "," << ind.objectives[1] << "," << ind.objectives[2] << std::endl;
    }

    out.flush();
  }

  void MedeaMapper::OutputParetoFrontFiles()
  {
    std::string dir = out_dir_ + "/pareto";
    int max_digits = std::to_string(population_size_).length();

    unsigned count = 1;
    for (auto &ind : parent_population_)
    {
      if (ind.rank)
        continue;

      std::string ind_id = std::to_string(count);

      std::string stats_filename = dir + "/" + out_prefix_ + ".stats." + std::string(max_digits - ind_id.length(), '0') + ind_id + ".txt";
      std::ofstream stats_file(stats_filename);
      stats_file << ind.engine << std::endl;
      stats_file.close();

      std::string statsy_filename = dir + "/" + out_prefix_ + ".stats." + std::string(max_digits - ind_id.length(), '0') + ind_id + ".yaml";
      std::ofstream statsy_file(statsy_filename);
      statsy_file << ind << std::endl;
      ArchProperties arch_props(arch_specs_);
      ind.genome.DumpYaml(statsy_file, arch_specs_.topology.StorageLevelNames(), arch_props);
      statsy_file.close();

      count++;
    }
  }


  void MedeaMapper::Run()
  {
    auto chrono_start = std::chrono::high_resolution_clock::now();
    // Output file names.
    const std::string stats_file_name = out_dir_ + "/medea.stats.txt";
    const std::string pop_txt_file_name = out_dir_ + "/medea.populations.txt";
    const std::string workload_file_name = out_dir_ + "/medea.workload.yaml";

    // Thread Start
    std::vector<MedeaMapperThread *> threads_;
    for (unsigned t = 0; t < num_threads_; t++)
    {
      threads_.push_back(
          new MedeaMapperThread(
              t,
              config_,
              out_dir_,
              workload_,
              arch_specs_,
              arch_config_,
              mapspace_,
              constraints_,
              immigrant_population_,
              parent_population_,
              population_,
              thread_orchestrator_,
              &global_mutex_,
              num_threads_,
              population_size_,
              immigrant_population_size_,
              num_generations_,
              fill_mutation_prob_,
              parallel_mutation_prob_,
              random_mutation_prob_,
              use_tournament_,
              accelergy_,
              user_mapping_,
              user_mapping_defined_,
              if_rng_,
              lp_rng_,
              db_rng_,
              sp_rng_,
              crossover_rng_));
    }

    for (unsigned t = 0; t < num_threads_; t++)
      threads_.at(t)->Start();

    thread_orchestrator_->LeaderDone();
    thread_orchestrator_->LeaderWait();

    std::cout << "[INFO] Initial Population Done." << std::endl;
    AssignRankAndCrowdingDistance(parent_population_);

    thread_orchestrator_->LeaderDone();

    std::ofstream population_file(pop_txt_file_name);

    for (uint32_t g = 0; g < num_generations_; g++)
    {

      // Wait Crossover, Mutation and Immigration
      thread_orchestrator_->LeaderWait();

      // Select for next generation
      Merging();
      AssignRankAndCrowdingDistance(merged_population_);
      Survival();

      std::cout << "[RANKS] ";
      for (Individual &ind : parent_population_)
      {
        if (ind.rank < 10)
          std::cout << ind.rank;
        else if (ind.rank < 36)
          std::cout << (char)(ind.rank + 55);
        else if (ind.rank < 61)
          std::cout << (char)(ind.rank + 61);
        else
          std::cout << "+";
      }
      std::cout << std::endl;

      std::cout << "[INFO] Generation " << g << " done." << std::endl;

      PrintGeneration(population_file, parent_population_, g);
      thread_orchestrator_->LeaderDone();
    }

    // ============
    // Termination.
    // ============
    population_file.close();

    auto chrono_end = std::chrono::high_resolution_clock::now();
    auto chrono_duration = std::chrono::duration_cast<std::chrono::seconds>(chrono_end - chrono_start).count();

    std::ofstream stats_file(stats_file_name);
    stats_file << "Search time: " << chrono_duration << " seconds" << std::endl;
    stats_file.close();

    std::ofstream workload_file(workload_file_name);
    workload_file << config_->getYConfig()["problem"] << std::endl;
    workload_file.close();

    std::cout << std::endl;
    std::cout << "Search time: " << chrono_duration << " seconds" << std::endl;

    OutputParetoFrontFiles();
  }

};
