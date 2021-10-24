#ifndef MEDEA_MAPPER_H_
#define MEDEA_MAPPER_H_

#include <sys/stat.h>
#include <errno.h>

#include <iomanip>
#include <cmath>

#include "compound-config/compound-config.hpp"
#include "util/numeric.hpp"
#include "mapping/loop.hpp"
#include "mapping/parser.hpp"
#include "mapping/constraints.hpp"
#include "mapspaces/mapspace-base.hpp"

#include "medea-common.hpp"
#include "medea-mapper-thread.hpp"

namespace medea
{

  class Medea
  {
  protected:
    config::CompoundConfig *config_;
    problem::Workload workload_;
    model::Engine::Specs arch_specs_;
    config::CompoundConfigNode arch_config_;
    mapspace::MapSpace *mapspace_;
    mapping::Constraints *constraints_;

    std::string out_dir_ = ".";
    std::string out_prefix_ = "medea";

    uint32_t num_generations_;
    uint32_t population_size_;
    uint32_t immigrant_population_size_;

    Population population_, parent_population_, immigrant_population_, merged_population_;

    double fill_mutation_prob_, parallel_mutation_prob_, random_mutation_prob_;
    std::string fast_accelergy_path_;
    bool use_tournament_;

    Orchestrator *thread_orchestrator_;
    std::mutex best_mutex_;

    Individual best_individual_;

    uint32_t num_threads_;

    RandomGenerator128 *if_rng_, *lp_rng_, *db_rng_, *sp_rng_, *crossover_rng_;

    std::random_device rand_dev;
    std::mt19937_64 rng;
    //std::default_random_engine rng;
    std::uniform_real_distribution<> proba;

    enum Dominance
    {
      DOMINATING,
      DOMINATED,
      FRONTIER
    };

    Mapping user_mapping_;
    bool user_mapping_defined_;

  public:
  
    Medea(config::CompoundConfig *config, std::string out_dir);

    ~Medea();

    void Run();

  protected:

    void Survival();

    Dominance CheckDominance(const Individual &a, const Individual &b);

    void AssignCrowdingDistance(Population &population, std::vector<uint64_t> &pareto_front);

    void AssignRankAndCrowdingDistance(Population &population);

    void Merging();

    void PrintGeneration(std::ofstream &out, Population &pop, uint64_t gen_id);

    void OutputParetoFrontStats();
  };

}

#endif // MEDEA_MAPPER_H_