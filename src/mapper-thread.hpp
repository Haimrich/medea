#ifndef MEDEA_MAPPER_THREAD_H_
#define MEDEA_MAPPER_THREAD_H_

#include "mapspaces/mapspace-base.hpp"
#include "mapping/constraints.hpp"

#include "common.hpp"
#include "individual.hpp"
#include "accelergy.hpp"

namespace medea
{

  class MedeaMapperThread
  {

  private:
    unsigned thread_id_;

    config::CompoundConfig *config_;

    problem::Workload &workload_;
    model::Engine::Specs arch_specs_;
    config::CompoundConfigNode arch_config_;
    ArchProperties arch_props_;
    mapspace::MapSpace *mapspace_;
    mapping::Constraints *constraints_;

    Population &immigrant_population_, &parent_population_, &population_;

    Orchestrator *thread_orchestrator_;
    uint64_t next_iteration_ = 1;
    std::mutex *global_mutex_;
    uint32_t num_threads_;

    uint32_t population_size_;
    uint32_t immigrant_population_size_;
    uint32_t num_generations_;

    double fill_mutation_prob_, parallel_mutation_prob_, random_mutation_prob_;
    bool use_tournament_, update_ert_;
    Accelergy& accelergy_;

    Mapping user_mapping_;
    bool user_mapping_defined_;
    uint32_t inj_gen_;

    std::thread thread_;

    RandomGenerator128 *if_rng_, *lp_rng_, *db_rng_, *sp_rng_, *crossover_rng_;
    std::default_random_engine rng_;
    std::exponential_distribution<double> exp_distribution_;
    std::uniform_real_distribution<double> uni_distribution_;
    std::uniform_int_distribution<uint64_t> tour_distribution_;

  protected:

    uint64_t gcd(uint64_t a, uint64_t b);

    bool EngineSuccess(std::vector<model::EvalStatus> &status_per_level);

    bool RandomMapping(Mapping *mapping);

    LoopRange GetSubnestRangeAtLevel(const Mapping &mapping, unsigned level);

    uint64_t GetParallelAtLevel(const Mapping &mapping, spacetime::Dimension dim, uint64_t level);

    std::vector<loop::Descriptor> GetSubnestAtLevel(const Mapping &mapping, unsigned level);

    uint64_t GetDimFactorInSubnest(problem::Shape::DimensionID dimension, std::vector<loop::Descriptor> &subnest);

    uint64_t GetStrideInSubnest(problem::Shape::DimensionID dimension, std::vector<loop::Descriptor> &subnest);

    void UpdateArchitecture(Mapping& mapping, model::Engine& engine, MinimalArchSpecs& arch);

    bool Evaluate(Mapping mapping, Individual &individual);

    void FactorCompensation(const problem::Shape::DimensionID &dim, const uint64_t stride, const uint64_t old_factor, const uint64_t new_factor, const uint64_t level, loop::Nest &nest);

    void Crossover(const Mapping &parent_a, const Mapping &parent_b, Mapping &offspring_a, Mapping &offspring_b);

    void FanoutMutation(Mapping &mapping);

    // Fill buffer at lower levels - Funziona solo con quelli che contengono un solo datatype per ora forse
    void FillMutation(model::Engine &engine, Mapping &mapping);

    void RandomMutation(Mapping &mapping);

    void Mutation(Individual &individual);

    void RandomIndividual(uint32_t p, Population &population);

    void InjectUserDefinedMapping(Population &pop, uint32_t id);

    void RandomPopulation(uint32_t p, uint32_t pop_slice_end, Population &population);

    uint64_t Tournament();

  public:
    MedeaMapperThread(
        unsigned thread_id,
        config::CompoundConfig *config,
        problem::Workload &workload,
        model::Engine::Specs arch_specs,
        config::CompoundConfigNode arch_config,
        mapspace::MapSpace *mapspace,
        mapping::Constraints *constraints,
        std::vector<Individual> &immigrant_population,
        std::vector<Individual> &parent_population,
        std::vector<Individual> &population,
        Orchestrator *thread_orchestrator,
        std::mutex *global_mutex,
        uint32_t num_threads,
        uint32_t population_size,
        uint32_t immigrant_population_size,
        uint32_t generations,
        double fill_mutation_prob,
        double parallel_mutation_prob,
        double random_mutation_prob,
        bool use_tournament,
        bool update_ert,
        Accelergy &accelergy,
        Mapping user_mapping,
        bool user_mapping_defined,
        RandomGenerator128 *if_rng,
        RandomGenerator128 *lp_rng,
        RandomGenerator128 *db_rng,
        RandomGenerator128 *sp_rng,
        RandomGenerator128 *crossover_rng);

    ~MedeaMapperThread();

    void Start();

    void Join();

    void Run();

  };

}

#endif // MEDEA_MAPPER_THREAD_H_