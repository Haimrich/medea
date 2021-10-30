#ifndef MEDEA_NEGOTIATOR_H_
#define MEDEA_NEGOTIATOR_H_

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"

#include "common.hpp"
#include "individual.hpp"
#include "accelergy.hpp"
#include "mapping.hpp"


namespace medea
{

  class MedeaMapping
  {
  public:
    unsigned id = 0;
    MinimalArchSpecs arch;
    Mapping mapping;
    double area, energy, cycles;

    MedeaMapping() = default;
    MedeaMapping(unsigned id, config::CompoundConfig &config, model::Engine::Specs &arch_specs, problem::Workload &workload);
  };

  struct NegotiatorIndividual
  {

    struct Eval
    {
      double energy, cycles;
      bool need_evaluation;
    };

    std::vector<size_t> mapping_id_set;
    std::vector<Eval> mapping_evaluations;

    MinimalArchSpecs negotiated_arch;
    model::Engine::Specs negotiated_arch_specs;

    unsigned rank;
    double crowding_distance;
    std::array<double, 3> objectives; // Energy, Cycles, Area
  };

  typedef std::vector<NegotiatorIndividual> NegotiatorPopulation;

  class MedeaNegotiator
  {

  protected:

    config::CompoundConfig *config_;
    model::Engine::Specs default_arch_;

    std::string input_dir_;
    std::vector<size_t> layer_workload_lookup_;
    std::string out_dir_;
    Accelergy &accelergy_;

    Orchestrator *thread_orchestrator_;
    std::mutex global_mutex_;

    std::vector<problem::Workload> workloads_;
    std::vector<std::vector<MedeaMapping>> workload_mappings_;
    
    unsigned num_threads_;
    std::mt19937* rng_;

    unsigned num_generations_;
    size_t population_size_;
    NegotiatorPopulation parent_population_, population_, merged_population_;

    double mutation_prob_, individual_mutation_prob_;
    double crossover_prob_, individual_crossover_prob_;

    size_t num_layers_;

  public:

    MedeaNegotiator(config::CompoundConfig *config, std::string in_dir, std::vector<size_t> lookup, std::string out_dir, Accelergy &accelergy);

    ~MedeaNegotiator();

    unsigned Run();

  private:

    void AssignRankAndCrowdingDistance(NegotiatorPopulation &population);

    void AssignCrowdingDistance(NegotiatorPopulation &population, std::vector<size_t> &pareto_front);

    void Merging();

    void Survival();

    unsigned OutputParetoFrontFiles();

  private:
  
    NegotiatorIndividual RandomIndividual();

    void EvaluateIndividual(NegotiatorIndividual &individual);

    bool NegotiateArchitecture(NegotiatorIndividual &individual);

    void MutateIndividual(NegotiatorIndividual &individual);

    void Crossover(NegotiatorIndividual &offspring_a, NegotiatorIndividual &offspring_b);

    Dominance CheckDominance(const NegotiatorIndividual &a, const NegotiatorIndividual &b);

  };


}

#endif // MEDEA_NEGOTIATOR_H_