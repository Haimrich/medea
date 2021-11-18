#ifndef MEDEA_NEGOTIATOR_H_
#define MEDEA_NEGOTIATOR_H_

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"

#include "common.hpp"
#include "individual.hpp"
#include "accelergy.hpp"
#include "mapping.hpp"
#include "negotiator-individual.hpp"

namespace medea
{

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
    double pareto_sieving_;

    size_t num_layers_;

  public:

    MedeaNegotiator(config::CompoundConfig *config, std::string in_dir, std::vector<size_t> lookup, std::string out_dir, Accelergy &accelergy);

    ~MedeaNegotiator();

    unsigned Run();

  private:

    void InjectSingleObjectiveBestIndividuals(NegotiatorPopulation& population);

    Dominance CheckDominance(const NegotiatorIndividual &a, const NegotiatorIndividual &b);

    void AssignCrowdingDistance(NegotiatorPopulation &population, std::vector<size_t> &pareto_front);

    void AssignRankAndCrowdingDistance(NegotiatorPopulation &population);

    void Merging();

    void Survival();

    unsigned OutputParetoFrontFiles();

  };


}

#endif // MEDEA_NEGOTIATOR_H_