#ifndef MEDEA_NEGOTIATOR_THREAD_H_
#define MEDEA_NEGOTIATOR_THREAD_H_

#include <vector>
#include <random>

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"
#include "workload/workload.hpp"

#include "common.hpp"
#include "accelergy.hpp"
#include "negotiator-individual.hpp"

namespace medea {

  class MedeaNegotiatorThread
  {

  protected:
    unsigned thread_id_;

    config::CompoundConfig *config_;
    model::Engine::Specs default_arch_;

    Accelergy &accelergy_;

    Orchestrator *thread_orchestrator_;
    uint64_t next_iteration_;
    unsigned num_threads_;
    std::mt19937* rng_;

    size_t num_layers_;
    std::vector<size_t> &layer_workload_lookup_;
    std::vector<problem::Workload> &workloads_;
    std::vector<std::vector<MedeaMapping>> &workload_mappings_;

    unsigned num_generations_;
    size_t population_size_;
    NegotiatorPopulation &parent_population_;
    NegotiatorPopulation &population_;

    double mutation_prob_, individual_mutation_prob_;
    double crossover_prob_, individual_crossover_prob_;

    std::thread thread_;

  public:

    MedeaNegotiatorThread(
      unsigned thread_id,
      config::CompoundConfig *config, 
      model::Engine::Specs default_arch,
      Accelergy &accelergy,
      Orchestrator *thread_orchestrator,
      unsigned num_threads,
      std::mt19937 *rng,
      std::vector<size_t> &layer_workload_lookup,
      std::vector<problem::Workload> &workloads,
      std::vector<std::vector<MedeaMapping>> &workload_mappings,
      unsigned num_generations,
      size_t population_size,
      NegotiatorPopulation &parent_population,
      NegotiatorPopulation &population,
      double mutation_prob,
      double individual_mutation_prob,
      double crossover_prob,
      double individual_crossover_prob
    );

    ~MedeaNegotiatorThread();
  
    void Start();

    void Join();

    void Run();

  private:
  
    NegotiatorIndividual RandomIndividual();

    void RandomPopulation(size_t p, size_t end, NegotiatorPopulation &pop);

    void EvaluateIndividual(NegotiatorIndividual &individual);

    bool NegotiateArchitecture(NegotiatorIndividual &individual);

    void MutateIndividual(NegotiatorIndividual &individual);

    void Crossover(NegotiatorIndividual &offspring_a, NegotiatorIndividual &offspring_b);

  };
}

#endif // MEDEA_NEGOTIATOR_THREAD_H_