#ifndef MEDEA_NEGOTIATOR_H_
#define MEDEA_NEGOTIATOR_H_

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"
#include "mapping.hpp"

#include "common.hpp"
#include "individual.hpp"
#include "accelergy.hpp"


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


  class NegotiatorIndividual 
  {
  private:
    std::vector<MedeaMapping> mapping_set_;
    MinimalArchSpecs negotiated_arch_;

    std::mt19937* rng_;
    std::vector<model::Engine> set_engines_;
    std::vector<bool> need_evaluation_;
    model::Engine::Specs default_arch_specs_;
    model::Engine::Specs negotiated_arch_specs_;
    Accelergy* accelergy_;

    size_t num_layers_;

    bool NegotiateArchitecture();

  public:

    unsigned rank;
    double crowding_distance;

    double objectives[3];
    double& energy = objectives[0];
    double& cycles = objectives[1];
    double& area = objectives[2];

    // Random Individual    
    NegotiatorIndividual(std::mt19937* rng, std::vector<problem::Workload> &workloads, const std::vector<size_t> &lookup, 
                         const std::vector<std::vector<MedeaMapping>> &mappings, model::Engine::Specs arch_specs, Accelergy *accelergy);

    NegotiatorIndividual() = default;

    void Evaluate(std::vector<problem::Workload> &workloads, const std::vector<size_t> &lookup);

    void Mutate(double mutation_prob,  const std::vector<std::vector<MedeaMapping>> &mappings, const std::vector<size_t> &lookup);

    static void Crossover(double crossover_prob, NegotiatorIndividual& offspring_a, NegotiatorIndividual& offspring_b);

    Dominance CheckDominance(const NegotiatorIndividual& other);

    NegotiatorIndividual& operator=(const NegotiatorIndividual& m);
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

  public:

    MedeaNegotiator(config::CompoundConfig *config, std::string in_dir, std::vector<size_t> lookup, std::string out_dir, Accelergy &accelergy);

    ~MedeaNegotiator();

    unsigned Run();

  private:

    void AssignRankAndCrowdingDistance(NegotiatorPopulation &population);

    void AssignCrowdingDistance(NegotiatorPopulation &population, std::vector<size_t> &pareto_front);

    void Merging();

    void Survival();

  };


}

#endif // MEDEA_NEGOTIATOR_H_