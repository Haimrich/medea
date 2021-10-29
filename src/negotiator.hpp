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
    problem::Workload workload_;
    std::vector<MedeaMapping> mapping_set;
    MinimalArchSpecs negotiated_arch;
    double area, energy, cycles;

    std::mt19937* rng_;
    std::vector<model::Engine> set_engines_;
    std::vector<bool> need_evaluation_;
    model::Engine::Specs default_arch_specs_;
    model::Engine::Specs negotiated_arch_specs_;
    Accelergy &accelergy_;

    size_t num_layers_;

    bool NegotiateArchitecture();

    void UpdateEngineArchitecture();

  public:
    // Random Individual
    NegotiatorIndividual(problem::Workload &workload, const std::vector<std::vector<MedeaMapping>> &mappings, 
                         const std::vector<size_t> &lookup, model::Engine::Specs arch_specs, Accelergy &accelergy_);

    void Evaluate(bool evaluate_all = false);

    void Mutate();

    static void Crossover(const NegotiatorIndividual& parent_a, const NegotiatorIndividual& parent_b, NegotiatorIndividual& offspring_a, NegotiatorIndividual& offspring_b);

    Dominance CheckDominance(const NegotiatorIndividual& other);
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

    unsigned num_generations_;
    size_t population_size_;
    NegotiatorPopulation parent_population_, population_;

  public:

    MedeaNegotiator(config::CompoundConfig *config, std::string in_dir, std::vector<size_t> lookup, std::string out_dir, Accelergy &accelergy);

    ~MedeaNegotiator();

    unsigned Run();

  private:

    void Survival();

  };


}

#endif // MEDEA_NEGOTIATOR_H_