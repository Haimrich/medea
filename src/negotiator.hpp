#ifndef MEDEA_NEGOTIATOR_H_
#define MEDEA_NEGOTIATOR_H_

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"
#include "mapping/mapping.hpp"

#include "common.hpp"
#include "individual.hpp"
#include "accelergy.hpp"


namespace medea
{
  class MedeaNegotiator
  {
  public:

    struct MedeaMapping {
      MinimalArchSpecs arch;
      Mapping mapping;
      double area, energy, cycles;

      MedeaMapping() = default;
      MedeaMapping(config::CompoundConfig &config, model::Engine::Specs &arch_specs, problem::Workload &workload);
    };

  protected:

    config::CompoundConfig *config_;
    model::Engine::Specs default_arch_;

    std::string input_dir_;
    std::vector<unsigned> layer_workload_lookup_;
    std::string out_dir_;
    Accelergy &accelergy_;

    Orchestrator *thread_orchestrator_;
    std::mutex global_mutex_;

    std::vector<problem::Workload> workloads_;
    std::vector<std::vector<MedeaMapping>> workload_mappings_;

    unsigned num_threads_;

  public:

    MedeaNegotiator(config::CompoundConfig *config, std::string in_dir, std::vector<unsigned> lookup, std::string out_dir, Accelergy &accelergy);

    ~MedeaNegotiator();

    unsigned Run();

  protected:

  };
}

#endif // MEDEA_NEGOTIATOR_H_