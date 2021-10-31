#ifndef MEDEA_NEGOTIATOR_INDIVIDUAL_H_
#define MEDEA_NEGOTIATOR_INDIVIDUAL_H_

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"
#include "workload/workload.hpp"

#include "individual.hpp"
#include "mapping.hpp"

namespace medea {

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

}

#endif // MEDEA_NEGOTIATOR_INDIVIDUAL_H_