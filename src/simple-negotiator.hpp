#ifndef MEDEA_SIMPLE_NEGOTIATOR_H_
#define MEDEA_SIMPLE_NEGOTIATOR_H_

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"

#include "common.hpp"
#include "accelergy.hpp"
#include "mapping.hpp"

namespace medea
{

  class MedeaSimpleNegotiator
  {

  protected:

    config::CompoundConfig *config_;
    model::Engine::Specs default_arch_;

    std::string workload_dir_;
    std::string mapping_dir_;
    std::vector<size_t> layer_workload_lookup_;
    std::string out_dir_;
    Accelergy &accelergy_;

    size_t num_layers_;
    size_t num_workloads_;

    std::vector<problem::Workload> workloads_;
    std::vector<Mapping> mappings_;
    std::vector<model::Engine> engines_;
    model::Engine global_engine_;

  public:

    MedeaSimpleNegotiator(
      config::CompoundConfig *config, 
      std::string workload_dir, 
      std::string mapping_dir,
      std::vector<size_t> lookup, 
      std::string out_dir, 
      Accelergy &accelergy
    );

    ~MedeaSimpleNegotiator();

    void Run();

  private:

    model::Engine::Specs NegotiateArchitecture();

    // TODO: Move these type of methods in Mapping extended class
    uint64_t GetParallelAtLevel(const Mapping &mapping, spacetime::Dimension dim, uint64_t level);

  };


}

#endif // MEDEA_SIMPLE_NEGOTIATOR_H_