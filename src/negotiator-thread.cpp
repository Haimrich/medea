#include "negotiator-thread.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <thread>

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"
#include "workload/workload.hpp"

#include "negotiator-individual.hpp"
#include "common.hpp"
#include "accelergy.hpp"

namespace medea {


  MedeaNegotiatorThread::MedeaNegotiatorThread(
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
  ) :
    thread_id_(thread_id),
    config_(config),
    default_arch_(default_arch),
    accelergy_(accelergy),
    thread_orchestrator_(thread_orchestrator),
    next_iteration_(1),
    num_threads_(num_threads),
    rng_(rng),
    layer_workload_lookup_(layer_workload_lookup),
    workloads_(workloads),
    workload_mappings_(workload_mappings),
    num_generations_(num_generations),
    population_size_(population_size),
    parent_population_(parent_population),
    population_(population),
    mutation_prob_(mutation_prob),
    individual_mutation_prob_(individual_mutation_prob),
    crossover_prob_(crossover_prob),
    individual_crossover_prob_(individual_crossover_prob)
  {
    num_layers_ = layer_workload_lookup_.size();
  }


  MedeaNegotiatorThread::~MedeaNegotiatorThread()
  {
  }


  void MedeaNegotiatorThread::Start()
  {
    thread_ = std::thread(&MedeaNegotiatorThread::Run, this);
  }


  void MedeaNegotiatorThread::Join()
  {
    thread_.join();
  }


  void MedeaNegotiatorThread::Run()
  {
    size_t slice_size = population_size_ / num_threads_;
    slice_size -= slice_size % 2;
    size_t pop_slice_start = thread_id_ * slice_size;
    size_t pop_slice_end = (thread_id_ == (num_threads_ - 1)) ? population_size_ : (pop_slice_start + slice_size);

    // Initial population.
    thread_orchestrator_->FollowerWait(next_iteration_);
    RandomPopulation(pop_slice_start, pop_slice_end, parent_population_);
    thread_orchestrator_->FollowerDone();

    // Wait for others
    thread_orchestrator_->FollowerWait(next_iteration_);
    
    std::uniform_real_distribution<double> dice(0, 1);

    for (size_t g = 0; g < num_generations_; g++)
    {
      for (size_t p = pop_slice_start; p < pop_slice_end; p += 2)
      {
        //if (use_tournament_)
        if (dice(*rng_) < crossover_prob_)
          Crossover(population_[p], population_[p+1]);

        if (dice(*rng_) < mutation_prob_) 
          MutateIndividual(population_[p]);
        if (dice(*rng_) < mutation_prob_)
          MutateIndividual(population_[p+1]);

        EvaluateIndividual(population_[p]);
        EvaluateIndividual(population_[p+1]);
      }
      std::cout << "#" << std::flush;
      thread_orchestrator_->FollowerDone();

      // Wait for others and merging in main
      thread_orchestrator_->FollowerWait(next_iteration_);
    }
  }


  NegotiatorIndividual MedeaNegotiatorThread::RandomIndividual()
  {
    NegotiatorIndividual out;
    out.mapping_id_set.reserve(num_layers_);
    out.mapping_evaluations.resize(num_layers_);

    for (size_t l = 0; l < num_layers_; l++)
    {
      size_t workload_id = layer_workload_lookup_[l];
      std::uniform_int_distribution<size_t> uni_dist(0, workload_mappings_[workload_id].size() - 1);
      out.mapping_id_set.push_back(uni_dist(*rng_));
      out.mapping_evaluations[l].need_evaluation = true;
    }

    return out;
  }


  void MedeaNegotiatorThread::RandomPopulation(size_t p, size_t end, NegotiatorPopulation& pop) 
  {
    for (; p < end; p++) {
      NegotiatorIndividual ind = RandomIndividual();
      EvaluateIndividual(ind);
      pop[p] = ind;   
    }
  }


  bool MedeaNegotiatorThread::NegotiateArchitecture(NegotiatorIndividual& ind)
  {
    auto old_negotiated_arch = ind.negotiated_arch;

    ind.negotiated_arch = workload_mappings_[layer_workload_lookup_[0]][ind.mapping_id_set[0]].arch;
    for (size_t l = 0; l < num_layers_; l++)
      ind.negotiated_arch &= workload_mappings_[layer_workload_lookup_[l]][ind.mapping_id_set[l]].arch;

    if (old_negotiated_arch == ind.negotiated_arch)
      return false;

    auto new_specs = model::Topology::Specs(default_arch_.topology);

    auto minimal_arithmetic = ind.negotiated_arch.GetLevel(0);
    auto arithmetic = new_specs.GetArithmeticLevel();
    arithmetic->meshX = minimal_arithmetic.mesh_x;
    arithmetic->meshY = minimal_arithmetic.mesh_y;
    arithmetic->instances = minimal_arithmetic.mesh_x * minimal_arithmetic.mesh_y;

    std::map<std::string, uint64_t> updates;

    for (unsigned i = 1; i < default_arch_.topology.NumLevels(); i++)
    {
      auto buffer = new_specs.GetStorageLevel(i - 1);
      if (!buffer->size.IsSpecified())
        continue;

      auto minimal_buffer = ind.negotiated_arch.GetLevel(i);
      buffer->meshX = minimal_buffer.mesh_x;
      buffer->meshY = minimal_buffer.mesh_y;
      buffer->instances = minimal_buffer.mesh_x * minimal_buffer.mesh_y;
      buffer->size = minimal_buffer.size;
      buffer->effective_size = static_cast<uint64_t>(std::floor(minimal_buffer.size / buffer->multiple_buffering.Get()));

      updates[buffer->name.Get()] = buffer->size.Get() / buffer->block_size.Get();
    }

    std::string out_prefix = "medea." + std::to_string(thread_id_) + "_tmp";
    Accelergy::RT rt = accelergy_.GetReferenceTables(updates, out_prefix);

    ind.negotiated_arch_specs.topology = new_specs;
    ind.negotiated_arch_specs.topology.ParseAccelergyART(rt.area);
    ind.negotiated_arch_specs.topology.ParseAccelergyERT(rt.energy);

    return true;
  }


  void MedeaNegotiatorThread::EvaluateIndividual(NegotiatorIndividual &ind)
  {
    bool arch_changed = NegotiateArchitecture(ind);
    model::Engine engine;
    engine.Spec(ind.negotiated_arch_specs);

    for (size_t i = 0; i < num_layers_; i++)
      if (ind.mapping_evaluations[i].need_evaluation || arch_changed)
      {
        auto status = engine.Evaluate(workload_mappings_[layer_workload_lookup_[i]][ind.mapping_id_set[i]].mapping, workloads_[layer_workload_lookup_[i]]);

        if (!engine.IsEvaluated())
        {
          std::cout << "Error in workload " << layer_workload_lookup_[i] << " with mapping ";
          std::cout << workload_mappings_[layer_workload_lookup_[i]][ind.mapping_id_set[i]].mapping.id << std::endl;

          for (auto &s : status)
            std::cout << s.success << " " << s.fail_reason << std::endl;
          YAML::Emitter yout;
          yout << ind.negotiated_arch;
          std::cout << yout.c_str() << std::endl;
          exit(1);
        }

        ind.mapping_evaluations[i].energy = engine.Energy();
        ind.mapping_evaluations[i].cycles = engine.Cycles();
        ind.mapping_evaluations[i].need_evaluation = false;
      }

    ind.objectives[0] = 0;
    ind.objectives[1] = 0;
    ind.objectives[2] = engine.Area();

    for (size_t i = 0; i < num_layers_; i++)
    {
      ind.objectives[0] += ind.mapping_evaluations[i].energy;
      ind.objectives[1] += ind.mapping_evaluations[i].cycles;
    }
  }


  void MedeaNegotiatorThread::MutateIndividual(NegotiatorIndividual &ind)
  {
    std::uniform_real_distribution<double> dice(0, 1);

    for (size_t l = 0; l < num_layers_; l++)
      if (dice(*rng_) < individual_mutation_prob_)
      {
        size_t workload_id = layer_workload_lookup_[l];
        std::uniform_int_distribution<size_t> uni_dist(0, workload_mappings_[workload_id].size() - 1);
        ind.mapping_id_set[l] = uni_dist(*rng_);
        ind.mapping_evaluations[l].need_evaluation = true;
      }
  }

  void MedeaNegotiatorThread::Crossover(NegotiatorIndividual &offspring_a, NegotiatorIndividual &offspring_b)
  {
    std::uniform_real_distribution<double> dice(0, 1);

    for (size_t l = 0; l < num_layers_; l++)
      if (dice(*rng_) < individual_crossover_prob_)
      {
        std::swap(offspring_a.mapping_id_set[l], offspring_b.mapping_id_set[l]);

        offspring_a.mapping_evaluations[l].need_evaluation = true;
        offspring_b.mapping_evaluations[l].need_evaluation = true;
      }
  }

}