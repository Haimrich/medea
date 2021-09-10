/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "applications/medea/medea-common.hpp"
#include "applications/medea/medea-accelergy.hpp"

extern bool gTerminate;

namespace medea {

class MedeaThread
{
public:
  
private:
  unsigned thread_id_;

  config::CompoundConfig* config_;
  std::string out_dir_;

  problem::Workload &workload_;
  model::Engine::Specs arch_specs_;
  config::CompoundConfigNode arch_config_;
  ArchProperties arch_props_;
  mapspace::MapSpace* mapspace_;
  mapping::Constraints* constraints_;

  Individual* global_best_individual_;
  Population &immigrant_population_, &parent_population_, &population_;

  Orchestrator* thread_orchestrator_;
  uint64_t next_iteration_ = 1;
  std::mutex* global_mutex_;
  uint32_t num_threads_;

  uint32_t population_size_;
  uint32_t immigrant_population_size_;
  uint32_t num_generations_;

  double fill_mutation_prob_, parallel_mutation_prob_, random_mutation_prob_;
  bool use_tournament_;

  Mapping user_mapping_;
  bool user_mapping_defined_;

  std::thread thread_;

  RandomGenerator128 *if_rng_, *lp_rng_, *db_rng_, *sp_rng_, *crossover_rng_;
  std::default_random_engine rng_;
  std::exponential_distribution<double> exp_distribution_;
  std::uniform_real_distribution<double> uni_distribution_;
  std::uniform_int_distribution<uint64_t> tour_distribution_;

  Individual best_individual_;
  
 protected:


  uint64_t gcd(uint64_t a, uint64_t b)
  {
    if (b == 0) return a;
    return gcd(b, a % b);
  }

  double Fitness(model::Engine engine) {
    auto stats = engine.GetTopology().GetStats();
    return - stats.energy * stats.cycles;
  }

  bool EngineSuccess(std::vector<model::EvalStatus>& status_per_level) {
    return std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                [](bool cur, const model::EvalStatus& status)
                                { return cur && status.success; });
  }

  bool RandomMapping(Mapping* mapping) {
      // Prepare a new mapping ID.
      std::unique_lock<std::mutex> lock(*global_mutex_);
      mapspace::ID mapping_id = mapspace::ID(mapspace_->AllSizes());

      mapping_id.Set(unsigned(mapspace::Dimension::DatatypeBypass), db_rng_->Next() % mapspace_->Size(mapspace::Dimension::DatatypeBypass));
      mapping_id.Set(unsigned(mapspace::Dimension::Spatial), sp_rng_->Next() % mapspace_->Size(mapspace::Dimension::Spatial));
      mapping_id.Set(unsigned(mapspace::Dimension::LoopPermutation), lp_rng_->Next() % mapspace_->Size(mapspace::Dimension::LoopPermutation));

      uint128_t if_id = if_rng_->Next() % mapspace_->Size(mapspace::Dimension::IndexFactorization);
      mapping_id.Set(unsigned(mapspace::Dimension::IndexFactorization), if_id);
      mapspace_->InitPruned(if_id);

      auto construction_status = mapspace_->ConstructMapping(mapping_id, mapping);

      return std::accumulate(construction_status.begin(), construction_status.end(), true,
                                  [](bool cur, const mapspace::Status& status)
                                  { return cur && status.success; });
  }

  uint64_t GetSpatialSpread(const Mapping& mapping, spacetime::Dimension dim, uint64_t level) {
    uint64_t result = 1;
    
    uint64_t start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level-1) + 1 : 0;
    uint64_t end = mapping.loop_nest.storage_tiling_boundaries.at(level) + 1;

    for (auto it = mapping.loop_nest.loops.begin() + start; it != mapping.loop_nest.loops.begin() + end; it++)
      if (it->spacetime_dimension == dim)
        result *= it->end;

    return result;
  }

  void UpdateArchitecture(const Mapping& mapping, model::Engine& engine) {
    // Update area and revaluate sigh
    
    //YAML::Node root_node = arch_config_.getYNode();
    std::map<std::string, uint64_t> updates;

    unsigned buffer_update_granularity = 16; // This should be configurable FIXME
    
    
    auto new_specs = model::Topology::Specs(arch_specs_.topology);
    for (unsigned i = 0; i < arch_specs_.topology.NumStorageLevels(); i++) {
      auto buffer = new_specs.GetStorageLevel(i);

      auto tile_sizes = engine.GetTopology().GetStats().tile_sizes.at(i);
      auto utilized_capacity = std::accumulate(tile_sizes.begin(), tile_sizes.end(), 0);

      if (!buffer->block_size.IsSpecified()) continue;
      auto block_size = buffer->block_size.Get();

      if (!buffer->size.IsSpecified()) continue;

      unsigned needed_depth = (utilized_capacity / block_size) + 1;
      unsigned remainder = needed_depth % buffer_update_granularity;
      unsigned new_depth = remainder ? needed_depth + buffer_update_granularity - remainder : needed_depth;
      
      buffer->size = new_depth * block_size;
      buffer->effective_size = static_cast<uint64_t>(std::floor(buffer->size.Get() / buffer->multiple_buffering.Get()));
      updates[buffer->name.Get()] = new_depth;
    }

    for (int i = arch_specs_.topology.NumLevels() - 2; i >= 0; i--)
    {
      if (i == 0)
      {
        auto arithmetic = new_specs.GetArithmeticLevel();

        arithmetic->meshX = GetSpatialSpread(mapping, spacetime::Dimension::SpaceX, i) * new_specs.GetStorageLevel(i)->meshX.Get();
        arithmetic->meshY = GetSpatialSpread(mapping, spacetime::Dimension::SpaceY, i) * new_specs.GetStorageLevel(i)->meshY.Get();
        arithmetic->instances = arithmetic->meshX.Get() * arithmetic->meshY.Get();
      }
      else
      {
        auto buffer = new_specs.GetStorageLevel(i-1);

        buffer->meshX = GetSpatialSpread(mapping, spacetime::Dimension::SpaceX, i) * new_specs.GetStorageLevel(i)->meshX.Get();
        buffer->meshY = GetSpatialSpread(mapping, spacetime::Dimension::SpaceY, i) * new_specs.GetStorageLevel(i)->meshY.Get();
        buffer->instances = buffer->meshX.Get() * buffer->meshY.Get();
      }
    }

    std::string out_prefix = "medea." + std::to_string(thread_id_) + "_tmp";
    auto art = getARTfromAccelergy(config_, updates, out_prefix);
   
    model::Engine::Specs new_engine_specs;
    new_engine_specs.topology = new_specs;
    new_engine_specs.topology.ParseAccelergyART(art);
    engine.Spec(new_engine_specs);
 
      /*
      YAML::Node node = root_node;
      while (node["subtree"]) {
        node = node["subtree"][0];
        auto local_node = node["local"];

        if (local_node) 
          for (auto&& buffer : local_node) 
            if (buffer["name"]) {
              auto buffer_name = buffer["name"].as<std::string>();
              if (buffer_name.compare(0, storage_name.size(), storage_name) == 0 && buffer["attributes"])
                if (buffer["attributes"]["depth"])
                  buffer["attributes"]["depth"] = 
            }
          
      }
      */
    
  }

  bool Evaluate(Mapping mapping, Individual& individual) {
    model::Engine engine;
    engine.Spec(arch_specs_);
    
    // Lightweight pre-eval
    auto status_per_level = engine.PreEvaluationCheck(mapping, workload_);
    if (!EngineSuccess(status_per_level))
      return false;

    // Heavyweight evaluation
    status_per_level = engine.Evaluate(mapping, workload_);
    if (!EngineSuccess(status_per_level)) 
      return false;

    // Update storage capacities based on mappping -> update area
    UpdateArchitecture(mapping, engine);
    status_per_level = engine.Evaluate(mapping, workload_);
    assert( EngineSuccess(status_per_level) );

    // Population update
    individual.genome = mapping;
    individual.objectives[0] = engine.Energy();
    individual.objectives[1] = (double) engine.Cycles();
    individual.objectives[2] = engine.Area();
    individual.fitness = Fitness(engine);
    individual.engine = engine;

    // Best update
    if (!best_individual_.engine.IsSpecced() || individual.fitness > best_individual_.fitness) {
      best_individual_ = individual;     
      assert( best_individual_.engine.GetTopology().MACCs() == individual.engine.GetTopology().MACCs() );
    }

    return true;
  
  }

  uint64_t GetDimensionAtLevel(problem::Shape::DimensionID dimension, std::vector<loop::Descriptor>& subnest) {
    uint64_t factor = 1;
    for (auto &l : subnest) 
      if (l.dimension == dimension)
        factor *= l.end;
    return factor;
  }

  uint64_t GetStrideAtLevel(problem::Shape::DimensionID dimension, std::vector<loop::Descriptor>& subnest) {
    for (auto &l : subnest) 
      if (l.dimension == dimension)
        return l.stride;
    
    return 1;
  }

  std::vector<loop::Descriptor> GetNestAtLevel(const Mapping& mapping, unsigned level) {
    loop::Nest nest = mapping.loop_nest;
    uint64_t start = level > 0 ? nest.storage_tiling_boundaries.at(level-1) + 1 : 0;
    uint64_t end = nest.storage_tiling_boundaries.at(level) + 1;
    std::vector<loop::Descriptor> level_nest(nest.loops.begin() + start, nest.loops.begin() + end); 
    
    return level_nest;
  }

  void FactorCompensation(const problem::Shape::DimensionID& dim, const uint64_t stride, const uint64_t old_factor, const uint64_t new_factor, const uint64_t level, loop::Nest& nest) {
    
    if (new_factor < old_factor) {
        // Prima passare da old_factor a 1 poi da 1 a new_factor -> ricorsivo
        if (old_factor % new_factor) {
          FactorCompensation(dim, stride, old_factor, 1, level, nest);
          FactorCompensation(dim, stride, 1, new_factor, level, nest);
          return;
        }
        
        // Fattore diminuito -> compensiamo aumentando in RAM.
        int64_t factor = old_factor / new_factor;

        int64_t ram_level = nest.storage_tiling_boundaries.size() - 2;
        uint64_t ram_start = ram_level > 0 ? nest.storage_tiling_boundaries.at(ram_level) + 1 : 0;

        auto ram_loop = std::find_if(nest.loops.begin() + ram_start, nest.loops.end(), [&](const loop::Descriptor& x) { return x.dimension == dim;});

        if (ram_loop != nest.loops.end()) {
          ram_loop->end *= factor;
        } else {
          loop::Descriptor new_loop(dim, 0, factor, stride, spacetime::Dimension::Time);
          nest.loops.push_back(new_loop);
          nest.storage_tiling_boundaries.back()++;
        }


      } else if (new_factor > old_factor) {
        // Fattore aumentato -> Compensiamo diminuendo in RAM o nel primo che troviamo a scendere
        if (new_factor % old_factor) {
          FactorCompensation(dim, stride, old_factor, 1, level, nest);
          FactorCompensation(dim, stride, 1, new_factor, level, nest);
          return;
        }

        int64_t factor = new_factor / old_factor;

        for (int64_t l = nest.storage_tiling_boundaries.size() - 1; l >= 0 && factor != 1; l--) {
          // Cerca fattore da ridurre (escluso livello in cui l'abbiamo incrementato)

          if (l != (int64_t)level) {
            uint64_t l_start = l > 0 ? nest.storage_tiling_boundaries.at(l-1) + 1 : 0;
            uint64_t l_end = nest.storage_tiling_boundaries.at(l) + 1;

            for (auto l_loop = nest.loops.begin() + l_start; l_loop != nest.loops.begin() + l_end && factor != 1; l_loop++) {
              if (l_loop->dimension == dim) {
                uint64_t common = gcd(factor, l_loop->end);
    
                factor /= common;
                l_loop->end /= common;
              } 
            }
          }
        }
      }

  }

  void Crossover(const Mapping& parent_a, const Mapping& parent_b, Mapping& offspring_a, Mapping& offspring_b) {
    global_mutex_->lock();
    offspring_a = parent_a;
    offspring_b = parent_b;
    
    uint64_t level = crossover_rng_->Next().convert_to<uint64_t>() % (parent_a.loop_nest.storage_tiling_boundaries.size() - 1);
    global_mutex_->unlock();

    loop::Nest nest_a = parent_a.loop_nest;
    uint64_t a_start = level > 0 ? nest_a.storage_tiling_boundaries.at(level-1) + 1 : 0;
    uint64_t a_end = nest_a.storage_tiling_boundaries.at(level) + 1;
    std::vector<loop::Descriptor> a_level(nest_a.loops.begin() + a_start, nest_a.loops.begin() + a_end);
    
    loop::Nest nest_b = parent_b.loop_nest;
    uint64_t b_start = level > 0 ? nest_b.storage_tiling_boundaries.at(level-1) + 1 : 0;
    uint64_t b_end = nest_b.storage_tiling_boundaries.at(level) + 1;
    std::vector<loop::Descriptor> b_level(nest_b.loops.begin() + b_start, nest_b.loops.begin() + b_end); 

    // Factor compensation
    for (int idim = 0; idim < int(problem::GetShape()->NumDimensions); idim++)
    {
      problem::Shape::DimensionID dimension = problem::Shape::DimensionID(idim);

      uint64_t factor_a = GetDimensionAtLevel(dimension, a_level);
      uint64_t factor_b = GetDimensionAtLevel(dimension, b_level);
      uint64_t stride_a = GetStrideAtLevel(dimension, a_level);
      uint64_t stride_b = GetStrideAtLevel(dimension, b_level);

      FactorCompensation(dimension, stride_a, factor_a, factor_b, level, offspring_a.loop_nest);
      FactorCompensation(dimension, stride_b, factor_b, factor_a, level, offspring_b.loop_nest);
    }

    a_start = level > 0 ? parent_a.loop_nest.storage_tiling_boundaries.at(level-1) + 1 : 0;
    a_end = parent_a.loop_nest.storage_tiling_boundaries.at(level) + 1;
    b_start = level > 0 ? parent_b.loop_nest.storage_tiling_boundaries.at(level-1) + 1 : 0;
    b_end = parent_b.loop_nest.storage_tiling_boundaries.at(level) + 1;

    // Elimino vecchi loop da a
    offspring_a.loop_nest.loops.erase(offspring_a.loop_nest.loops.begin() + a_start, offspring_a.loop_nest.loops.begin() + a_end);
    offspring_a.loop_nest.loops.insert(offspring_a.loop_nest.loops.begin() + a_start, b_level.begin(), b_level.end());

    // Substituting loops in B
    offspring_b.loop_nest.loops.erase(offspring_b.loop_nest.loops.begin() + b_start,  offspring_b.loop_nest.loops.begin() + b_end);
    offspring_b.loop_nest.loops.insert(offspring_b.loop_nest.loops.begin() + b_start, a_level.begin(), a_level.end());
 

    int64_t diff = a_level.size() - b_level.size();
    #ifdef DNABUG 
      std::cout << "DIFF: " << diff  << std::endl; 
    #endif
    for (unsigned i = level; i < offspring_a.loop_nest.storage_tiling_boundaries.size(); i++)
      offspring_a.loop_nest.storage_tiling_boundaries[i] -=  diff;

    for (unsigned i = level; i < offspring_b.loop_nest.storage_tiling_boundaries.size(); i++)
      offspring_b.loop_nest.storage_tiling_boundaries[i] +=  diff;


    // Swap datatype bypass
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      bool bit_a = offspring_a.datatype_bypass_nest.at(pvi).test(level);
      bool bit_b = offspring_b.datatype_bypass_nest.at(pvi).test(level);
      
      if (bit_a) offspring_b.datatype_bypass_nest.at(pvi).set(level);
      else offspring_b.datatype_bypass_nest.at(pvi).reset(level);

      if (bit_b) offspring_a.datatype_bypass_nest.at(pvi).set(level);
      else offspring_a.datatype_bypass_nest.at(pvi).reset(level);
    }
  }

  void FanoutMutation(Mapping& mapping) {
    // Set spatial loops bounds to maximum possible
    for (uint32_t level = 0; level < mapping.loop_nest.storage_tiling_boundaries.size(); level++) {
      if (arch_props_.Fanout(level) <= 1) continue;

      bool is_constrained;
      std::map<problem::Shape::DimensionID, int> factors;
      try {
        auto tiling_level_id = arch_props_.SpatialToTiling(level);
        factors = constraints_->Factors().at(tiling_level_id);
        is_constrained = true;
      } catch (const std::out_of_range& oor) {
        is_constrained = false;
      }

      std::vector<loop::Descriptor> level_nest = GetNestAtLevel(mapping, level);
      
      bool x_loop_found = false;
      bool y_loop_found = false;
      uint32_t x_product = 1;
      uint32_t y_product = 1;
      for (auto& s : level_nest) {
        if (s.spacetime_dimension == spacetime::Dimension::SpaceX) {
          x_product *= s.end;
          x_loop_found = true;
        } else if (s.spacetime_dimension == spacetime::Dimension::SpaceY) {
          y_product *= s.end;
          y_loop_found = true;
        }
      }
      
      if (x_loop_found || y_loop_found) {
        for (auto& s : level_nest) {
          if (s.spacetime_dimension == spacetime::Dimension::Time) continue;
          
          uint64_t new_factor = 0;
          if (is_constrained) {
            auto factor = factors.find(s.dimension);
            if (factor != factors.end()) {
              new_factor = factor->second;

              if (s.spacetime_dimension == spacetime::Dimension::SpaceX && x_product * new_factor / s.end > arch_props_.FanoutX(level))
                continue;
              if (s.spacetime_dimension == spacetime::Dimension::SpaceY && y_product * new_factor / s.end > arch_props_.FanoutY(level))
                continue;
            }
          }
          
          if (new_factor == 0) {
            std::cout << "FM_P: " << mapping.PrintCompact() << std::endl;
            new_factor = s.spacetime_dimension == spacetime::Dimension::SpaceX ? 
                    arch_props_.FanoutX(level) / (x_product / s.end) :
                    arch_props_.FanoutY(level) / (y_product / s.end) ;
           
            if ((uint64_t)workload_.GetBound(s.dimension) < new_factor)
              new_factor = workload_.GetBound(s.dimension);
            else if (workload_.GetBound(s.dimension) % new_factor)
              continue; 
              // TODO - Find greatest divisor of workload_.GetBound(s->dimension) less than Fanout (new_factor)
          }

          uint64_t old_factor = s.end;
          s.end = new_factor ;

          FactorCompensation(s.dimension, s.stride, old_factor, s.end, level, mapping.loop_nest); 
        }


        unsigned start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level-1) + 1 : 0;
        unsigned end = mapping.loop_nest.storage_tiling_boundaries.at(level) + 1;
        mapping.loop_nest.loops.erase(mapping.loop_nest.loops.begin() + start, mapping.loop_nest.loops.begin() + end);
        mapping.loop_nest.loops.insert(mapping.loop_nest.loops.begin() + start, level_nest.begin(), level_nest.end());
      }

      if (!y_loop_found && level_nest.size() > 1) {
        /*
        std::cout << "FANOUT LEVEL " << level << " Size: " << level_nest.size() << std::endl;
        for (auto l : level_nest)
          std::cout << l;
        for (uint32_t level = 0; level < mapping.loop_nest.storage_tiling_boundaries.size(); level++) 
          std::cout << " " << arch_props_.Fanout(level);
        std::cout << std::endl << "FM_P: " << mapping.PrintCompact() << std::endl;
        */

        unsigned start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level-1) + 1 : 0;
        level_nest = GetNestAtLevel(mapping, level);

        int loop_tc = -1;
        for (unsigned j = 0; j < level_nest.size(); j++) { 
          if ((unsigned)level_nest[j].end <= arch_props_.FanoutY(level) && 
              level_nest[j].end > 1 &&
              (loop_tc == -1 || level_nest[j].end > level_nest[loop_tc].end) ) 
            loop_tc = j;
        }

        if (loop_tc > -1) {
          mapping.loop_nest.loops.at(start + loop_tc).spacetime_dimension = spacetime::Dimension::SpaceY;
          if (loop_tc != 0) 
            std::swap(mapping.loop_nest.loops.at(start + loop_tc), mapping.loop_nest.loops.at(start));
  
        }

      
      }
      //std::cout << "FM_D: " << mapping.PrintCompact() << std::endl;
    }
  }

  // Fill buffer at lower levels - Funziona solo con quelli che contengono un solo datatype per ora forse
  void FillMutation(model::Engine& engine, Mapping& mapping) {
    unsigned level = unsigned((arch_props_.StorageLevels()-1) * exp_distribution_(rng_));
    level = (level == arch_props_.StorageLevels()-1) ? arch_props_.StorageLevels() - 2 : level;

    // Vedere bypass
    unsigned n_datatypes_in_buffer = 0; 
    unsigned datatype = 0;

    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      if (mapping.datatype_bypass_nest[pv][level]) {
        n_datatypes_in_buffer++;
        datatype = pv;
      }
    
    if (n_datatypes_in_buffer != 1) return; // Non supportato
    
    auto utilized_capacity = engine.GetTopology().GetStats().tile_sizes.at(level).at(datatype);
    auto buffer_capacity = arch_specs_.topology.GetStorageLevel(level)->size.Get();
    
    // FIXME - questa cosa non ha senso con l'input dataspace o comunque quando gli indici sono composti
    // Va considerato lo stride la dilation e tante altre cose...
    uint64_t factor_needed = buffer_capacity / utilized_capacity;
    if (factor_needed == 1) return;
    
    uint64_t start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level-1) + 1 : 0;
    uint64_t end = mapping.loop_nest.storage_tiling_boundaries.at(level) + 1;
    
    for (auto l = mapping.loop_nest.loops.begin() + start; l != mapping.loop_nest.loops.begin() + end; l++) {
      if (l->spacetime_dimension != spacetime::Dimension::Time) continue;

      // Verificare se dimensione appartiene a daataspace
      bool is_proj = false;
      for (auto& proj : problem::GetShape()->Projections[datatype])
        for (auto& projex : proj)
          if (projex.first == problem::GetShape()->NumCoefficients && projex.second == l->dimension)
            is_proj = true;
      
      if (!is_proj) continue;
      
      // Trovo fattore divisore della dimensione del workload, in maniera un po' brutta
      uint64_t factor_div = factor_needed;

      uint64_t max_factor = workload_.GetBound(l->dimension) / l->end;
      if (max_factor < factor_needed) continue;

      while (max_factor % factor_div) factor_div--;

      if (factor_div == 1) continue;

      uint64_t factor_avaiable = 1;
      for (auto ln = l+1; ln != mapping.loop_nest.loops.end(); ln++) 
        if (l->dimension == ln->dimension)
          factor_avaiable *= l->end;
      
      if (factor_div > factor_avaiable) continue;

      uint64_t old_factor = l->end;
      l->end *= factor_div;

      FactorCompensation(l->dimension, l->stride, old_factor, l->end, level, mapping.loop_nest);
      std::cout << "Fill Mutation" << std::endl;
    }

    // Aumentare fattore opportunamente
    
    // Funziona solo con 3d conv così :(
    // W = UP + R - U 
    // H = UQ + S - U
 
  }

  void RandomMutation(Mapping& mapping) {
    // Random loop factor swapping
    if (uni_distribution_(rng_) < 0.5) {

      unsigned num_levels = mapping.loop_nest.storage_tiling_boundaries.size() - 1;
      unsigned level_a = unsigned( num_levels * uni_distribution_(rng_) );
      unsigned level_b = unsigned( num_levels * uni_distribution_(rng_) );
      if (level_a == level_b) return;

      auto level_a_nest = GetNestAtLevel(mapping, level_a);
      unsigned loop_a = unsigned( level_a_nest.size() * uni_distribution_(rng_) );

      auto level_b_nest = GetNestAtLevel(mapping, level_b);
      unsigned loop_b = unsigned( level_b_nest.size() * uni_distribution_(rng_) );

      if (level_a_nest[loop_a].spacetime_dimension != spacetime::Dimension::Time ||
          level_b_nest[loop_b].spacetime_dimension != spacetime::Dimension::Time ) 
        return;

      auto dim_a = level_a_nest.at(loop_a).dimension;
      int id_same_dim_in_b = -1;
      for (unsigned l = 0; l < level_b_nest.size(); l++) 
        if (level_b_nest[l].dimension == dim_a && level_b_nest[l].spacetime_dimension == spacetime::Dimension::Time)
          id_same_dim_in_b = l;

      auto dim_b = level_b_nest.at(loop_b).dimension;
      int id_same_dim_in_a = -1;
      for (unsigned l = 0; l < level_a_nest.size(); l++) 
        if (level_a_nest[l].dimension == dim_b && level_a_nest[l].spacetime_dimension == spacetime::Dimension::Time)
          id_same_dim_in_a = l;

      unsigned start_a = level_a > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level_a-1) + 1 : 0;
      mapping.loop_nest.loops.at(start_a+loop_a).end = 1;

      if (id_same_dim_in_a >= 0) {
        mapping.loop_nest.loops.at(start_a+id_same_dim_in_a).end *= level_b_nest[loop_b].end;
      } else {
        mapping.loop_nest.loops.insert(mapping.loop_nest.loops.begin() + start_a + level_a_nest.size() - 1, level_b_nest[loop_b]);

        for (unsigned i = level_a; i < mapping.loop_nest.storage_tiling_boundaries.size(); i++)
          mapping.loop_nest.storage_tiling_boundaries[i] +=  1;

      }

      unsigned start_b = level_b > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level_b-1) + 1 : 0;
      mapping.loop_nest.loops.at(start_b+loop_b).end = 1;
      
      if (id_same_dim_in_b >= 0) {
        mapping.loop_nest.loops.at(start_b+id_same_dim_in_b).end *= level_a_nest[loop_a].end;
      } else {
        mapping.loop_nest.loops.insert(mapping.loop_nest.loops.begin() + start_b + level_b_nest.size() - 1, level_a_nest[loop_a]);

        for (unsigned i = level_b; i < mapping.loop_nest.storage_tiling_boundaries.size(); i++)
          mapping.loop_nest.storage_tiling_boundaries[i] +=  1;
      }

    // Random loop permutation
    } else {
      
      unsigned num_levels = mapping.loop_nest.storage_tiling_boundaries.size();
      unsigned level = unsigned( num_levels * uni_distribution_(rng_) );
      assert(level < num_levels);
      
      unsigned start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level-1) + 1 : 0;
      auto level_nest = GetNestAtLevel(mapping, level);
      unsigned loop_a = start + unsigned( level_nest.size() * uni_distribution_(rng_) );
      unsigned loop_b = start + unsigned( level_nest.size() * uni_distribution_(rng_) );

      if (loop_a != loop_b &&
          mapping.loop_nest.loops.at(loop_a).spacetime_dimension == spacetime::Dimension::Time &&
          mapping.loop_nest.loops.at(loop_b).spacetime_dimension == spacetime::Dimension::Time)
        std::swap(mapping.loop_nest.loops.at(loop_a), mapping.loop_nest.loops.at(loop_b));
    }
  }

  void Mutation(Individual& individual) {
    if (uni_distribution_(rng_) < fill_mutation_prob_ && individual.engine.IsEvaluated())
      FillMutation(individual.engine, individual.genome);

    if (uni_distribution_(rng_) < parallel_mutation_prob_)
      FanoutMutation(individual.genome);

    if (uni_distribution_(rng_) < random_mutation_prob_)
      RandomMutation(individual.genome);
  }

  void UpdateBestMapping() {
    std::unique_lock<std::mutex> lock(*global_mutex_);
    // Best update
    if (!global_best_individual_->engine.IsSpecced() || best_individual_.fitness > global_best_individual_->fitness)
    {
      *global_best_individual_ = best_individual_;

      auto stats = best_individual_.engine.GetTopology().GetStats();
      std::cout << "[INFO] Utilization = " << std::setw(4) << std::fixed << std::setprecision(2) << stats.utilization 
          << " | pJ/MACC = " << std::setw(8) << std::fixed << std::setprecision(3) << stats.energy / stats.maccs
          << " | Cycles = " << std::setw(8) << std::fixed << std::setprecision(1) << stats.cycles
          << " | Energy = " << std::setw(8) << std::fixed << std::setprecision(1) << stats.energy
          << " | Fitness = " << std::setw(8) << std::fixed << std::setprecision(1) << best_individual_.fitness
          << std::endl << "[INFO] Mapping = " << best_individual_.genome.PrintCompact()
          << std::endl;      
    }
  }

  void RandomIndividual(uint32_t p, Population& population) {
    while(true) {
      Mapping mapping;
      if (!RandomMapping(&mapping)) 
        continue;
      if (Evaluate(mapping, population[p]))
        return;
    }
  }

  void RandomPopulation(uint32_t p, uint32_t pop_slice_end, Population& population) {
    if (p == 0 && p < pop_slice_end && user_mapping_defined_) {
      // This should fix CoSa mapping fanout problems.
      for (uint32_t level = 0; level < user_mapping_.loop_nest.storage_tiling_boundaries.size(); level++) {
        uint64_t start = level > 0 ? user_mapping_.loop_nest.storage_tiling_boundaries.at(level-1) + 1 : 0;
        uint64_t end = user_mapping_.loop_nest.storage_tiling_boundaries.at(level) + 1;

        uint32_t x_product = 1;
        uint32_t y_product = 1;
        for (auto s = user_mapping_.loop_nest.loops.begin() + start; s != user_mapping_.loop_nest.loops.begin() + end; s++) {
          if (s->spacetime_dimension == spacetime::Dimension::SpaceX) {
            x_product *= s->end;
          } else if (s->spacetime_dimension == spacetime::Dimension::SpaceY) {
            y_product *= s->end;
          }
        }
        if (x_product > arch_props_.FanoutX(level) || y_product > arch_props_.FanoutY(level)) {
          for (auto s = user_mapping_.loop_nest.loops.begin() + start; s != user_mapping_.loop_nest.loops.begin() + end; s++) {
            if (s->spacetime_dimension == spacetime::Dimension::SpaceX)
              s->spacetime_dimension = spacetime::Dimension::SpaceY;
            else if (s->spacetime_dimension == spacetime::Dimension::SpaceY)
              s->spacetime_dimension = spacetime::Dimension::SpaceX;
          }
        }
      }
      assert(Evaluate(user_mapping_, population[p]));
      p++;
    }

    while (p < pop_slice_end)
    {
      // Mapping generation
      Mapping mapping;
      if (!RandomMapping(&mapping)) 
        continue;
      if (Evaluate(mapping, population[p]))
        p++;
    }
  }

  uint64_t Tournament() {
    uint64_t b1 = tour_distribution_(rng_);
    uint64_t b2 = tour_distribution_(rng_);

    if (parent_population_[b1].rank < parent_population_[b2].rank) {
      return b1;
    } else if (parent_population_[b1].rank == parent_population_[b2].rank) {
      if (parent_population_[b1].crowding_distance > parent_population_[b2].crowding_distance)
        return b1;
      else 
        return b2;
    } else {
      return b2;
    }
  }

 public:
  MedeaThread(
    unsigned thread_id,
    config::CompoundConfig* config,
    std::string out_dir,
    problem::Workload &workload,
    model::Engine::Specs arch_specs,
    config::CompoundConfigNode arch_config,
    mapspace::MapSpace* mapspace,
    mapping::Constraints* constraints,
    Individual* best_individual,
    std::vector<Individual>& immigrant_population,
    std::vector<Individual>& parent_population,
    std::vector<Individual>& population,
    Orchestrator* thread_orchestrator,
    std::mutex* global_mutex,
    uint32_t num_threads,
    uint32_t population_size,
    uint32_t immigrant_population_size,
    uint32_t generations,
    double fill_mutation_prob,
    double parallel_mutation_prob,
    double random_mutation_prob,
    bool use_tournament,
    Mapping user_mapping,
    bool user_mapping_defined,
    RandomGenerator128* if_rng,
    RandomGenerator128* lp_rng,
    RandomGenerator128* db_rng,
    RandomGenerator128* sp_rng,
    RandomGenerator128* crossover_rng
    ) :
      thread_id_(thread_id),
      config_(config),
      out_dir_(out_dir),
      workload_(workload),
      arch_specs_(arch_specs),
      arch_config_(arch_config),
      arch_props_(arch_specs),
      mapspace_(mapspace),
      constraints_(constraints),
      global_best_individual_(best_individual),
      immigrant_population_(immigrant_population),
      parent_population_(parent_population),
      population_(population),
      thread_orchestrator_(thread_orchestrator),
      global_mutex_(global_mutex),
      num_threads_(num_threads),
      population_size_(population_size),
      immigrant_population_size_(immigrant_population_size),
      num_generations_(generations),
      fill_mutation_prob_(fill_mutation_prob),
      parallel_mutation_prob_(parallel_mutation_prob),
      random_mutation_prob_(random_mutation_prob),
      use_tournament_(use_tournament),
      user_mapping_(user_mapping),
      user_mapping_defined_(user_mapping_defined),
      thread_(),
      if_rng_(if_rng),
      lp_rng_(lp_rng),
      db_rng_(db_rng),
      sp_rng_(sp_rng),
      crossover_rng_(crossover_rng),
      rng_(thread_id),
      exp_distribution_(3.5),
      uni_distribution_(0, 1),
      tour_distribution_(0, population_size_-1)
  {
     best_individual_.fitness = - std::numeric_limits<double>::max();
  }

  ~MedeaThread()
  {
  }

  void Start()
  {
    thread_ = std::thread(&MedeaThread::Run, this);
  }

  void Join()
  {
    thread_.join();
  }


  void Run()
  {

    uint32_t slice_size = population_size_ / num_threads_;
    slice_size -= slice_size % 2;
    uint32_t pop_slice_start = thread_id_ * slice_size;
    uint32_t pop_slice_end = (thread_id_ == (num_threads_ - 1)) ? population_size_ : (pop_slice_start + slice_size);

    uint32_t imm_slice_size = immigrant_population_size_ / num_threads_;
    imm_slice_size -= imm_slice_size % 2;
    uint32_t imm_pop_slice_start = thread_id_ * imm_slice_size;
    uint32_t imm_pop_slice_end = (thread_id_ == (num_threads_ - 1)) ? immigrant_population_size_ : (imm_pop_slice_start + imm_slice_size);
    
    // Initial population.
    thread_orchestrator_->FollowerWait(next_iteration_);
    RandomPopulation(pop_slice_start, pop_slice_end, parent_population_);

    thread_orchestrator_->FollowerDone();

    UpdateBestMapping();

    // Wait for others
    thread_orchestrator_->FollowerWait(next_iteration_);
    for (uint32_t g = 0; g < num_generations_; g++) {
      uint64_t debug_cross_count = 0;
      for (uint32_t ep = pop_slice_start; ep < pop_slice_end; ep += 2) {
        
        if (use_tournament_)
          Crossover(parent_population_[Tournament()].genome, parent_population_[Tournament()].genome,
                    population_[ep].genome, population_[ep+1].genome);
        else
          Crossover(parent_population_[ep].genome, parent_population_[ep+1].genome,
                    population_[ep].genome, population_[ep+1].genome);
                  

        Mutation(population_[ep]);
        Mutation(population_[ep+1]);

        //std::cout << population_[ep].genome.PrintCompact() << std::endl;

        if (Evaluate(population_[ep].genome, population_[ep]))
          debug_cross_count++;
        else 
          RandomIndividual(ep, population_);

        //std::cout << population_[ep+1].genome.PrintCompact() << std::endl;
        
        if (Evaluate(population_[ep+1].genome, population_[ep+1])) 
          debug_cross_count++;
        else 
          RandomIndividual(ep+1, population_);
      }

      std::cout << "[T"<< thread_id_ << "] Successfully evaluated " << debug_cross_count << "/" << pop_slice_end - pop_slice_start << " crossed mappings." << std::endl;

      UpdateBestMapping();

      // Immigration
      RandomPopulation(imm_pop_slice_start, imm_pop_slice_end, immigrant_population_);

      std::cout << "[T"<< thread_id_ << "] Immigration completed" << std::endl;
      thread_orchestrator_->FollowerDone();
      UpdateBestMapping(); 

      // Wait for others and merging in main
      thread_orchestrator_->FollowerWait(next_iteration_);

    }



    // if (gTerminate)
  }
};


}