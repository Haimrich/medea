#include "mapper-thread.hpp"

#include "mapspaces/mapspace-base.hpp"
#include "mapping/constraints.hpp"

#include "common.hpp"
#include "accelergy.hpp"

namespace medea
{

  uint64_t MedeaMapperThread::gcd(uint64_t a, uint64_t b)
  {
    if (b == 0)
      return a;
    return gcd(b, a % b);
  }

  double MedeaMapperThread::Fitness(model::Engine engine)
  {
    auto stats = engine.GetTopology().GetStats();
    return -stats.energy * stats.cycles;
  }

  bool MedeaMapperThread::EngineSuccess(std::vector<model::EvalStatus> &status_per_level)
  {
    return std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                           [](bool cur, const model::EvalStatus &status)
                           { return cur && status.success; });
  }

  bool MedeaMapperThread::RandomMapping(Mapping *mapping)
  {
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
                           [](bool cur, const mapspace::Status &status)
                           { return cur && status.success; });
  }

  LoopRange MedeaMapperThread::GetSubnestRangeAtLevel(const Mapping &mapping, unsigned level)
  {
    size_t start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level - 1) + 1 : 0;
    size_t end = mapping.loop_nest.storage_tiling_boundaries.at(level) + 1;
    return LoopRange(mapping.loop_nest.loops.begin() + start, mapping.loop_nest.loops.begin() + end);
  }

  uint64_t MedeaMapperThread::GetParallelAtLevel(const Mapping &mapping, spacetime::Dimension dim, uint64_t level)
  {
    uint64_t result = 1;
    LoopRange subnest = GetSubnestRangeAtLevel(mapping, level);
    for (auto &l : subnest)
      if (l.spacetime_dimension == dim)
        result *= l.end;
    return result;
  }

  std::vector<loop::Descriptor> MedeaMapperThread::GetSubnestAtLevel(const Mapping &mapping, unsigned level)
  {
    LoopRange subnest_range = GetSubnestRangeAtLevel(mapping, level);
    return std::vector<loop::Descriptor>(subnest_range.begin(), subnest_range.end());
  }

  uint64_t MedeaMapperThread::GetDimFactorInSubnest(problem::Shape::DimensionID dimension, std::vector<loop::Descriptor> &subnest)
  {
    uint64_t factor = 1;
    for (auto &l : subnest)
      if (l.dimension == dimension)
        factor *= l.end;
    return factor;
  }

  uint64_t MedeaMapperThread::GetStrideInSubnest(problem::Shape::DimensionID dimension, std::vector<loop::Descriptor> &subnest)
  {
    for (auto &l : subnest)
      if (l.dimension == dimension)
        return l.stride;

    return 1;
  }

  void MedeaMapperThread::UpdateArchitecture(Mapping &mapping, model::Engine &engine)
  {

    std::map<std::string, uint64_t> updates;

    unsigned buffer_update_granularity = 16; // This should be configurable FIXME

    auto new_specs = model::Topology::Specs(arch_specs_.topology);
    for (unsigned i = 0; i < arch_specs_.topology.NumStorageLevels(); i++)
    {
      auto buffer = new_specs.GetStorageLevel(i);

      auto tile_sizes = engine.GetTopology().GetStats().tile_sizes.at(i);
      auto utilized_capacity = std::accumulate(tile_sizes.begin(), tile_sizes.end(), 0);

      if (!buffer->block_size.IsSpecified())
        continue;
      auto block_size = buffer->block_size.Get();

      if (!buffer->size.IsSpecified())
        continue;

      unsigned needed_depth = (utilized_capacity / block_size) + 1;
      unsigned remainder = needed_depth % buffer_update_granularity;
      unsigned new_depth = remainder ? needed_depth + buffer_update_granularity - remainder : needed_depth;

      buffer->size = new_depth * block_size;
      buffer->effective_size = static_cast<uint64_t>(std::floor(buffer->size.Get() / buffer->multiple_buffering.Get()));
      updates[buffer->name.Get()] = new_depth;
    }

    int i;

    for (i = arch_specs_.topology.NumLevels() - 2; i > 0; i--)
    {
      auto buffer = new_specs.GetStorageLevel(i - 1);

      buffer->meshX = GetParallelAtLevel(mapping, spacetime::Dimension::SpaceX, i) * new_specs.GetStorageLevel(i)->meshX.Get();
      buffer->meshY = GetParallelAtLevel(mapping, spacetime::Dimension::SpaceY, i) * new_specs.GetStorageLevel(i)->meshY.Get();
      buffer->instances = buffer->meshX.Get() * buffer->meshY.Get();
    }

    if (i == 0)
    {
      auto arithmetic = new_specs.GetArithmeticLevel();

      arithmetic->meshX = GetParallelAtLevel(mapping, spacetime::Dimension::SpaceX, i) * new_specs.GetStorageLevel(i)->meshX.Get();
      arithmetic->meshY = GetParallelAtLevel(mapping, spacetime::Dimension::SpaceY, i) * new_specs.GetStorageLevel(i)->meshY.Get();
      arithmetic->instances = arithmetic->meshX.Get() * arithmetic->meshY.Get();
    }

    std::string out_prefix = "medea." + std::to_string(thread_id_) + "_tmp";
    auto accelergy = Accelergy(fast_accelergy_path_, config_, updates, out_prefix);

    model::Engine::Specs new_engine_specs;
    new_engine_specs.topology = new_specs;
    new_engine_specs.topology.ParseAccelergyART(accelergy.GetART());
    new_engine_specs.topology.ParseAccelergyERT(accelergy.GetERT());
    engine.Spec(new_engine_specs);
  }

  bool MedeaMapperThread::Evaluate(Mapping mapping, Individual &individual)
  {
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
    assert(EngineSuccess(status_per_level));

    // Population update
    individual.genome = mapping;
    individual.objectives[0] = engine.Energy();
    individual.objectives[1] = (double)engine.Cycles();
    individual.objectives[2] = engine.Area();
    individual.engine = engine;

    return true;
  }

  void MedeaMapperThread::FactorCompensation(const problem::Shape::DimensionID &dim, const uint64_t stride, const uint64_t old_factor, const uint64_t new_factor, const uint64_t level, loop::Nest &nest)
  {

    if (new_factor < old_factor)
    {
      // Prima passare da old_factor a 1 poi da 1 a new_factor -> ricorsivo
      if (old_factor % new_factor)
      {
        FactorCompensation(dim, stride, old_factor, 1, level, nest);
        FactorCompensation(dim, stride, 1, new_factor, level, nest);
        return;
      }

      // Fattore diminuito -> compensiamo aumentando in RAM.
      int64_t factor = old_factor / new_factor;

      int64_t ram_level = nest.storage_tiling_boundaries.size() - 2;
      uint64_t ram_start = ram_level > 0 ? nest.storage_tiling_boundaries.at(ram_level) + 1 : 0;

      auto ram_loop = std::find_if(nest.loops.begin() + ram_start, nest.loops.end(), [&](const loop::Descriptor &x)
                                   { return x.dimension == dim; });

      if (ram_loop != nest.loops.end())
      {
        ram_loop->end *= factor;
      }
      else
      {
        loop::Descriptor new_loop(dim, 0, factor, stride, spacetime::Dimension::Time);
        nest.loops.push_back(new_loop);
        nest.storage_tiling_boundaries.back()++;
      }
    }
    else if (new_factor > old_factor)
    {
      // Fattore aumentato -> Compensiamo diminuendo in RAM o nel primo che troviamo a scendere
      if (new_factor % old_factor)
      {
        FactorCompensation(dim, stride, old_factor, 1, level, nest);
        FactorCompensation(dim, stride, 1, new_factor, level, nest);
        return;
      }

      int64_t factor = new_factor / old_factor;

      for (int64_t l = nest.storage_tiling_boundaries.size() - 1; l >= 0 && factor != 1; l--)
      {
        // Cerca fattore da ridurre (escluso livello in cui l'abbiamo incrementato)

        if (l != (int64_t)level)
        {
          uint64_t l_start = l > 0 ? nest.storage_tiling_boundaries.at(l - 1) + 1 : 0;
          uint64_t l_end = nest.storage_tiling_boundaries.at(l) + 1;

          for (auto l_loop = nest.loops.begin() + l_start; l_loop != nest.loops.begin() + l_end && factor != 1; l_loop++)
          {
            if (l_loop->dimension == dim)
            {
              uint64_t common = gcd(factor, l_loop->end);

              factor /= common;
              l_loop->end /= common;
            }
          }
        }
      }
    }
  }

  void MedeaMapperThread::Crossover(const Mapping &parent_a, const Mapping &parent_b, Mapping &offspring_a, Mapping &offspring_b)
  {
    global_mutex_->lock();
    offspring_a = parent_a;
    offspring_b = parent_b;

    uint64_t level = crossover_rng_->Next().convert_to<uint64_t>() % (parent_a.loop_nest.storage_tiling_boundaries.size() - 1);
    global_mutex_->unlock();

    loop::Nest nest_a = parent_a.loop_nest;
    uint64_t a_start = level > 0 ? nest_a.storage_tiling_boundaries.at(level - 1) + 1 : 0;
    uint64_t a_end = nest_a.storage_tiling_boundaries.at(level) + 1;
    std::vector<loop::Descriptor> a_level(nest_a.loops.begin() + a_start, nest_a.loops.begin() + a_end);

    loop::Nest nest_b = parent_b.loop_nest;
    uint64_t b_start = level > 0 ? nest_b.storage_tiling_boundaries.at(level - 1) + 1 : 0;
    uint64_t b_end = nest_b.storage_tiling_boundaries.at(level) + 1;
    std::vector<loop::Descriptor> b_level(nest_b.loops.begin() + b_start, nest_b.loops.begin() + b_end);

    // Factor compensation
    for (int idim = 0; idim < int(problem::GetShape()->NumDimensions); idim++)
    {
      problem::Shape::DimensionID dimension = problem::Shape::DimensionID(idim);

      uint64_t factor_a = GetDimFactorInSubnest(dimension, a_level);
      uint64_t factor_b = GetDimFactorInSubnest(dimension, b_level);
      uint64_t stride_a = GetStrideInSubnest(dimension, a_level);
      uint64_t stride_b = GetStrideInSubnest(dimension, b_level);

      FactorCompensation(dimension, stride_a, factor_a, factor_b, level, offspring_a.loop_nest);
      FactorCompensation(dimension, stride_b, factor_b, factor_a, level, offspring_b.loop_nest);
    }

    LoopRange range_a = GetSubnestRangeAtLevel(offspring_a, level);
    LoopRange range_b = GetSubnestRangeAtLevel(offspring_b, level);

    offspring_a.loop_nest.loops.erase(range_a.begin(), range_a.end());
    offspring_a.loop_nest.loops.insert(range_a.begin(), b_level.begin(), b_level.end());

    offspring_b.loop_nest.loops.erase(range_b.begin(), range_b.end());
    offspring_b.loop_nest.loops.insert(range_b.begin(), a_level.begin(), a_level.end());

    int64_t diff = a_level.size() - b_level.size();
#ifdef DNABUG
    std::cout << "DIFF: " << diff << std::endl;
#endif
    for (unsigned i = level; i < offspring_a.loop_nest.storage_tiling_boundaries.size(); i++)
      offspring_a.loop_nest.storage_tiling_boundaries[i] -= diff;

    for (unsigned i = level; i < offspring_b.loop_nest.storage_tiling_boundaries.size(); i++)
      offspring_b.loop_nest.storage_tiling_boundaries[i] += diff;

    // Swap datatype bypass
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      bool bit_a = offspring_a.datatype_bypass_nest.at(pvi).test(level);
      bool bit_b = offspring_b.datatype_bypass_nest.at(pvi).test(level);

      if (bit_a)
        offspring_b.datatype_bypass_nest.at(pvi).set(level);
      else
        offspring_b.datatype_bypass_nest.at(pvi).reset(level);

      if (bit_b)
        offspring_a.datatype_bypass_nest.at(pvi).set(level);
      else
        offspring_a.datatype_bypass_nest.at(pvi).reset(level);
    }
  }

  void MedeaMapperThread::FanoutMutation(Mapping &mapping)
  {
    // Set spatial loops bounds to maximum possible
    for (uint32_t level = 0; level < mapping.loop_nest.storage_tiling_boundaries.size(); level++)
    {
      if (arch_props_.Fanout(level) <= 1)
        continue;

      bool is_constrained;
      std::map<problem::Shape::DimensionID, int> factors;
      try
      {
        auto tiling_level_id = arch_props_.SpatialToTiling(level);
        factors = constraints_->Factors().at(tiling_level_id);
        is_constrained = true;
      }
      catch (const std::out_of_range &oor)
      {
        is_constrained = false;
      }

      std::vector<loop::Descriptor> level_nest = GetSubnestAtLevel(mapping, level);

      bool x_loop_found = false;
      bool y_loop_found = false;
      uint32_t x_product = 1;
      uint32_t y_product = 1;
      for (auto &s : level_nest)
      {
        if (s.spacetime_dimension == spacetime::Dimension::SpaceX)
        {
          x_product *= s.end;
          x_loop_found = true;
        }
        else if (s.spacetime_dimension == spacetime::Dimension::SpaceY)
        {
          y_product *= s.end;
          y_loop_found = true;
        }
      }

      if (x_loop_found || y_loop_found)
      {
        for (auto &s : level_nest)
        {
          if (s.spacetime_dimension == spacetime::Dimension::Time)
            continue;

          uint64_t new_factor = 0;
          if (is_constrained)
          {
            auto factor = factors.find(s.dimension);
            if (factor != factors.end())
            {
              new_factor = factor->second;

              if (s.spacetime_dimension == spacetime::Dimension::SpaceX && x_product * new_factor / s.end > arch_props_.FanoutX(level))
                continue;
              if (s.spacetime_dimension == spacetime::Dimension::SpaceY && y_product * new_factor / s.end > arch_props_.FanoutY(level))
                continue;
            }
          }

          if (new_factor == 0)
          {
            // std::cout << "FM_P: " << mapping.PrintCompact() << std::endl;
            new_factor = s.spacetime_dimension == spacetime::Dimension::SpaceX ? arch_props_.FanoutX(level) / (x_product / s.end) : arch_props_.FanoutY(level) / (y_product / s.end);

            if (new_factor == 0)
              return;

            if ((uint64_t)workload_.GetBound(s.dimension) < new_factor)
              new_factor = workload_.GetBound(s.dimension);
            else if (workload_.GetBound(s.dimension) % new_factor)
              continue;
            // TODO - Find greatest divisor of workload_.GetBound(s->dimension) less than Fanout (new_factor)
          }

          uint64_t old_factor = s.end;
          s.end = new_factor;

          FactorCompensation(s.dimension, s.stride, old_factor, s.end, level, mapping.loop_nest);
        }

        LoopRange range = GetSubnestRangeAtLevel(mapping, level);
        mapping.loop_nest.loops.erase(range.begin(), range.end());
        mapping.loop_nest.loops.insert(range.begin(), level_nest.begin(), level_nest.end());
      }

      if (!y_loop_found && level_nest.size() > 1)
      {
        /*
    std::cout << "FANOUT LEVEL " << level << " Size: " << level_nest.size() << std::endl;
    for (auto l : level_nest)
        std::cout << l;
    for (uint32_t level = 0; level < mapping.loop_nest.storage_tiling_boundaries.size(); level++) 
        std::cout << " " << arch_props_.Fanout(level);
    std::cout << std::endl << "FM_P: " << mapping.PrintCompact() << std::endl;
    */

        unsigned start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level - 1) + 1 : 0;
        level_nest = GetSubnestAtLevel(mapping, level);

        int loop_tc = -1;
        for (unsigned j = 0; j < level_nest.size(); j++)
        {
          if ((unsigned)level_nest[j].end <= arch_props_.FanoutY(level) &&
              level_nest[j].end > 1 &&
              (loop_tc == -1 || level_nest[j].end > level_nest[loop_tc].end))
            loop_tc = j;
        }

        if (loop_tc > -1)
        {
          mapping.loop_nest.loops.at(start + loop_tc).spacetime_dimension = spacetime::Dimension::SpaceY;
          if (loop_tc != 0)
            std::swap(mapping.loop_nest.loops.at(start + loop_tc), mapping.loop_nest.loops.at(start));
        }
      }
      //std::cout << "FM_D: " << mapping.PrintCompact() << std::endl;
    }
  }

  // Fill buffer at lower levels - Funziona solo con quelli che contengono un solo datatype per ora forse
  void MedeaMapperThread::FillMutation(model::Engine &engine, Mapping &mapping)
  {
    unsigned level = unsigned((arch_props_.StorageLevels() - 1) * exp_distribution_(rng_));
    level = (level == arch_props_.StorageLevels() - 1) ? arch_props_.StorageLevels() - 2 : level;

    // Vedere bypass
    unsigned n_datatypes_in_buffer = 0;
    unsigned datatype = 0;

    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      if (mapping.datatype_bypass_nest[pv][level])
      {
        n_datatypes_in_buffer++;
        datatype = pv;
      }

    if (n_datatypes_in_buffer != 1)
      return; // Non supportato

    auto utilized_capacity = engine.GetTopology().GetStats().tile_sizes.at(level).at(datatype);
    auto buffer_capacity = arch_specs_.topology.GetStorageLevel(level)->size.Get();

    // FIXME - questa cosa non ha senso con l'input dataspace o comunque quando gli indici sono composti
    // Va considerato lo stride la dilation e tante altre cose...
    uint64_t factor_needed = buffer_capacity / utilized_capacity;
    if (factor_needed == 1)
      return;

    uint64_t start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level - 1) + 1 : 0;
    uint64_t end = mapping.loop_nest.storage_tiling_boundaries.at(level) + 1;

    for (auto l = mapping.loop_nest.loops.begin() + start; l != mapping.loop_nest.loops.begin() + end; l++)
    {
      if (l->spacetime_dimension != spacetime::Dimension::Time)
        continue;

      // Verificare se dimensione appartiene a daataspace
      bool is_proj = false;
      for (auto &proj : problem::GetShape()->Projections[datatype])
        for (auto &projex : proj)
          if (projex.first == problem::GetShape()->NumCoefficients && projex.second == l->dimension)
            is_proj = true;

      if (!is_proj)
        continue;

      // Trovo fattore divisore della dimensione del workload, in maniera un po' brutta
      uint64_t factor_div = factor_needed;

      uint64_t max_factor = workload_.GetBound(l->dimension) / l->end;
      if (max_factor < factor_needed)
        continue;

      while (max_factor % factor_div)
        factor_div--;

      if (factor_div == 1)
        continue;

      uint64_t factor_avaiable = 1;
      for (auto ln = l + 1; ln != mapping.loop_nest.loops.end(); ln++)
        if (l->dimension == ln->dimension)
          factor_avaiable *= l->end;

      if (factor_div > factor_avaiable)
        continue;

      uint64_t old_factor = l->end;
      l->end *= factor_div;

      FactorCompensation(l->dimension, l->stride, old_factor, l->end, level, mapping.loop_nest);
      std::cout << "Fill Mutation" << std::endl;
    }

    // 3D conv only :(

    // TODO INPUT DATATYPE
    // W = Stride * (P - 1) + Dilation * (R - 1)
    // H = Stride * (Q - 1) + Dilation * (S - 1)
  }

  void MedeaMapperThread::RandomMutation(Mapping &mapping)
  {
    // Random loop factor swapping
    if (uni_distribution_(rng_) < 0.5)
    {

      unsigned num_levels = mapping.loop_nest.storage_tiling_boundaries.size() - 1;
      unsigned level_a = unsigned(num_levels * uni_distribution_(rng_));
      unsigned level_b = unsigned(num_levels * uni_distribution_(rng_));
      if (level_a == level_b)
        return;

      auto level_a_nest = GetSubnestAtLevel(mapping, level_a);
      unsigned loop_a = unsigned(level_a_nest.size() * uni_distribution_(rng_));

      auto level_b_nest = GetSubnestAtLevel(mapping, level_b);
      unsigned loop_b = unsigned(level_b_nest.size() * uni_distribution_(rng_));

      if (level_a_nest[loop_a].spacetime_dimension != spacetime::Dimension::Time ||
          level_b_nest[loop_b].spacetime_dimension != spacetime::Dimension::Time)
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

      unsigned start_a = level_a > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level_a - 1) + 1 : 0;
      mapping.loop_nest.loops.at(start_a + loop_a).end = 1;

      if (id_same_dim_in_a >= 0)
      {
        mapping.loop_nest.loops.at(start_a + id_same_dim_in_a).end *= level_b_nest[loop_b].end;
      }
      else
      {
        mapping.loop_nest.loops.insert(mapping.loop_nest.loops.begin() + start_a + level_a_nest.size() - 1, level_b_nest[loop_b]);

        for (unsigned i = level_a; i < mapping.loop_nest.storage_tiling_boundaries.size(); i++)
          mapping.loop_nest.storage_tiling_boundaries[i] += 1;
      }

      unsigned start_b = level_b > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level_b - 1) + 1 : 0;
      mapping.loop_nest.loops.at(start_b + loop_b).end = 1;

      if (id_same_dim_in_b >= 0)
      {
        mapping.loop_nest.loops.at(start_b + id_same_dim_in_b).end *= level_a_nest[loop_a].end;
      }
      else
      {
        mapping.loop_nest.loops.insert(mapping.loop_nest.loops.begin() + start_b + level_b_nest.size() - 1, level_a_nest[loop_a]);

        for (unsigned i = level_b; i < mapping.loop_nest.storage_tiling_boundaries.size(); i++)
          mapping.loop_nest.storage_tiling_boundaries[i] += 1;
      }

      // Random loop permutation
    }
    else
    {

      unsigned num_levels = mapping.loop_nest.storage_tiling_boundaries.size();
      unsigned level = unsigned(num_levels * uni_distribution_(rng_));
      assert(level < num_levels);

      unsigned start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level - 1) + 1 : 0;
      auto level_nest = GetSubnestAtLevel(mapping, level);
      unsigned loop_a = start + unsigned(level_nest.size() * uni_distribution_(rng_));
      unsigned loop_b = start + unsigned(level_nest.size() * uni_distribution_(rng_));

      if (loop_a != loop_b &&
          mapping.loop_nest.loops.at(loop_a).spacetime_dimension == spacetime::Dimension::Time &&
          mapping.loop_nest.loops.at(loop_b).spacetime_dimension == spacetime::Dimension::Time)
        std::swap(mapping.loop_nest.loops.at(loop_a), mapping.loop_nest.loops.at(loop_b));
    }
  }

  void MedeaMapperThread::Mutation(Individual &individual)
  {
    if (uni_distribution_(rng_) < fill_mutation_prob_ && individual.engine.IsEvaluated())
      FillMutation(individual.engine, individual.genome);

    if (uni_distribution_(rng_) < parallel_mutation_prob_)
      FanoutMutation(individual.genome);

    if (uni_distribution_(rng_) < random_mutation_prob_)
      RandomMutation(individual.genome);
  }

  void MedeaMapperThread::RandomIndividual(uint32_t p, Population &population)
  {
    while (true)
    {
      Mapping mapping;
      if (!RandomMapping(&mapping))
        continue;
      if (Evaluate(mapping, population[p]))
        return;
    }
  }

  void MedeaMapperThread::InjectUserDefinedMapping(Population &pop, uint32_t id)
  {
    // This should fix CoSA mapping fanout problems.
    for (uint32_t level = 0; level < user_mapping_.loop_nest.storage_tiling_boundaries.size(); level++)
    {
      uint64_t start = level > 0 ? user_mapping_.loop_nest.storage_tiling_boundaries.at(level - 1) + 1 : 0;
      uint64_t end = user_mapping_.loop_nest.storage_tiling_boundaries.at(level) + 1;

      uint32_t x_product = 1;
      uint32_t y_product = 1;
      for (auto s = user_mapping_.loop_nest.loops.begin() + start; s != user_mapping_.loop_nest.loops.begin() + end; s++)
      {
        if (s->spacetime_dimension == spacetime::Dimension::SpaceX)
        {
          x_product *= s->end;
        }
        else if (s->spacetime_dimension == spacetime::Dimension::SpaceY)
        {
          y_product *= s->end;
        }
      }
      if (x_product > arch_props_.FanoutX(level) || y_product > arch_props_.FanoutY(level))
      {
        for (auto s = user_mapping_.loop_nest.loops.begin() + start; s != user_mapping_.loop_nest.loops.begin() + end; s++)
        {
          if (s->spacetime_dimension == spacetime::Dimension::SpaceX)
            s->spacetime_dimension = spacetime::Dimension::SpaceY;
          else if (s->spacetime_dimension == spacetime::Dimension::SpaceY)
            s->spacetime_dimension = spacetime::Dimension::SpaceX;
        }
      }
    }

    assert(Evaluate(user_mapping_, pop[id]));
  }

  void MedeaMapperThread::RandomPopulation(uint32_t p, uint32_t pop_slice_end, Population &population)
  {
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

  uint64_t MedeaMapperThread::Tournament()
  {
    uint64_t b1 = tour_distribution_(rng_);
    uint64_t b2 = tour_distribution_(rng_);

    if (parent_population_[b1].rank < parent_population_[b2].rank)
    {
      return b1;
    }
    else if (parent_population_[b1].rank == parent_population_[b2].rank)
    {
      if (parent_population_[b1].crowding_distance > parent_population_[b2].crowding_distance)
        return b1;
      else
        return b2;
    }
    else
    {
      return b2;
    }
  }

  MedeaMapperThread::MedeaMapperThread(
      unsigned thread_id,
      config::CompoundConfig *config,
      std::string out_dir,
      problem::Workload &workload,
      model::Engine::Specs arch_specs,
      config::CompoundConfigNode arch_config,
      mapspace::MapSpace *mapspace,
      mapping::Constraints *constraints,
      std::vector<Individual> &immigrant_population,
      std::vector<Individual> &parent_population,
      std::vector<Individual> &population,
      Orchestrator *thread_orchestrator,
      std::mutex *global_mutex,
      uint32_t num_threads,
      uint32_t population_size,
      uint32_t immigrant_population_size,
      uint32_t generations,
      double fill_mutation_prob,
      double parallel_mutation_prob,
      double random_mutation_prob,
      bool use_tournament,
      std::string fast_accelergy_path,
      Mapping user_mapping,
      bool user_mapping_defined,
      RandomGenerator128 *if_rng,
      RandomGenerator128 *lp_rng,
      RandomGenerator128 *db_rng,
      RandomGenerator128 *sp_rng,
      RandomGenerator128 *crossover_rng) : thread_id_(thread_id),
                                           config_(config),
                                           out_dir_(out_dir),
                                           workload_(workload),
                                           arch_specs_(arch_specs),
                                           arch_config_(arch_config),
                                           arch_props_(arch_specs),
                                           mapspace_(mapspace),
                                           constraints_(constraints),
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
                                           fast_accelergy_path_(fast_accelergy_path),
                                           user_mapping_(user_mapping),
                                           user_mapping_defined_(user_mapping_defined),
                                           inj_gen_(9),
                                           thread_(),
                                           if_rng_(if_rng),
                                           lp_rng_(lp_rng),
                                           db_rng_(db_rng),
                                           sp_rng_(sp_rng),
                                           crossover_rng_(crossover_rng),
                                           rng_(thread_id),
                                           exp_distribution_(3.5),
                                           uni_distribution_(0, 1),
                                           tour_distribution_(0, population_size_ - 1)
  {
  }

  MedeaMapperThread::~MedeaMapperThread()
  {
  }

  void MedeaMapperThread::Start()
  {
    thread_ = std::thread(&MedeaMapperThread::Run, this);
  }

  void MedeaMapperThread::Join()
  {
    thread_.join();
  }

  void MedeaMapperThread::Run()
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

    // Wait for others
    thread_orchestrator_->FollowerWait(next_iteration_);
    for (uint32_t g = 0; g < num_generations_; g++)
    {
      // User defined map injection
      if (g == inj_gen_ && thread_id_ == num_threads_ - 1 && user_mapping_defined_)
      {
        InjectUserDefinedMapping(parent_population_, population_size_ - 1);
      }

      uint64_t debug_cross_count = 0;
      for (uint32_t ep = pop_slice_start; ep < pop_slice_end; ep += 2)
      {

        if (use_tournament_)
          Crossover(parent_population_[Tournament()].genome, parent_population_[Tournament()].genome,
                    population_[ep].genome, population_[ep + 1].genome);
        else
          Crossover(parent_population_[ep].genome, parent_population_[ep + 1].genome,
                    population_[ep].genome, population_[ep + 1].genome);

        Mutation(population_[ep]);
        Mutation(population_[ep + 1]);

        //std::cout << population_[ep].genome.PrintCompact() << std::endl;

        if (Evaluate(population_[ep].genome, population_[ep]))
          debug_cross_count++;
        else
          RandomIndividual(ep, population_);

        //std::cout << population_[ep+1].genome.PrintCompact() << std::endl;

        if (Evaluate(population_[ep + 1].genome, population_[ep + 1]))
          debug_cross_count++;
        else
          RandomIndividual(ep + 1, population_);
      }

      std::cout << "[T" << thread_id_ << "] Successfully evaluated " << debug_cross_count << "/" << pop_slice_end - pop_slice_start << " crossed mappings." << std::endl;

      // Immigration
      RandomPopulation(imm_pop_slice_start, imm_pop_slice_end, immigrant_population_);

      std::cout << "[T" << thread_id_ << "] Immigration completed" << std::endl;
      thread_orchestrator_->FollowerDone();

      // Wait for others and merging in main
      thread_orchestrator_->FollowerWait(next_iteration_);
    }
  }

}