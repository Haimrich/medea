#include "mapping.hpp"

#include "yaml-cpp/yaml.h"

namespace medea
{

  void Mapping::DumpYaml(std::ostream &out, const std::vector<std::string> &storage_level_names, ArchProperties& arch_props) const
  {
    auto num_storage_levels = loop_nest.storage_tiling_boundaries.size();
    YAML::Emitter yaml;
    yaml << YAML::BeginMap << YAML::Key << "mapping" << YAML::BeginSeq;

    // Datatype Bypass.
    auto mask_nest = tiling::TransposeMasks(datatype_bypass_nest);
    for (unsigned level = 0; level < num_storage_levels; level++)
    {
      yaml << YAML::BeginMap;

      yaml << YAML::Key << "target" << YAML::Value << storage_level_names.at(level);
      yaml << YAML::Key << "type" << YAML::Value << "datatype";

      auto &compound_mask = mask_nest.at(level);
      std::vector<std::string> keep_dataspace_names;
      std::vector<std::string> bypass_dataspace_names;
      for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
      {
        problem::Shape::DataSpaceID pv = problem::Shape::DataSpaceID(pvi);
        if (compound_mask.at(pv))
          keep_dataspace_names.push_back(problem::GetShape()->DataSpaceIDToName.at(pv));
        else
          bypass_dataspace_names.push_back(problem::GetShape()->DataSpaceIDToName.at(pv));
      }
      yaml << YAML::Key << "keep" << YAML::Flow << keep_dataspace_names;
      yaml << YAML::Key << "bypass" << YAML::Flow << bypass_dataspace_names;

      yaml << YAML::EndMap;
    }

    // Factors and Permutations.
    unsigned loop_level = 0;
    for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
    {
      std::map<spacetime::Dimension, std::string> permutations;
      std::map<spacetime::Dimension, std::map<problem::Shape::DimensionID, unsigned>> factors;
      unsigned spatial_split;

      for (unsigned sdi = 0; sdi < unsigned(spacetime::Dimension::Num); sdi++)
      {
        auto sd = spacetime::Dimension(sdi);
        permutations[sd] = "";
        for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
          factors[sd][problem::Shape::DimensionID(idim)] = 1;
      }

      for (; loop_level <= loop_nest.storage_tiling_boundaries.at(storage_level); loop_level++)
      {
        auto &loop = loop_nest.loops.at(loop_level);
        if (loop.end > 1)
        {
          factors.at(loop.spacetime_dimension).at(loop.dimension) = loop.end;
          permutations.at(loop.spacetime_dimension) += problem::GetShape()->DimensionIDToName.at(loop.dimension);
        }

      }

      // Determine X-Y split.
      spatial_split = permutations.at(spacetime::Dimension::SpaceX).size();

      // Merge spatial X and Y factors and permutations.
      std::string spatial_permutation =
          permutations.at(spacetime::Dimension::SpaceX) +
          permutations.at(spacetime::Dimension::SpaceY);
      
      if ( spatial_permutation.size() > 0 || arch_props.FanoutX(storage_level) > 1 || arch_props.FanoutY(storage_level) > 1 ) 
      {
        std::string spatial_factor_string = "";

        std::map<problem::Shape::DimensionID, unsigned> spatial_factors;
        for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
        {
          auto dim = problem::Shape::DimensionID(idim);
          spatial_factors[dim] =
              factors.at(spacetime::Dimension::SpaceX).at(dim) *
              factors.at(spacetime::Dimension::SpaceY).at(dim);

          spatial_factor_string += problem::GetShape()->DimensionIDToName.at(dim);
          char factor[8];
          sprintf(factor, "%d", spatial_factors.at(dim));
          spatial_factor_string += factor;
          if (idim != unsigned(problem::GetShape()->NumDimensions) - 1)
            spatial_factor_string += " ";

          // If the factor is 1, concatenate it to the permutation.
          if (spatial_factors.at(dim) == 1)
            spatial_permutation += problem::GetShape()->DimensionIDToName.at(dim);
        }

        yaml << YAML::BeginMap;
        yaml << YAML::Key << "target" << YAML::Value << storage_level_names.at(storage_level);
        yaml << YAML::Key << "type" << YAML::Value << "spatial";
        yaml << YAML::Key << "factors" << YAML::Value << spatial_factor_string;
        yaml << YAML::Key << "permutation" << YAML::Value << spatial_permutation;
        yaml << YAML::Key << "split" << YAML::Value << static_cast<int>(spatial_split);
        yaml << YAML::EndMap;
      }

      auto &temporal_permutation = permutations.at(spacetime::Dimension::Time);
      auto &temporal_factors = factors.at(spacetime::Dimension::Time);
      std::string temporal_factor_string = "";

      // Temporal factors: if the factor is 1, concatenate it into the permutation.
      for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
      {
        auto dim = problem::Shape::DimensionID(idim);

        temporal_factor_string += problem::GetShape()->DimensionIDToName.at(dim);
        char factor[8];
        sprintf(factor, "%d", temporal_factors.at(dim));
        temporal_factor_string += factor;
        if (idim != unsigned(problem::GetShape()->NumDimensions) - 1)
          temporal_factor_string += " ";

        if (temporal_factors.at(dim) == 1)
          temporal_permutation += problem::GetShape()->DimensionIDToName.at(dim);
      }

      yaml << YAML::BeginMap;
      yaml << YAML::Key << "target" << YAML::Value << storage_level_names.at(storage_level);
      yaml << YAML::Key << "type" << YAML::Value << "temporal";
      yaml << YAML::Key << "factors" << YAML::Value << temporal_factor_string;
      yaml << YAML::Key << "permutation" << YAML::Value << temporal_permutation;
      yaml << YAML::EndMap;
    }

    yaml << YAML::EndSeq << YAML::EndMap;

    out << yaml.c_str() << std::endl;
  }
}