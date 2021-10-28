#ifndef MEDEA_INDIVIDUAL_H_
#define MEDEA_INDIVIDUAL_H_

#include "model/engine.hpp"
#include "model/topology.hpp"
#include "mapping.hpp"

#include "mapspaces/mapspace-base.hpp"

#include "yaml-cpp/yaml.h"

namespace medea
{
  class MinimalArchSpecs
  {
    struct Level
    {
      std::string name;
      int mesh_x, mesh_y;
      int size;
    };

    std::vector<Level> levels;

  public:
    void reset(size_t size)
    {
      levels.clear();
      levels.reserve(size);
    }

    void add(std::string name, int mesh_x, int mesh_y, int size)
    {
      levels.push_back((Level){name, mesh_x, mesh_y, size});
    }

    void add(std::string name, int mesh_x, int mesh_y)
    {
      add(name, mesh_x, mesh_y, 0);
    }

    friend YAML::Emitter &operator<<(YAML::Emitter &out, const MinimalArchSpecs &arch)
    {
      if (arch.levels.empty())
        return out;

      out << YAML::BeginSeq;

      out << YAML::BeginMap;
      out << YAML::Key << "name" << YAML::Value << arch.levels[0].name;
      out << YAML::Key << "mesh_x" << YAML::Value << arch.levels[0].mesh_x;
      out << YAML::Key << "mesh_y" << YAML::Value << arch.levels[0].mesh_y;
      out << YAML::EndMap;

      for (size_t i = 1; i < arch.levels.size(); i++)
      {
        out << YAML::BeginMap;
        out << YAML::Key << "name" << YAML::Value << arch.levels[i].name;
        out << YAML::Key << "mesh_x" << YAML::Value << arch.levels[i].mesh_x;
        out << YAML::Key << "mesh_y" << YAML::Value << arch.levels[i].mesh_y;
        out << YAML::Key << "size" << YAML::Value << arch.levels[i].size;
        out << YAML::EndMap;
      }
      out << YAML::EndSeq;

      return out;
    }


    MinimalArchSpecs(const YAML::Node &yaml) {
      add(yaml[0]["name"].as<std::string>(), yaml[0]["mesh_x"].as<int>(), yaml[0]["mesh_y"].as<int>());

      for (size_t i=1; i < yaml.size(); i++) 
        add(yaml[i]["name"].as<std::string>(), yaml[i]["mesh_x"].as<int>(), yaml[i]["mesh_y"].as<int>(), yaml[i]["size"].as<int>());
    }

    MinimalArchSpecs() = default;

  };

  class Individual
  {
  public:
    Mapping genome;
    model::Engine engine;
    std::array<double, 3> objectives; // energy, latency, area
    uint32_t rank;
    double crowding_distance;
    MinimalArchSpecs arch;

    friend std::ostream &operator<<(std::ostream &out, const Individual &ind)
    {
      YAML::Emitter yout;

      yout << YAML::BeginMap;
      yout << YAML::Key << "stats" << YAML::Value << YAML::BeginMap;
      yout << YAML::Key << "energy" << YAML::Value << ind.engine.Energy();
      yout << YAML::Key << "cycles" << YAML::Value << ind.engine.Cycles();
      yout << YAML::Key << "area" << YAML::Value << ind.engine.Area();
      yout << YAML::EndMap;
      yout << YAML::Key << "arch" << YAML::Value << ind.arch;
      yout << YAML::EndMap;

      out << yout.c_str();
      return out;
    }
  };

  typedef std::vector<Individual> Population;
}

#endif // MEDEA_INDIVIDUAL_H_
