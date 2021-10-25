#ifndef MEDEA_COMMON_H_
#define MEDEA_COMMON_H_

#include <condition_variable>
#include <vector>
#include <random>
#include <thread>
#include <algorithm>
#include <fstream>

#include "model/engine.hpp"
#include "model/topology.hpp"
#include "mapping/arch-properties.hpp"
#include "mapspaces/mapspace-base.hpp"

#include "yaml-cpp/yaml.h"

namespace medea
{

  class MinimalArchSpecs {
    struct Level {
      std::string name;
      int mesh_x, mesh_y;
      int size, effective_size;
    };

    std::vector<Level> levels;

  public:

    void reset(size_t size) { 
      levels.clear(); 
      levels.reserve(size);
    }

    void add(std::string name, int mesh_x, int mesh_y, int size, int effective_size) {
      levels.push_back((Level){name, mesh_x, mesh_y, size, effective_size});
    }

    void add(std::string name, int mesh_x, int mesh_y) {
      add(name, mesh_x, mesh_y, 0, 0);
    }

    friend YAML::Emitter& operator << (YAML::Emitter& out, const MinimalArchSpecs& arch)
    {
      if (arch.levels.empty()) return out;

      out << YAML::BeginSeq;

      out << YAML::BeginMap;
      out << YAML::Key << "name" << YAML::Value << arch.levels[0].name;
      out << YAML::Key << "mesh_x" << YAML::Value << arch.levels[0].mesh_x;
      out << YAML::Key << "mesh_y" << YAML::Value << arch.levels[0].mesh_y;
      out << YAML::EndMap;

      for (size_t i = 1; i < arch.levels.size(); i++) {
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

    friend std::ostream& operator << (std::ostream& out, const Individual& ind)
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

  class Orchestrator
  {
  private:
    uint64_t max_workers_;
    std::mutex sync_mutex_;
    std::condition_variable ready_;
    std::condition_variable done_;
    uint64_t current_workers_ = 0;
    uint64_t current_iteration_ = 0;

  public:
    Orchestrator(uint64_t max_workers) : max_workers_(max_workers)
    {
    }

    void LeaderDone()
    {
      std::unique_lock<std::mutex> lock(sync_mutex_);
      current_workers_ = max_workers_;
      ++current_iteration_;
      ready_.notify_all();
    }

    void LeaderWait()
    {
      std::unique_lock<std::mutex> lock(sync_mutex_);
      done_.wait(lock, [&]
                 { return current_workers_ == 0; });
    }

    void FollowerWait(uint64_t &next_iteration)
    {
      std::unique_lock<std::mutex> lock(sync_mutex_);
      ready_.wait(lock, [&]
                  { return current_iteration_ == next_iteration; });
      lock.unlock();

      ++next_iteration;
    }

    void FollowerDone()
    {
      std::unique_lock<std::mutex> lock(sync_mutex_);
      if (--(current_workers_) == 0)
      {
        lock.unlock();
        done_.notify_one();
      }
    }
  };

  template <class Iter>
  class Iterange
  {
    Iter start_;
    Iter end_;

  public:
    Iterange(Iter start, Iter end) : start_(start), end_(end) {}

    Iter begin() { return start_; }
    Iter end() { return end_; }
  };

  using LoopRange = Iterange<std::vector<loop::Descriptor>::const_iterator>;

}

#endif // MEDEA_COMMON_H_