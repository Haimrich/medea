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

namespace medea
{

  struct Individual
  {
    Mapping genome;
    model::Engine engine;
    std::array<double, 3> objectives; // energy, latency, area
    uint32_t rank;
    double crowding_distance;
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