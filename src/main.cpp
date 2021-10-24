#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "compound-config/compound-config.hpp"

#include "mapper.hpp"
#include "negotiator.hpp"

using namespace std;
using namespace medea;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char *argv[])
{
  cout << R"(
   __  __ ___ ___  ___   _    
  |  \/  | __|   \| __| /_\
  | |\/| | _|| |) | _| / _ \
  |_|  |_|___|___/|___/_/ \_\
  )" << endl;

  try
  {

    po::options_description desc("MEDEA: A Multi-objective Evolutionary Approach to DNN Hardware Mapping\nAllowed options:");
    desc.add_options()
      ("help", "produce help message")
      ("input-files,i", po::value<vector<string>>(), "input files")
      ("output,o", po::value<string>()->default_value("."), "output directory")
      ("workload-dir,w", po::value<string>(), "workloads input directory")
      ("lookup,l", po::value<vector<int>>()->multitoken(), "layer workload lookup")
      ("map,m", "Map only, don't negotiate.")
    ;

    po::positional_options_description p;
    p.add("input-files", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
      cout << desc << "\n";
      return 0;
    }

    if (!vm.count("input-files"))
    {
      cout << "Missing input files. \n";
      return 0;
    }

    vector<fs::path> workloads;
    if (!vm.count("workload-dir")) {
      cout << "Missing workload input folder. Searching for a workload in input files. \n";
      // TODO
    } else {
      fs::path workload_dir( vm["workload-dir"].as<string>() );

      if (fs::exists(workload_dir) && fs::is_directory(workload_dir)) {
        for (const auto & p : fs::directory_iterator(workload_dir))
          if (fs::is_regular_file(p) && p.path().extension() == ".yaml")
            workloads.emplace_back(p);
            
        sort(workloads.begin(), workloads.end());

        if (workloads.size()) {
          cout << "Found " << workloads.size() << " workloads in " << workload_dir.string() << endl; 
        } else {
          cerr << "Error: No workload found in " << workload_dir.string() << endl; 
          return 1;
        }
        
      } else {
        cerr << "Error: " << workload_dir.string() << " : No such directory." << endl;
        return 1;
      }
    }

    if (vm.count("map"))
    {
      cout << "Mapping only. \n";
      // TODO
    }

    

    for (int i : vm["lookup"].as<vector<int>>())
    {
      cout << i << " ";
    }
    cout << "\n";

    vector<string> input_files = vm["input-files"].as<vector<string>>();
    string out_dir = vm["output"].as<string>();

    auto pre_config = new config::CompoundConfig(input_files);
    
    if (workloads.size()) { // Run Mapper for each workload 
      if (pre_config->getRoot().exists("problem")) {
        cerr << "Error: both a workload input directory and a workload input file were provided." << endl;
        return 1;
      }

      for (auto& workload : workloads) {
        // TODO: Skip already mapped

        string workload_name = workload.stem().string();
        fs::create_directories(out_dir + "/" + workload_name + "/pareto");

        vector<string> input_files_w = input_files;
        input_files_w.push_back(workload.string());
        auto config = new config::CompoundConfig(input_files_w); 
        if (!config->getRoot().exists("problem")) {
          cerr << "Error:  " + workload.string() + " is not a workload specification." << endl;
          return 1;
        } 
        
        MedeaMapper mapper(config, out_dir + "/" + workload_name);
        mapper.Run();
      }

    } else { // Run Mapper for the only workload provided
      fs::create_directories(out_dir + "/pareto");
      MedeaMapper mapper(pre_config, out_dir);
      mapper.Run();
    }
  }
  catch (exception &e)
  {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }
  catch (...)
  {
    cerr << "Exception of unknown type!" << endl;
    return 1;
  }

  return 0;
}

bool gTerminateEval = false;

