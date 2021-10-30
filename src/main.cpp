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

    po::options_description desc(
      "MEDEA: A Multi-objective Evolutionary Approach to DNN Hardware Mapping\n"
      "Allowed options"
    );
    desc.add_options()
      ("help,h", "Print help information.")
      ("input-files,i", po::value<vector<string>>(), 
        "Input .yaml config files. These should specify paremeters about the architecture, "
        "arch. components, datatype bypass, medea search. In map mode, if an input directory "
        "is not specified, it is possible to provide a single workload file as input."
      )
      ("output,o", po::value<string>()->default_value("."), "Output directory")
      ("input-dir,d", po::value<string>(), 
        "Input directory. In default (map and negotiate) and map only mode this directory" 
        "should contain a set of workload specifications .yaml file. In negotiate only "
        "mode this directory should contain the output of a previous Medea mapping run.")
      ("lookup,l", po::value<vector<size_t>>()->multitoken(), 
        "Layer workload lookup. Ex. -l 0 1 2 2 3 3 4 . This provide, for each actual network layer "
        "(ex. AlexNet layers) the index of the workload in the input directory (in alphabetic order) "
        "that matches its dimensions (filter sizes, number of channels, ecc.). This because some "
        "layers in networks share the same dimensions and can be mapped once."
      )
      ("map,m", "Map only, don't negotiate.")
      ("negotiate,n", "Negotiate only, don't map.")
      ("clean,c", "By default, if in the output folder there is already mapping output data for "
       "a specific workload, Medea skips its mapping. To ignore previous run outputs use this flag.")
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

    /* ==== Preparation ==== */

    std::string input_dir;
    fs::path input_dir_path;
    if (vm.count("input-dir")) {
      input_dir = vm["input-dir"].as<string>();
      input_dir_path = fs::path(input_dir);
    } else {
      input_dir = ".";
    }

    vector<size_t> layer_workload_lookup = vm.count("lookup") ? 
      vm["lookup"].as<vector<size_t>>() : vector<size_t>();

    vector<string> input_files = vm["input-files"].as<vector<string>>();
    string out_dir = vm["output"].as<string>();

    auto pre_config = new config::CompoundConfig(input_files);
    Accelergy accelergy(pre_config);


    /* ====  PER-WORKLOAD MAPPING PHASE  ==== */
    if (!vm.count("negotiate")) {

      vector<fs::path> workloads;
      if (!vm.count("input-dir")) {
        cout << "Missing workload input folder. Searching for a workload in input files." << endl;
      } else {

        if ( fs::is_directory(input_dir_path) ) {
          for (const auto & p : fs::directory_iterator(input_dir_path))
            if (fs::is_regular_file(p) && p.path().extension() == ".yaml")
              workloads.emplace_back(p);
              
          sort(workloads.begin(), workloads.end());

          if (workloads.size()) {
            cout << "Found " << workloads.size() << " workloads in " << input_dir_path.string() << endl; 
          } else {
            cerr << "Error: No workload found in " << input_dir_path.string() << endl; 
            return 1;
          }
        } else {
          cerr << "Error: " << input_dir_path.string() << " : No such directory." << endl;
          return 1;
        }
      }

      if (workloads.size()) { // Run Mapper for each workload 
        if (pre_config->getRoot().exists("problem")) {
          cerr << "Error: both a workload input directory and a workload input file provided." << endl;
          return 1;
        }

        for (size_t i = 0; i < workloads.size(); i++) {
          // TODO: Skip already mapped

          cout << "[MEDEA] Started mapping of the workload " << i+1 << "/" << workloads.size() << "." << endl;

          string workload_name = workloads[i].stem().string();
          fs::create_directories(out_dir + "/" + workload_name + "/pareto");

          vector<string> input_files_w = input_files;
          input_files_w.push_back(workloads[i].string());
          auto config = new config::CompoundConfig(input_files_w); 
          if (!config->getRoot().exists("problem")) {
            cerr << "Error:  " + workloads[i].string() + " is not a workload specification." << endl;
            return 1;
          } 
          
          MedeaMapper mapper(config, out_dir + "/" + workload_name, accelergy);
          mapper.Run();

          cout << "[MEDEA] Workload mapping completed." << endl;
        }

      } else { // Run Mapper for the only workload provided
        fs::create_directories(out_dir + "/pareto");
        MedeaMapper mapper(pre_config, out_dir, accelergy);
        mapper.Run();
      }
    }

    /* ==== END-TO-END NEGOTIATION PHASE ==== */
    if (vm.count("map")) {
      cout << "[MEDEA] Mapping only flag provided. Skipping negotiation." << endl;
    } else {
      if (!vm.count("negotiate")) input_dir = out_dir;
      
      MedeaNegotiator negotiator(pre_config, input_dir, layer_workload_lookup, out_dir, accelergy);
      unsigned num_design_points = negotiator.Run();
      
      cout << "[MEDEA] Negotiation completed. " << num_design_points << " pareto design points found." << endl;
    }


    delete pre_config;
  
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

