#include <boost/program_options.hpp>

#include "compound-config/compound-config.hpp"

#include "medea-mapper.hpp"

using namespace std;
using namespace medea;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  try {

      po::options_description desc("Allowed options");
      desc.add_options()
          ("help", "produce help message")
          ("input-files", po::value< vector<string> >(), "input files")
          ("o", po::value<std::string>()->default_value("."), "output directory")
      ;

      po::positional_options_description p;
      p.add("input-files", -1);

      po::variables_map vm;
      po::store(po::command_line_parser(argc, argv).
                options(desc).positional(p).run(), vm);
      po::notify(vm);

      if (vm.count("help")) {
          cout << desc << "\n";
          return 0;
      }

      if (!vm.count("input-files")) {
          cout << "Missing input files." << "\n";
          return 0;
      }

      
      cout << " __  __ ___ ___  ___   _  \n ";
      cout << "|  \\/  | __|   \\| __| /_\\  \n";
      cout << "| |\\/| | _|| |) | _| / _ \\ \n";
      cout << "|_|  |_|___|___/|___/_/ \\_\\\n\n";

      vector<string> input_files =  vm["input-files"].as< vector<string> >();
      auto config = new config::CompoundConfig(input_files);

      Medea application(config, vm["o"].as<string>());  
      application.Run();

  }
  catch(exception& e) {
      cerr << "Error: " << e.what() << "\n";
      return 1;
  }
  catch(...) {
      cerr << "Exception of unknown type!\n";
  }

  return 0;
}

bool gTerminateEval = false;
