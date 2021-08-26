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

#include "medea.hpp"
#include "compound-config/compound-config.hpp"
#include "util/args.hpp"

bool gTerminate = false;
bool gTerminateEval = false;

using namespace medea;

//--------------------------------------------//
//                    MAIN                    //
//--------------------------------------------//

int main(int argc, char* argv[])
{
  assert(argc >= 2);

  std::vector<std::string> input_files;
  std::string output_dir = ".";

  bool success = ParseArgs(argc, argv, input_files, output_dir);
  (void) success; // TODO
  auto config = new config::CompoundConfig(input_files);

  std::vector<std::string> banner = {
    " __  __ ___ ___  ___   _   ",
    "|  \\/  | __|   \\| __| /_\\  ",
    "| |\\/| | _|| |) | _| / _ \\ ",
    "|_|  |_|___|___/|___/_/ \\_\\",
  };
  
  for (auto& line: banner)
    std::cout << line << std::endl;
  std::cout << std::endl;

  Medea application(config, output_dir);  
  application.Run();

  return 0;
}
