#ifndef MEDEA_MAPPING_PARSER_FIX_H_
#define MEDEA_MAPPING_PARSER_FIX_H_

#include "mapping.hpp"
#include "model/engine.hpp"
#include "compound-config/compound-config.hpp"

#include "mapping/parser.hpp"
#include "mapping/arch-properties.hpp"

// Fix problems that occur when parsing mapping multiple times
// using timeloop ParseAndConstruct method caused by arch_props_
// state retention.

namespace mapping {

    extern ArchProperties arch_props_;

    Mapping ParseAndConstructFixed(config::CompoundConfigNode config, model::Engine::Specs& arch_specs, problem::Workload workload) {
        arch_props_ = {}; // Reset state
        return ParseAndConstruct(config, arch_specs, workload);
    }

}

#endif // MEDEA_MAPPING_PARSER_FIX_H_