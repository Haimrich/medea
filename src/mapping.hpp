#ifndef MEDEA_MAPPING_H_
#define MEDEA_MAPPING_H_

#include "mapping/mapping.hpp"
#include "mapping/arch-properties.hpp"

namespace medea
{
  class Mapping : public ::Mapping // Extending Timeloop Mapping Class
  {
  public:
    Mapping &operator=(const ::Mapping &m)
    {
      id = m.id;
      loop_nest = m.loop_nest;
      datatype_bypass_nest = m.datatype_bypass_nest;

      return *this;
    }

    //void LoadYaml(std::string& yaml_str);

    void DumpYaml(std::ostream& out, const std::vector<std::string> &storage_level_names, ArchProperties& arch_props) const;
    
  };
}

#endif