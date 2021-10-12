# MEDEA: A Multi-objective Evolutionary Approach to DNN Hardware Mapping

Multi-Objective Design Space Exploration framework leveraging [Timeloop](https://github.com/NVlabs/timeloop) model.

## Installation
Install dependancies
```
sudo apt-get update
sudo apt-get install libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libyaml-cpp-dev  git build-essential python3-pip cmake
```
Install [Accelergy](https://github.com/Accelergy-Project/accelergy)
```
mkdir accelergy; cd accelergy
git clone https://github.com/HewlettPackard/cacti.git
git clone https://github.com/Accelergy-Project/accelergy.git
git clone https://github.com/Accelergy-Project/accelergy-aladdin-plug-in.git
git clone https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git
git clone https://github.com/Accelergy-Project/accelergy-table-based-plug-ins.git
cd cacti; make
cd ../accelergy; pip3 install .
cd ../accelergy-aladdin-plug-in/; pip3 install .
cd ../accelergy-cacti-plug-in/; pip3 install .
cp -r ../cacti ~/.local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/
cd ../accelergy-table-based-plug-ins/; pip3 install .

accelergy; accelergyTables
```
Install MEDEA
```
git clone --recursive https://github.com/Haimrich/medea.git
cd medea/build
cmake ..
make
```



