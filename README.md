# medea
MEDEA: A Multi-objective Evolutionary Approach to DNN Hardware Mapping

## Installation
Install dependancies
```
sudo apt-get update
sudo apt-get install scons libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libyaml-cpp-dev libncurses-dev libtinfo-dev libgpm-dev git build-essential python3-pip
```
Install [Timeloop](https://github.com/NVlabs/timeloop) and [Accelergy](https://github.com/Accelergy-Project/accelergy)
```
mkdir timeloop-accelergy; cd timeloop-accelergy
git clone https://github.com/HewlettPackard/cacti.git
git clone https://github.com/Accelergy-Project/accelergy.git
git clone https://github.com/Accelergy-Project/accelergy-aladdin-plug-in.git
git clone https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git
git clone https://github.com/Accelergy-Project/accelergy-table-based-plug-ins.git
git clone -b v1.0 https://github.com/NVLabs/timeloop.git
cd cacti; make
cd ../accelergy; pip3 install .
cd ../accelergy-aladdin-plug-in/; pip3 install .
cd ../accelergy-cacti-plug-in/; pip3 install .
cp -r ../cacti ~/.local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/
cd ../accelergy-table-based-plug-ins/; pip3 install .
cd ../timeloop/src/; ln -s ../pat-public/src/pat .; cd ..
scons -j4 --accelergy --static; cd ../..
```
Install MEDEA
```
git clone https://github.com/Haimrich/medea.git
cd medea
scons -j4 timeloop_path="./timeloop-accelergy/timeloop/"
```



