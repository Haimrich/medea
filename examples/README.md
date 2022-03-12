
# Examples

Running `medea -h`:

```console
root@ff0c3f2100f1:/app/examples# medea -h

   __  __ ___ ___  ___   _    
  |  \/  | __|   \| __| /_\
  | |\/| | _|| |) | _| / _ \
  |_|  |_|___|___/|___/_/ \_\
  
MEDEA: A Multi-objective Evolutionary Approach to DNN Hardware Mapping
Allowed options:
  -h [ --help ]            Print help information.
  -i [ --input-files ] arg Input .yaml config files. These should specify 
                           paremeters about the architecture, arch. components,
                           datatype bypass, medea search. In map mode, if an 
                           input directory is not specified, it is possible to 
                           provide a single workload file as input.
  -o [ --output ] arg (=.) Output directory
  -d [ --input-dir ] arg   Input directory. In default (map and negotiate) and 
                           map only mode this directoryshould contain a set of 
                           workload specifications .yaml file. In negotiate 
                           only mode this directory should contain the output 
                           of a previous Medea mapping run.
  -l [ --lookup ] arg      Layer workload lookup. Ex. -l 0 1 2 2 3 3 4 . This 
                           provide, for each actual network layer (ex. AlexNet 
                           layers) the index of the workload in the input 
                           directory (in alphabetic order) that matches its 
                           dimensions (filter sizes, number of channels, ecc.).
                           This because some layers in networks share the same 
                           dimensions and can be mapped once.
  -m [ --map ]             Map only, don't negotiate.
  -n [ --negotiate ]       Negotiate only, don't map.
  -c [ --clean ]           By default, if in the output folder there is already
                           mapping output data for a specific workload, Medea 
                           skips its mapping. To ignore previous run outputs 
                           use this flag.

```

### Example 1 - Mapping a layer

In this example we run the MEDEA genetic algorithm in order to find the pareto set of mappings for the first layer of ResNet50.
```
medea medea.yaml \
      arch/architecture_v3.yaml \
      arch/components/*.yaml \
      arch/bypass.yaml \
      workloads/resnet50/001_7_112_3_64_2.yaml \
      -m -o outputs/e1/resnet50_001/
```

### Example 2 - Mapping and negotiating

In this example MEDEA map all the layers of VGG16 network. Then the framework will negotiate overall pareto point that consists in a combination of mappings (one for each layer) and the minimal architecture needed to execute them.
```
medea medea.yaml \
      arch/architecture_v3.yaml \
      arch/components/*.yaml \
      arch/bypass.yaml \
      -d workloads/vgg16/ \
      -l 0 1 2 3 4 5 5 6 7 7 8 8 8 9 10 11 \
      -o outputs/e2/vgg16/
```