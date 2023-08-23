# DIMOS: Synthesizing Diverse Human Motions in 3D Indoor Scenes



[[website](https://zkf1997.github.io/DIMOS/)] [[paper](https://arxiv.org/abs/2305.12411)] 

![teaser](https://zkf1997.github.io/DIMOS/images/teaser_canonical.png)

# License
* Third-party software employs their respective license. Here are some examples:
    * Code/model/data relevant to the SMPL-X body model follows its own license.
    * Code/model/data relevant to the AMASS dataset follows its own license.
    * Blender and its SMPL-X add-on employ their respective license.

* The rests employ the **Apache 2.0 license**. When using the code, please cite our work as above.

# Updates
* Initial release with code and documentation for motion generation demos. Documentation for model training will be updated later.  

# Installation

### Environment
* Tested on: Ubuntu 22.04, CUDA 11.8
* You may need to edit the `cudatoolkit` version.
```
conda env create -f env.yml
```

### Required External Data
* [**SMPL-X** body model](https://smpl-x.is.tue.mpg.de/)
* [**VPoser v1.0**](https://smpl-x.is.tue.mpg.de/)
* [**CMU 41** marker placement](https://drive.google.com/file/d/1CcNBZCXA7_Naa0SGlYKCxk_ecnzftbSj/view?usp=sharing)
* [**SSM2 67** marker placement](https://drive.google.com/file/d/1ozQuVjXoDLiZ3YGV-7RpauJlunPfcx_d/view?usp=sharing)

Please organize the downloaded data as follows: 
```
├── models_smplx_v1_1
│   ├── models
│   │   ├── markers
│   │   │   ├── CMU.json
│   │   │   ├── SSM2.json
│   │   ├── smplx
│   │   ├── vposer_v1_0
```
### Paths
To run the code properly it is important to set the paths of data, body model, and others. 
* Please set the paths in `exp_GAMMAPrimitive/utils/config_env.py`. 
* For experiments requiring scene/object datasets including [PROX](https://prox.is.tue.mpg.de/), [ShapeNet](https://shapenet.org/), [Replica](https://github.com/facebookresearch/Replica-Dataset), you may need to set the dataset paths accordingly. 

### Pretrained Checkpoints and Test Data
Please download the [pretrained models and test data](https://drive.google.com/drive/folders/1AvM4GvdkG1OkggaQnggNeGmt2xgipKRU?usp=sharing), extract and copy it to the project folder.

### Visualization 
* We use `trimesh` and `pyrender` for visualization. Some Visualization code may not directly run on headless servers, adaptations will be needed.  

# Motion Generation Demos
We provide three demos showcasing how to synthesize various human motions in general 3D scenes, and illustrate using an example scene included in the [download link](https://drive.google.com/drive/folders/1AvM4GvdkG1OkggaQnggNeGmt2xgipKRU?usp=sharing).

We assume the scene is given as a `ply` mesh file, with z-axis pointing up, and the floor plane is at z=0.

To generate object interactions, we assume the object instance is segmented and provided as a separate `ply` mesh.

### Locomotion in 3D Scenes
This demo generates locomotion in 3D scenes given a pair of start and target locations. You can also choose to manually specify the path consisting of waypoints.
* Generation:
  ```
  python synthesize/demo_locomotion.py
  ```
* Visualization:
  ```
  python vis_gen.py --seq_path 'results/locomotion/test_room/path_0/MPVAEPolicy_samp_collision/locomotion/policy_search/seq00*/results_ssm2_67_condi_marker_map_0.pkl'
  ```
  The argument `seq_path` specifies the result sequences you would like to visualize, and it supports glob paths.
### Fine-Grained Object Interaction
This demo generates interaction with a given object. The initial bodies are randomly sampled in front of the object. The goal interaction body markers can be loaded from provided data or generated using [COINS](https://github.com/zkf1997/COINS). 
For COINS generation, you need to build the PROX scenes following [PROX-S setup](https://github.com/zkf1997/COINS#prox-s-dataset) due to legacy code. This will be changed later. Moreover, some of the static interaction generations from COINS can have inferior quality, and you may need to manually filter to stably get better motion results. 

* Generation:
  ```
  python synthesize/demo_interaction.py
  ```
* Visualization:
  ```
  python vis_gen.py --seq_path 'results/interaction/test_room/sofa_sit_up_*/MPVAEPolicy_babel_marker/sit_1frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'
  ```

### Chaining Locomotion and Interaction
This demos shows generating complex motions involving alternation of locomotion and object interaction stages.
* Generation:
  ```
  python synthesize/demo_chain.py
  ```
* Visualization:
  ```
  python vis_gen.py --seq_path 'results/interaction/test_room/sit_sofa_0_*_down/MPVAEPolicy_babel_marker/sit_2frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'
  ```


