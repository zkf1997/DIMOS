# DIMOS: Synthesizing Diverse Human Motions in 3D Indoor Scenes

[[website](https://zkf1997.github.io/DIMOS/)] [[paper](https://arxiv.org/abs/2305.12411)] 

```
@inproceedings{Zhao:ICCV:2023,
   title = {Synthesizing Diverse Human Motions in 3D Indoor Scenes},
   author = {Zhao, Kaifeng and Zhang, Yan and Wang, Shaofei and Beeler, Thabo and and Tang, Siyu},
   booktitle = {International conference on computer vision (ICCV)},
   year = {2023}
}
```

![teaser](https://zkf1997.github.io/DIMOS/images/teaser_canonical.png)

# License
* Third-party software and datasets employs their respective license. Here are some examples:
    * Code/model/data relevant to the SMPL-X body model follows its own license.
    * Code/model/data relevant to the AMASS dataset follows its own license.
    * Blender and its SMPL-X add-on employ their respective license.

* The rests employ the **Apache 2.0 license**.

# Updates
* Uploaded a variant locomotion model trained on cleaned walking motion data: [link](https://drive.google.com/drive/folders/1WAwXlM8qvawgSJ8ys_KNEX1Y4hI0Ec8u).
* Initial release with code and documentation for motion generation demos. Documentation for model training and data preprocessing will be updated later.  

# Installation

### Environment
* Tested on: Ubuntu 22.04, CUDA 11.8
* You may need to edit the `cudatoolkit` version.
```
conda env create -f env.yml
```
* If conda environment solving takes a long time, try to configure the [libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community/) solver. This requires conda version >= 22.11.
```
sudo conda update -n base conda

sudo conda install -n base conda-libmamba-solver

conda config --set solver libmamba
```
* After changing the solver, please try installing the environment again.

### Required External Data
* [SMPL-X body model](https://smpl-x.is.tue.mpg.de/)
* [VPoser v1.0](https://smpl-x.is.tue.mpg.de/)
* [CMU 41 marker placement](https://drive.google.com/file/d/1CcNBZCXA7_Naa0SGlYKCxk_ecnzftbSj/view?usp=sharing) from [link](http://mocap.cs.cmu.edu/)
* [SSM2 67 marker placement](https://drive.google.com/file/d/1ozQuVjXoDLiZ3YGV-7RpauJlunPfcx_d/view?usp=sharing) from [link](https://amass.is.tue.mpg.de/)
* [ShapeNet](https://shapenet.org/) (Optional if you only want to test demos)
* [Replica](https://github.com/facebookresearch/Replica-Dataset) (Optional if you only want to test demos)
* [PROX-S](https://github.com/zkf1997/COINS#prox-s-dataset) (Optional if you only want to test demos)

Please put the downloaded data in the `data` folder and organize as follows: 
```
├── data
│   ├── models_smplx_v1_1
│   │   ├── models
│   │   │   ├── markers
│   │   │   │   ├── CMU.json
│   │   │   │   ├── SSM2.json
│   │   │   ├── smplx
│   │   │   ├── vposer_v1_0
│   ├── proxs
│   ├── replica
│   ├── ShapeNetCore.v2
```

[//]: # (### Paths)

[//]: # (To run the code properly it is important to set the paths of data, body model, and others. )

[//]: # (* If you get into path problems with body models, marker settings, and motion data, please check the paths in `exp_GAMMAPrimitive/utils/config_env.py`. )

[//]: # (* For experiments requiring scene/object datasets including [PROX]&#40;https://prox.is.tue.mpg.de/&#41;, [ShapeNet]&#40;https://shapenet.org/&#41;, [Replica]&#40;https://github.com/facebookresearch/Replica-Dataset&#41;, you may need to set the dataset paths accordingly. )

### Pretrained Checkpoints and Test Data
Please download the [pretrained models and test data](https://drive.google.com/drive/folders/1AvM4GvdkG1OkggaQnggNeGmt2xgipKRU?usp=sharing), extract and copy it to the project folder.

### Visualization 
* We use `trimesh` and `pyrender` for visualization. Some Visualization code may not directly run on headless servers, adaptations will be needed.
* Please refer to [pyrender viewer](https://pyrender.readthedocs.io/en/latest/generated/pyrender.viewer.Viewer.html) for the usage of the interactive viewer.

# Motion Generation Demos
We provide various demos showcasing how to synthesize various human motions in general 3D scenes, and illustrate using an example scene included in the [download link](https://drive.google.com/drive/folders/1AvM4GvdkG1OkggaQnggNeGmt2xgipKRU?usp=sharing).

We assume the scene and objects are given as `ply` mesh files, with z-axis pointing up, and the floor plane is at z=0.
To generate object interactions, we assume the object instance is segmented and provided as a separate `ply` mesh.

Each demo is provided as a single script and can be adapted to general scenes you want to test by changing the scene/object mesh paths. 
By running the demos, you can generate [**random**]() motions in given scenes. 
Due to the stochastic nature of our method, the generated results will have varied quality and may have failures.
 

### Locomotion in 3D Scenes
This demo generates locomotion in 3D scenes given a pair of start and target locations. Collision-free waypoints are generated using navigation mesh-based path-finding. 

You can also choose to manually specify the path of waypoints in the script. Visualization of the navmesh path-finding result can be switched on/off using the `visualize` variable.

You may see some pop-up visualization windows for navigation mesh generation and path-finding. You can close the window after confirming the results are reasonable, then the generation will start.
* Generation:
  ```
  python synthesize/demo_locomotion.py
  ```
  From the log you can see the paths of the generated sequences, which will be used for visualization as below.

* Visualization:
  ```
  python vis_gen.py --seq_path 'results/locomotion/test_room/path_0/MPVAEPolicy_samp_collision/locomotion/policy_search/seq00*/results_ssm2_67_condi_marker_map_0.pkl'
  ```
  The argument `seq_path` specifies the result sequences you would like to visualize, and it supports glob paths to visualize multiple sequences at the same time.

  If you want to visualize a single sequence or animating multiple sequences has a low FPS on your machine, you can change `seq_path` to refer to a single result file, e.g., changing `seq00*` to `seq000`. 
  
  Explanation of more visualization arguments can be found in [vis_gen.py](./vis_gen.py). Results generated by other demos can be visualized in the same way.

  
Example visualization [video recording](https://drive.google.com/file/d/116x92ydUGMqErrkPysiPrU2xSQQsEUDR/view?usp=sharing).

### Fine-Grained Object Interaction
This demo generates interaction with a given object. The initial bodies are randomly sampled in front of the object. The goal interaction body markers can be loaded from provided data or generated using [COINS](https://github.com/zkf1997/COINS). 

For COINS generation, you need to build the PROX scenes following [PROX-S setup](https://github.com/zkf1997/COINS#prox-s-dataset) even if you do not use PROX scenes due to legacy code. 

Moreover, some of the static interaction generations from COINS can have inferior quality, and you may need to manually filter to stably get better motion results. 

* Generation:
  ```
  python synthesize/demo_interaction.py
  ```
* Visualization:
  ```
  python vis_gen.py --seq_path 'results/interaction/test_room/inter_sofa_sit_up_*/MPVAEPolicy_sit_marker/sit_1frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'
  ```
Example visualization [video recording](https://drive.google.com/file/d/1SRe46zL0m0_2u4gxZM1iamZWVs1nC1lw/view?usp=sharing).

### Combining Locomotion and Interaction
These demos show generating motions involving walking to an object and then perform interactions.
* Generation in the test scene:
  ```
  python synthesize/demo_loco_inter.py
  python vis_gen.py --seq_path 'results/interaction/test_room/loco_inter_sit_sofa_0_*_down/MPVAEPolicy_sit_marker/sit_2frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'
  ```
* Generation in reconstructed scenes from PROX and Replica:

  To run this demo, you need to first download the [PROX](https://prox.is.tue.mpg.de/) and [Replica](https://github.com/facebookresearch/Replica-Dataset) scenes and put them in `data/`. The scenes need to be preprocessed using the `get_scene.py` script.
  ```
  python synthesize/get_scene.py  
  python synthesize/demo_prox.py
  python vis_gen.py --seq_path 'results/interaction/MPH8/sit_bed_9_*_down/MPVAEPolicy_sit_marker/sit_2frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'
  python synthesize/demo_replica.py
  python vis_gen.py --seq_path 'results/interaction/room_0/sit_stool_39_*_down/MPVAEPolicy_sit_marker/sit_2frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'
  ```
Example visualization video recordings:
* [test scene](https://drive.google.com/file/d/1VRnIbwGh_v9sar6DO2YUSd2FiiXJW_RO/view?usp=sharing)
* [PROX](https://drive.google.com/file/d/1dE8Po5a8U1FjltCfUEDBqat551fAlcHa/view?usp=sharing)
* [Replica](https://drive.google.com/file/d/1XoG1HOV8406UrMLu0NcoH0lgCv9-y3QU/view?usp=sharing)

### Sequences of Alternating Locomotion and Interaction
This demo shows generating motions involving sequences of interaction events where the human alternates between the locomotion and object interaction stages.
This demo could probably show unnatural transition when switching locomotion/interaction stages. 
* Generation in the test scene:
  ```
  python synthesize/demo_chain.py
  ```
* Visualization:
  ```
  python vis_gen.py --seq_path 'results/interaction/test_room/chain_sit_sofa_0_*_down/MPVAEPolicy_sit_marker/sit_2frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'
  ```
Example visualization [video recording](https://drive.google.com/file/d/1z46dwYTkd11qJdJh6wFZhYH15d7GCIyk/view?usp=sharing).