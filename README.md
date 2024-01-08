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
* Created the documentation for data preprocessing and model training. Updated the motion generation demos.  
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
* [AMASS](https://amass.is.tue.mpg.de/) (Optional if you only want to test demos)
* [SAMP](https://samp.is.tue.mpg.de/) (Optional if you only want to test demos)
* [BABEL](https://babel.is.tue.mpg.de/) (Optional if you only want to test demos)

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
│   ├── amass
│   │   ├── smplx_g
│   │   │   ├── ACCAD
│   │   │   ├── BMLMovi
│   │   │   ├── ...
│   │   ├── babel_v1-0_release
│   ├── samp
```

[//]: # (### Paths)

[//]: # (To run the code properly it is important to set the paths of data, body model, and others. )

[//]: # (* If you get into path problems with body models, marker settings, and motion data, please check the paths in `exp_GAMMAPrimitive/utils/config_env.py`. )

[//]: # (* For experiments requiring scene/object datasets including [PROX]&#40;https://prox.is.tue.mpg.de/&#41;, [ShapeNet]&#40;https://shapenet.org/&#41;, [Replica]&#40;https://github.com/facebookresearch/Replica-Dataset&#41;, you may need to set the dataset paths accordingly. )

### Pretrained Checkpoints and Test Data
Please download the [pretrained models and test data](https://drive.google.com/drive/folders/1AvM4GvdkG1OkggaQnggNeGmt2xgipKRU?usp=sharing), extract and copy it to the project folder.

#### Description of Policy Checkpoints
* Locomotion policies
  * _MPVAEPolicy_frame_label_walk_collision/map_walk_: Locomotion control policy trained with collision avoidance. 
  Its motion primitive model is trained on walking-only motion data. 
  This policy is trained with goal reaching termination which terminates training episodes when the agent reaches the goal. 
  Empirically this policy generates faster movements and the virtual humans go pass the goals. Can be used for locomotion-only generation.

  * _MPVAEPolicy_frame_label_walk_collision/map_nostop_: Locomotion control policy trained with collision avoidance, **without** the goal reaching termination.
  Its motion primitive model is trained on walking-only motion data.
  Empirically this policy generates slower movements and the virtual humans tend to reach and stay at the goals.
  More suitable for scenarios where the agent needs to reach the goal and perform object interactions.

  * _MPVAEPolicy_samp_collision/locomotion_: Locomotion control policy whose motion primitive model is trained on SAMP motion data that contain sitting and lying motions. This policy might sometimes demonstrate unwanted sitting motion during locomotion.

* Interaction policies
  * _MPVAEPolicy_sit_marker/sit_1frame_: Sitting control policy whose motion primitive model is trained on interaction-related BABEL motion data. This policy is trained with initial motion seeds of 1 frame.
  * _MPVAEPolicy_sit_marker/sit_2frame_: Sitting control policy whose motion primitive model is trained on interaction-related BABEL motion data. This policy is trained with initial motion seeds of 2 frames. This policy generates slightly smoother transition.
  
  * Lying policies follow similar rules as sitting policies. 


### Visualization 
* We use `trimesh` and `pyrender` for visualization. Some Visualization code may not directly run on headless servers, adaptations will be needed.
* Please refer to [pyrender viewer](https://pyrender.readthedocs.io/en/latest/generated/pyrender.viewer.Viewer.html) for the usage of the interactive viewer.

# Motion Generation Demos
We provide various demos showcasing how to synthesize various human motions in general 3D scenes, and illustrate using an example scene included in the [download link](https://drive.google.com/drive/folders/1AvM4GvdkG1OkggaQnggNeGmt2xgipKRU?usp=sharing).

We assume the scene and objects are given as `ply` mesh files, with z-axis pointing up, and the floor plane is at z=0.
To generate object interactions, we assume the object instance is segmented and provided as a separate `ply` mesh.

Each demo is provided as a single script and can be adapted to general scenes you want to test by changing the scene/object mesh paths. 
By running the demos, you can generate [**random**]() motions in given scenes. 
**Due to the stochastic nature of our method, the generated results will have varied quality and may have failures.**
 

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
  python vis_gen.py --seq_path 'results/locomotion/test_room/test_room_path_0/MPVAEPolicy_frame_label_walk_collision/map_walk/policy_search/seq00*/results_ssm2_67_condi_marker_map_0.pkl'
  ```
  The argument `seq_path` specifies the result sequences you would like to visualize, and it supports glob paths to visualize multiple sequences at the same time.

  If you want to visualize a single sequence or animating multiple sequences has a low FPS on your machine, you can change `seq_path` to refer to a single result file, e.g., changing `seq00*` to `seq000`. 
  
  Explanation of more visualization arguments can be found in [vis_gen.py](./vis_gen.py). Results generated by other demos can be visualized in the same way.

  
Example visualization [video recording](https://drive.google.com/file/d/116x92ydUGMqErrkPysiPrU2xSQQsEUDR/view?usp=sharing).

### Fine-Grained Object Interaction
This demo generates interaction with a given object. The initial bodies are randomly sampled in front of the object. The goal interaction body markers can be loaded from provided data or generated using [COINS](https://github.com/zkf1997/COINS). 

For COINS generation, you need to build the PROX scenes following [PROX-S setup](https://github.com/zkf1997/COINS#prox-s-dataset) even if you do not use PROX scenes due to legacy code. 

Moreover, some of the static interaction generations from COINS can have inferior quality, and you may need to manually filter the interaction goals to get more stable motion results. 

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
  python vis_gen.py --seq_path 'results/interaction/room_0/sit_chair_74_*_down/MPVAEPolicy_sit_marker/sit_2frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'
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

# Training

## Data Preparation

* Motion primitive data:
 
  We process AMASS and SAMP data to motion primitives, please run each command twice with num_motion_primitives=1 and num_motion_primitives=10.
  * SAMP data: 
  ```
  python exp_GAMMAPrimitive/utils/utils_canonicalize_samp.py [num_motion_primitives]
  ```

  * AMASS data, BABEL sequence label: 
  ```
  python exp_GAMMAPrimitive/utils/utils_canonicalize_babel.py [num_motion_primitives]
  ```

  * AMASS data, BABEL frame label: 
  ```
  python exp_GAMMAPrimitive/utils/utils_canonicalize_babel_frame_label.py [num_motion_primitives]
  ```


* Object mesh data: 
  
  We use [ShapeNet](https://shapenet.org/) for data of objects with real sizes. We export a small subset of objects with the following script:
  
  ```
  python utils/export_shapenet_real.py
  ```



* Random scene and navigation mesh:
  
  We train the locomotion control policy on random scenes generated using ShapeNet objects. The random scenes can be generated using the following script:
  ```
  python test_navmesh.py
  ```

* Goal interaction data:

  Training the object interaction control policies requires goal interaction data to initialize the motion seed or the interaction goals. 
We provide the goal interaction data at [data/interaction](https://drive.google.com/drive/folders/1AvM4GvdkG1OkggaQnggNeGmt2xgipKRU?usp=sharing). 
The goal interactions are obtained by retargeting the [PROX](https://prox.is.tue.mpg.de/) data to ShapeNet objects.

## Motion Primitive Model Training
We follow the motion primitive model training procedure in [GAMMA](https://github.com/yz-cnsdqz/GAMMA-release?tab=readme-ov-file#train).
We keep the GAMMA body regressor untouched, and train marker predictors separately for locomotion, sitting, and lying. Each motion primitive model have two variants of 1 and 2 frames of motion seed.

The marker predictor training contains two stages of training: we first train the model to predict one motion primitive, and then finetune the model with rollout training using 10 motion primitives sequences to improve long-term stability.

Please train each marker predictor following the description below:

* Train from scratch
  ```
  python exp_GAMMAPrimitive/train_GAMMAPredictor.py --cfg [1MP_CONFIG_NAME]
  ```
* Copy last checkpoint and rename

  Copy the last checkpoint in `results/GAMMAPrimitive/[1MP_CONFIG_NAME]/checkpoints` to `results/GAMMAPrimitive/[10MP_CONFIG_NAME]/checkpoints`, and rename the checkpoint as `epoch-000.ckp`.
    
* Finetune with rollout training
  ```
  python exp_GAMMAPrimitive/train_GAMAPredictor.py --cfg [10MP_CONFIG_NAME] --resume_training 1
  ```

List of config names:
* Locomotion:
  * 1 MP: [MPVAE_frame_label_walk_1frame](exp_GAMMAPrimitive/cfg/MPVAE_frame_label_walk_1frame.yml), [MPVAE_frame_label_walk_2frame](exp_GAMMAPrimitive/cfg/MPVAE_frame_label_walk_2frame.yml) 
  * 10 MP: [MPVAE_frame_label_walk_1frame_rollout](exp_GAMMAPrimitive/cfg/MPVAE_frame_label_walk_1frame_rollout.yml), [MPVAE_frame_label_walk_2frame_rollout](exp_GAMMAPrimitive/cfg/MPVAE_frame_label_walk_2frame_rollout.yml)
* Sit:
  * 1 MP: [sit_lie_1frame](exp_GAMMAPrimitive/cfg/sit_lie_1frame.yml), [sit_lie_2frame](exp_GAMMAPrimitive/cfg/sit_lie_2frame.yml) 
  * 10 MP: [sit_lie_1frame_rollout](exp_GAMMAPrimitive/cfg/sit_lie_1frame_rollout.yml), [sit_lie_2frame_rollout](exp_GAMMAPrimitive/cfg/sit_lie_2frame_rollout.yml) 
* Lie:
  * 1 MP: [lie_1frame](exp_GAMMAPrimitive/cfg/lie_1frame.yml), [lie_2frame](exp_GAMMAPrimitive/cfg/lie_2frame.yml) 
  * 10 MP: [lie_1frame_rollout](exp_GAMMAPrimitive/cfg/lie_1frame_rollout.yml), [lie_2frame_rollout](exp_GAMMAPrimitive/cfg/lie_2frame_rollout.yml)

When typing the names in the command line, please do not include `.yml`.

## Locomotion Control Policy Training
* Train with goal reaching termination:
  ```
  python exp_GAMMAPrimitive/train_GAMMAPolicy_collision.py --config-name MPVAEPolicy_frame_label_walk_collision wandb.name=map_walk lossconfig.weight_pene=1 lossconfig.weight_target_dist=1 lossconfig.weight_nonstatic=0 lossconfig.kld_weight=10 args.gpu_index=0
  ```
* Train without goal reaching termination:
  ```
  python exp_GAMMAPrimitive/train_GAMMAPolicy_collision.py --config-name MPVAEPolicy_frame_label_walk_collision wandb.name=map_nostop trainconfig.use_early_stop=false lossconfig.weight_pene=1 lossconfig.weight_target_dist=1 lossconfig.weight_nonstatic=0 lossconfig.kld_weight=10 args.gpu_index=0
  ```

## Interaction Control Policy Training
* Sit with 1 frame initial motion seed:
  ```
  python exp_GAMMAPrimitive/train_GAMMAPolicy_marker.py wandb.name=sit_policy_1frame modelconfig.body_repr=ssm2_67_condi_marker_inter lossconfig.weight_vp=0 lossconfig.weight_pene=1 lossconfig.weight_contact_feet=2 lossconfig.weight_interaction=0 lossconfig.weight_target_dist=1 lossconfig.kld_weight=10 trainconfig.max_depth=15 args.gpu_index=0
  ```
* Sit with 2 frames initial motion seed:
  ```
  python exp_GAMMAPrimitive/train_GAMMAPolicy_marker_2frame.py wandb.name=sit_policy_2frame modelconfig.body_repr=ssm2_67_condi_marker_inter lossconfig.weight_vp=0 lossconfig.weight_pene=1 lossconfig.weight_contact_feet=2 lossconfig.weight_interaction=0 lossconfig.weight_target_dist=1 lossconfig.kld_weight=10 trainconfig.max_depth=15 args.gpu_index=0
  ```
* Lie with 1 frame initial motion seed:
  ```
  python exp_GAMMAPrimitive/train_GAMMAPolicy_marker.py --config-name MPVAEPolicy_lie_marker wandb.name=lie_policy_1frame modelconfig.body_repr=ssm2_67_condi_marker_inter lossconfig.weight_vp=0 lossconfig.weight_pene=0.2 lossconfig.weight_contact_feet=0 lossconfig.weight_contact_feet_new=0 lossconfig.weight_interaction=0 lossconfig.weight_target_dist=1 lossconfig.kld_weight=10 trainconfig.max_depth=15 args.gpu_index=0
  ```
* Lie with 2 frames initial motion seed:
  ```
  python exp_GAMMAPrimitive/train_GAMMAPolicy_marker_2frame.py --config-name MPVAEPolicy_lie_marker wandb.name=lie_policy_2frame modelconfig.body_repr=ssm2_67_condi_marker_inter lossconfig.weight_vp=0 lossconfig.weight_pene=0.2 lossconfig.weight_contact_feet=0 lossconfig.weight_contact_feet_new=0 lossconfig.weight_interaction=0 lossconfig.weight_target_dist=1 lossconfig.kld_weight=10 trainconfig.max_depth=15 args.gpu_index=0
  ```

# Evaluation
* Locomotion

|                                     | time ↓   | avg. dist ↓ | contact ↑ | loco pene ↑ |
|-------------------------------------|----------|-------------|-----------|-------------|
| [SAMP](https://samp.is.tue.mpg.de/) | 5.97     | 0.14        | 0.84      | 0.94        |
| Ours w/o map, w/o search            | 2.62     | **0.03**    | 0.98      | 0.96        |
| Ours w/o map, with search           | **2.30** | 0.04        | 0.97      | 0.96        |
| Ours w map, w/o search              | 3.08     | 0.04        | 0.98      | **0.97**    |
| Ours w map, with search             | 2.64     | **0.03**    | **0.99**  | **0.97**    |

**Note that the reported locomotion evaluation table here has slightly different structure from the initial paper.** 
[GAMMA](https://yz-cnsdqz.github.io/eigenmotion/GAMMA/) is conceptually the ablation of our method without the walkability map and collision avoidance reward. However, the original released GAMMA is not capable of generating collision-free locomotion in complex indoor environments.
( The original GAMMA is designed and trained for locomotion between far goals in spacious environments and can not handle short-range goals that are common in complex indoor room navigation.) 
Therefore, we retrained the motion primitive model and trained the locomotion policy with closer-range goals suitable for complex indoor scenes to improve its navigation performance in complex environments.
We denote this ablative baseline as `Ours w/o map` in the table above (while it was denoted as `GAMMA` in the original paper). 
We also include the evaluation of locomotion results with and without using the tree-search based test-time optimization.

* Interaction

|                                         | time ↓   | contact ↑ | pene. mean ↓ | pene. max ↓ |
|-----------------------------------------|----------|-----------|--------------|-------------|
| [SAMP](https://samp.is.tue.mpg.de/) sit | 8.63     | 0.89      | 11.91        | 45.22       |
| Ours sit                                | **4.09** | **0.97**  | **1.91**     | **10.61**   |

|                                         | time ↓   | contact ↑ | pene. mean ↓ | pene. max ↓ |
|-----------------------------------------|----------|-----------|--------------|-------------|
| [SAMP](https://samp.is.tue.mpg.de/) lie | 12.55    | 0.73      | 44.77        | 238.81      |
| Ours lie                                | **4.20** | **0.78**  | **9.90**     | **44.61**   |