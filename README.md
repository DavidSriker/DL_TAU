# Deep Drone Racing - TAU
![teamwork-cat-dog](GIFs/sim.gif)
![teamwork-flip](GIFs/real.gif)

This repo contains tensorflow 2 implementation of a zero-shot sim2real method for deep drone racing.  
In addition, tensorflow lite model conversion is also featured.
The available code scripts are used to train and evaluate various *ResNet* based models.
The manuscript accompanying this work also attached, and a small portion of the data for fast activation and sanity check.


#### credits:
This work was performed on the basis of the UZH, Robotics and Perception Group.  

The relevant published paper is the: [Deep Drone Racing: From Simulation to Reality with Domain Randomization](http://rpg.ifi.uzh.ch/docs/TRO19_Loquercio.pdf).

For more information visit their project page. The repo can be found at https://github.com/uzh-rpg/sim2real_drone_racing

## Installation

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Requirements

This code was tested with various operating systems with python 3.6.9:
* Ubuntu 18.04
* Windows 10, 64 bit
* MacOS Catalina

*The code also runs on Jetson TX2, for which all dependencies need to be installed via NVIDIA JetPack SDK.*


### Step-by-Step Procedure
In order to set the virtual environment, apriori installation of Anaconda3 platform is required.

Use the following commands to create a new working virtual environment with all the required dependencies.

**CPU based enviroment**:
```
git clone https://github.com/DavidSriker/Deep_Learning_TAU.git
cd Deep_Learning_TAU
conda env create -f tf2_cpu.yml
conda activate tf2_cpu
pip install -r pip_requirements_cpu.txt
```

**GPU based enviroment**:
```
git clone https://github.com/DavidSriker/Deep_Learning_TAU.git
cd Deep_Learning_TAU
conda env create -f tf2_gpu.yml
conda activate tf2_gpu
pip install -r pip_requirements_gpu.txt
```

*In order to utilize the GPU implementation make sure your hardware and operation system are compatible for tensorflow 2 with python 3*

### Data Collection
* Ubuntu 18.4:   
Run the shell script ```DataCollect.sh```
* Windows 10:   
Download the training data from:   
http://rpg.ifi.uzh.ch/datasets/sim2real_ddr/simulation_training_data.zip  
Download the validation data from:  
http://rpg.ifi.uzh.ch/datasets/sim2real_ddr/validation_real_data.zip  
Create "Data" folder and place it in the project's directory.  
Unzip each dataset into the "Data" directory.
* MacOS Catalina:   
Run the shell script ```DataCollect.sh```

*Create a "test_data" folder and place it within the "Data" directory. Place your test data inside.*

## Training the model

Before training the model, possible modification can be done in the `CommonFlags` script under the following comments:
* Train parameters
* Train Optimizer Params
* Network name (modify to choose custom training)
    * `net_name`

In order to initiate trainining, sevral experiment regimes are possible and need to be tuned in the `train` script:
* `Experiment_num = 0`: run *ResNet8* model with *Adam* optimizer and ![formula](https://render.githubusercontent.com/render/math?math=\gamma)=0.1.
* `Experiment_num = 1`: run *ResNet8* model with ![formula](https://render.githubusercontent.com/render/math?math=\gamma)=0.1 for each defined optimizer. Currently defined are *Adam*,*SGD*, *Adadelta* and *Adagrad* optimizers.
* `Experiment_num = 2`: run *ResNet8* model with *Adam* optimizer for multiple ![formula](https://render.githubusercontent.com/render/math?math=\gamma) values. For each ![formula](https://render.githubusercontent.com/render/math?math=\gamma) the model is trained for multiple number of epochs.  
* `Experiment_num = 3`: run *ResNet8*, *ResNet7* and *TC-ResNet8* with *Adam* optimizer ![formula](https://render.githubusercontent.com/render/math?math=\gamma)=0.1.

Execute training by running:  
```
python Train.py
```

## Testing the model

Before testing the model, several option changes can be done in the `CommonFlags` script under the following comments:
* Network name
    * `net_name`

* Testing parameters
    * `tflite`
    * `export_test_data`
    * `num_test_img_save`
    * `test_img_save`
* Another issue is the amount of epochs the model was trained currently there are 3 models that was trained for 300 epochs in the case one retrain the model adjust appropretly in the `CommonFlags`:
  * Train parameters
  * Train Optimizer Params

Execute testing by running:  
```
python Test.py
```

## Model inference

In order to perform inference over the saved model, several options can be first tuned in the `CommonFlags` script under the following comments:
* Network name
    * `net_name`
* Another issue is the amount of epochs the model was trained currently there are 3 models that was trained for 300 epochs in the case one retrain the model adjust appropretly in the `CommonFlags`:
    * Train parameters
    * Train Optimizer Params

In addition, the image size should be set in the *DroneMode.py* script.

Execute inference by running:
```
python DroneMode.py
```

*If the inference part is required over an embedded-gpu hardware for actual UAV applications, then make sure to use communication based ROS2 that support python 3.*

## Jetson TX2

The code works as expected, but in order to activate it on the Jetson TX2 one should install all the right dependencies via NVIDIA-SDK Manager; For more details please refer to the document: **Articles/DL_TAU.pdf** in the repository.

## Tasks:
- [x] TF-2 implementation
- [x] TF-Lite conversion
- [x] Build *ResNet8*, *ResNet7* & *TC-ResNet8*
- [x] Data analysis code
- [x] Clean code
- [x] Inference implementation
- [x] Jetson TX2 implementation

## Authors

* **David Sriker** - *David.Sriker@gmail.com*
* **Lidor Karako** - *Lidorkarako@gmail.com*
