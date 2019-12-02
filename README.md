# GFCamera

## Install

### Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1+
- opencv3, protobuf, python3-tk
- slidingwindow
  - https://github.com/adamrehn/slidingwindow
  - I copied from the above git repo to modify few things.

### Pre-Install Jetson case

```bash
$ sudo apt-get install libllvm-7-ocaml-dev libllvm7 llvm-7 llvm-7-dev llvm-7-doc llvm-7-examples llvm-7-runtime
$ export LLVM_CONFIG=/usr/bin/llvm-config-7 
```

### Install

Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://github.com/Jrebort/GF_camera
$ cd GF_camera
$ pip3 install -r requirements.txt
```

Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

### Package Install

Alternatively, you can install this repo as a shared package using pip.

```bash
$ git clone https://github.com/Jrebort/GF_camera
$ cd GF_camera
$ python setup.py install  # Or, `pip install -e .`
```
### pre-process images of pose
We provide a dataset of pose which is pre-process by our estimator.
You can get it at: https://pan.baidu.com/s/1A3nLiz8shgcQWxvSQxptaw
Then unzip it in path: './pose_images/'

## Inference
### Realtime Process
```
$ python run.py --model=mobilenet_thin --resize=432x368 --camera=0
```
Apply TensoRT 

```
$ python run.py --model=mobilenet_thin --resize=432x368 --camera=0 --tensorrt=True
```


## KeyPoints schematic diagram
![KeyPoints](/KeyPointsDisplay.png)
