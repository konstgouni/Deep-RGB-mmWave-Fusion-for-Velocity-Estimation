# Deep-RGB-mmWave-Fusion-for-Velocity-Estimation
This code is intended for replicating the results that correspond to a manuscript to be submitted for publication. The inference script evaluates the deep fusion model over consequtive experiments, fusing two consecutive red-green-blue (RGB) image  frames with mmWave reference signal received power measurements and estimates vehicle-UE velocity.

To run the experiments, please:

1) Download Deep Sense 6G Scenario 9 dataset provided and licensed by Wireless Intelligence Lab @ Arizona State University:  https://www.deepsense6g.net/scenarios/Scenarios%201-9/scenario-9

2) Place the downloaded and unzipped folder 'scenario9_dev' at the directory of your preference. Please bear in mind that this will be the parent directory where the driving experiment cycles' annotation file 'annotatedCycles.txt' as well as the related deep learning models: the novel deep fusion model 'Adam3_FineTunedRGB_mmWave_Velocity_ModelF_savedmodel' (tf / keras folder) and the depth retrieval model for the optical flow baseline './mmWave_Polar_Localizer.keras' should be. This is the directory where the inference file 'inference.py' should live.
' 
3) Create a new python (conda) environment with the following creation / activation commands and the packages / dependecies:

conda create -n deep_fusion -c conda-forge ^
  python=3.10.19 ^
  cudatoolkit=11.2.2 ^
  cudnn=8.1.0.77 ^
  cmake=4.1.2

conda activate deep_fusion

python -m pip install --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1

python -m pip install --index-url https://download.pytorch.org/whl/cu113 ^
  torch==1.12.1+cu113 ^
  torchvision==0.13.1+cu113 ^
  torchaudio==0.12.1+cu113

python -m pip install requirements.txt

4) 
   
