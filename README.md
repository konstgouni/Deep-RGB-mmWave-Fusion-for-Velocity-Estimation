# Deep-RGB-mmWave-Fusion-for-Velocity-Estimation
This code is intended for replicating the results that correspond to a manuscript to be submitted for publication. The inference script evaluates the deep fusion model over consequtive experiments, fusing two consecutive red-green-blue (RGB) image  frames with mmWave reference signal received power measurements and estimates vehicle-UE velocity.

To run the experiments, please:

1) Download Deep Sense 6G Scenario 9 dataset provided and licensed by Wireless Intelligence Lab @ Arizona State University:  https://www.deepsense6g.net/scenarios/Scenarios%201-9/scenario-9

2) Place the downloaded and unzipped folder 'scenario9_dev' at the directory of your preference. Please bear in mind that this will be the parent directory where the driving experiments' annotation file 'annotatedCycles.txt' as well as the related deep learning models: the novel deep fusion model 'Adam3_FineTunedRGB_mmWave_Velocity_ModelF_savedmodel' (tf / keras folder) and the depth retrieval model for the optical flow baseline './mmWave_Polar_Localizer.keras' should be. This is the directory where the inference file 'inference.py' should live.
' 
3) Create a new python (conda) environment with the following creation / activation commands and the packages / dependecies:

        conda create -n deep_fusion -c conda-forge python=3.10.19 cudatoolkit=11.2.2 cudnn=8.1.0.77 cmake=4.1.2
        
        conda activate deep_fusion
        
        python -m pip install --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1
        
        python -m pip install --index-url https://download.pytorch.org/whl/cu113 torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113
        
        python -m pip install requirements.txt

4) Run the inference file:
   
        python inference.py

You will be prompted to input the test data cycle index for the inference to begin. Type 2 if you want to replicate the 4 consecutive driving cycle experiments. Upon successful termination of this script, you will already be having: a) the velocity estimation plots (deep fusion vs optical flow baseline vs ground truth) for each driving experiment and b) the results corresponding to the concatenation fo the driving experiments.

6) The reported estimation performance and inference latency correspond to inference using the following hardware:

  A laptop with an Intel Core $i7-12700H$ CPU
  16 gigabytes of RAM
  nVIDIA GeForce RTX $3060$ graphics processing unit (GPU) with 8 gigabytes of dedicated video memory.
  Windows 11 Home with the Python 3 interpreter installed.

The dense optical flow-based model was implemented using the Open Computer Vision Library OpenCV (www.opencv.org) and executed on the CPU. Both the dense optical flow-based model and the proposed deep fusion model were evaluated under identical CPU hardware and runtime conditions to ensure a fair comparison. The inference performance of the deep model was also evaluated exploiting GPU acceleration with TensorFlow and DirectML support, resulting in a mean (per-sample) inference latency of approximately 0.02 s over the test data.

Should there be any issue with running the code and replicating the inference, please contact me in one of the email addresses: kgounis@ece.auth.gr, konstgouni@csd.auth.gr
   
