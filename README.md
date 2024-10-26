CPU inferencing setup:
1. Install Anaconda or Miniconda
  https://www.anaconda.com/download/success
2. Open Anaconda prompt and run the following commands:
  conda create -n tf-cpu tensorflow python=3.9
  conda activate tf-cpu
  pip install opencv-python
  pip install scikit-learn
  pip install matplotlib
3. Run predictvideo.py

GPU inferencing setup:
1. Install Anaconda or Miniconda
  https://www.anaconda.com/download/success
2. Open Anaconda prompt and run the following commands:
  conda create -n tf-gpu tensorflow-gpu python=3.9
  conda activate tf-gpu
  pip install opencv-python
  pip install scikit-learn
  pip install matplotlib
3. Run predictvideo.py


Follow instructions for GPU TensorFlow at:
https://docs.anaconda.com/working-with-conda/applications/tensorflow/

Within the virtual environment created, run the following commands to install dependencies:

pip install cudatoolkit==11.2.2
pip install cudnn==8.1.0.77
pip install keras==2.10.0
pip install keras-preprocessing==1.1.2
pip install numpy
pip install pandas
pip install opencv-python
pip install scikit-learn
pip install tensorflow==2.10.1


AI/ML Video Prediction:

usage: python predictvideo.py [-h] -i INPUT [-o OUTPUT] [-v VIDEO] [-s FRAMESKIP]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input video(s)
  -o OUTPUT, --output OUTPUT
                        path to output csv(s)
  -v VIDEO, --video VIDEO
                        path to output video(s)
  -s FRAMESKIP, --frameskip FRAMESKIP
                        number of frames to skip in between each measurement


Making charts:

usage: python makeplots.py [-h] input output        

positional arguments:
  input       Input csv file or directory    
  output      Output directory

options:
  -h, --help  show this help message and exit