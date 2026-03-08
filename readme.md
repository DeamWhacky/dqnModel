### Setup
We use a Conda environment. To recreate it:

conda env create -f environment.yml
conda activate gymenv


For training the model for 2000 episodes (if your folder lacks dqn_model.pth):

python main.py


For watching the AI play 20 random episodes (if your folder contains dqn_model.pth):

python main.py watch