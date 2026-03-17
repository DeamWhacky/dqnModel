### Setup
We use a Conda environment. To recreate it:

conda env create -f environment.yml
conda activate flappy-dqn


Valid Reward Types = ["sparse", "basic", "survival", "pipe"]
For training the model for 2000 episodes (if your folder lacks dqn_sparse.pth):

# python main.py {reward type}
python main.py sparse 


For watching the AI play 20 random episodes (only if your folder contains dqn_sparse.pth):

# python main,py {reward type} watch
python main.py sparse watch 