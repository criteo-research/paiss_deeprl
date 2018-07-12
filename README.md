# PAISS Deep Reinforcement Learning Practical Session by Criteo Research

The repo contains all the material needed for a 1,5 hour practical session on Deep Reinforcement Learning.

The idea is to practice basic techniques so as to solve toy problems and also introduce useful techniques and diagnostic tools that allow to ultimately solve harder problems.

Authors that developed this educational material are cited in the AUTHORS file.

## Setup Instructions

1. Install Miniconda (cross-platform Python distribution):
    - download & run installer from [here](https://conda.io/miniconda.html)
1. Create environment for the session:
    ```
    $ conda create --name paiss_deeprl python=3.6 Keras==2.2.0 tensorflow matplotlib
    ```
1. Activate environment:
    ```
    $ source activate paiss_deeprl
    ```
1. Install jupyter
    ```
    $ pip install jupyter
    ```
1. Gym RL toolkit dependencies
    - Linux (known to work on Ubuntu 16.04)
        ```
        $ sudo apt-get install cmake swig zlib1g-dev
        ```
    - Mac OSX
        ```
        $ brew install cmake swig
        ```
    - Windows
        - we recommend creating a Linux virtual machine
        - you could try [this answer on StackOverflom](https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows) at your own risk

1. Gym RL toolkit install
    ```
    $ git clone https://github.com/openai/gym.git
    $ cd gym
    $ pip install -e .[all]
    ```
1. Verify everything is working
    ```
    $ cd - && python3 -c "import keras, gym"
    ```
1. Pull the notebook with exercises
    ```
    $ git clone https://github.com/criteo-research/paiss_deeprl.git    
    ```


## Exercises (FOR STUDENTS)

To run the notebook & start experimenting:
```
$ jupyter notebook exercises.ipynb
```

Note that there is also a PyTorch version : see `exercises_pytorch.ipynb`.
If you wish to use it you'll need to `conda install pytorch torchvision -c pytorch`. 


## Slides (FOR PRESENTER)
```
$ jupyter nbconvert slides.ipynb --to slides --post serve
```
