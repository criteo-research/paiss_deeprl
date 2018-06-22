# PAISS Deep Reinforcement Learning Practical Session by Criteo Research

## Environment

1. Install Miniconda (cross-platform Python distribution): download & run installer from https://conda.io/miniconda.html
1. Create environment for the session:
    ```
    $ conda create --name paiss_deeprl python=3.6 Keras==2.2.0 tensorflow=1.7.0 matplotlib=2.2.2
    ```
1. Activate environment:
    ```
    $ source activate paiss_deeprl
    ```
1. Install jupyter
    ```
    $ pip install jupyter
    ```
1. Gym RL toolkit install
    ```
    $ sudo apt-get install cmake swig zlib1g-dev
    $ git clone https://github.com/openai/gym.git
    $ cd gym
    $ pip install -e .[all]
    ```
1. Verify everything is working
    ```
    $ python3 -c "import keras, gym"
    ```
