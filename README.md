

# Commons Game MultiAgent environment
Multiagent task in CPR (common pool resources) 

# Install Conda venv:

1. Install Miniconda/Anaconda 
- Installation guide - https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html"

2. Create new virtual environment

	- Open conda command line
	- Create new conda venv using: `conda create --name <venv name> python=3.9`
	- Enter the environment we just created using: `conda activate <venv name>

3. Install Tensorflow GPU
	- Tensorflow: `conda install -c anaconda tensorflow-gpu`
4. Install requirements
	- Tensorboard:  `conda install -c conda-forge tensorboard`
	- gym:  `conda install -c conda-forge gym`
	- cv2:  `conda install -c conda-forge opencv`
	- tqdm:  `conda install -c conda-forge tqdm`
	- scikit-image:	`conda install scikit-image`
	- PIL: `conda install -c anaconda pillow`


# Run code
1. Training
	 Execute "src\train_HCLDDQN.py" using command line or Pycharm.
	 inside of "src\train_HCLDDQN.py" there is variable called model_name, all logs of training will  saved to "logs\\<model_name>"
	


# Analytics - Tensorboard
First of all, make sure that conda command line is enabled on the environment we have created. If not, then use: `conda activate <venv_name>`.
Then set conda path to where the project was saved using `cd`.

To activate Tensorboard:
1. Spesipic run: In conda command line execute: `tensorboard --logdir=logs\HLCDDQN\<run_name>`
2. All runs: In conda command line execute: `tensorboard --logdir=logs\HLCDDQN`
- Tensberboard is supposed to return something like:
	 
	"*2022-01-25 02:33:03.319349: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.5.0 at `http://localhost:6006/` (Press CTRL+C to quit)"*

	The highlighted address should be copied to the browser
