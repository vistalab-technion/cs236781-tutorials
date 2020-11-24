# CS236781 Tutorials

This repo contains the code and notebooks shown during course tutorials.

You can also view the tutorial notebooks in your browser using `nbviewer` by clicking the
button below.

<a href="https://nbviewer.jupyter.org/github/vistalab-technion/cs236781-tutorials/tree/master/"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" height="50px"/></a>

## Environment set-up

1. Install the python3 version of [miniconda](https://conda.io/miniconda.html).
   Follow the [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
   for your platform.

2. Use conda to create a virtual environment for the assignment.
   From the assignment's root directory, run

   ```shell
   conda env create -f environment.yml
   ```

   This will install all the necessary packages into a new conda virtual environment named `cs236781`.

3. Activate the new environment by running

   ```shell
   conda activate cs236781-tutorials
   ```

4. Optionally, execute the `run-all.sh` script to run all notebook and test that
   everything is installed correctly.

Notes:
- On Windows, you should also install Microsoft's [Build Tools for Visual
  Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019)
  before installing the conda env.  Make sure "C++ Build Tools" is selected during installation.
- After a new tutorial is added, you should run `conda env update` from the repo
  directory to update your dependencies since each new tutorial might add ones.


