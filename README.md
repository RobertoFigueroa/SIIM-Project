# SIIM-Project
This is a project for CC3066 course.


### Prerequisites

```
conda create -n venv python=3.6
```

Then

```
conda activate myenv
```

Adding virtual env to jupyter:

```
python -m ipykernel install --user --name=myenv
```

After that, choose venv as your jupyter kernel.

Install some useful conda packages (specially for manage DICOM images):

```
conda install -c conda-forge gdcm -y
```
