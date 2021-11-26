# SIIM-Project
This is a project for CC3066 course.


## Prerequisites

```
conda create -n venv python=3.8
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

## Detect Covid-19

The selected model to develop an "Analyzer & Covid-19 Predictor" was Yolov5x, instead of our other two models: CNN with CheXNet weights and Vanilla CNN. This is due the good metrics obtained after several trains. This app is under the `app` folder, made with Flask. 

1. When we were training Yolov5 model, it generated a **weights** file under /models/yolov5/runs/train/experimentXXXX folder. You need to check your best experiment based on its metrics.

2. Copy the weights under `app` folder and update its references on Flask code.

3. Install dependecies 

```bash
pip3 install -r app/requirements.txt
```

4. Run Flask app 

```bash
cd app
flask run
```

5. Visit http://localhost:5000/index.html & load your test image.
