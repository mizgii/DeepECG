# DeepECG

This repository hosts the project for the Biomedical Signal and Image Processing Lab at the University of Milan, focusing on ecg-based biometric recognition. The project is based on the methodology described by (deep ecg article) and utilizes the Smart Health for Assessing the Risk of Events via ECG Database.

## Requirements

To start the program you're gonna need:

* **Python 3.9** -> more info you can find https://www.python.org/downloads/release/python-390/ 
* **pip** -> more info about installation you can find https://pip.pypa.io/en/stable/cli/pip_install/ 
* **git** -> more info you can find https://git-scm.com/download/win
* **virtual environment (venv)** -> more info you can find https://virtualenv.pypa.io/en/stable/installation.html

## Seting up the virtual enviroment 

1. in Command Line go to cloned DeepECG repository. Create virtual environment - after this you shuld have aditional folfer **venv**

```commandline
C:\path\DeepECG> python -m venv C:\path\DeepECG\venv
```

2. activate environment - after this step *(venv)* should appear before path in the line
```commandline
C:\path\DeepECG> C:\path_to_program\DeepECG\venv\Scripts\activate
```

3. install requirements
```commandline
(venv) C:\path\DeepECG> pip install -r requirements.txt
```


## Authors
Dr. Massimo W. Rivolta

Zofia Mizgalewicz

Christian R. Cuenca

To see whats happaning in the [BISP](https://bisp.di.unimi.it/)

For more details on the dataset, visit [SHAREEDB](https://physionet.org/content/shareedb/1.0.0/)

