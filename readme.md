Data:
1. Create the data folder
2. There are 3 types of datasets each one with differents types of images and its text lables
3. You can combine them mannually using these following cammands 
        mkdir Combined
        cp D1/*  Combined
        cp D2/*  Combined
        cp D3/*  Combined

Setting up virtual environment and Installation of required packages:
    1.virtualenv venv
    2.pip isntalll -r requirements.txt 

There are seperate notebooks for each experiment and notebooks are Available directly to analyze every step.

Also structed the code in
    - run.py
    - preprocessing.py
    - model_help.py
    
Other Resources:  
    - Notebooks
        D1
        D2
        D3
        Combined
    - models
    - requirement.txt

Command to run:
    You can run notebook or following command can also be used.
    python run.py <PATH>
    e.g. python  run.py ./Data/D4
