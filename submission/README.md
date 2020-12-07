# Team22 COVID-19 Prediction Project Code Submission

## Team Info

Hamlin Liu
Juan Estrada
Justin Yi
Shriniket Buche
Yash Lala

## Build Instructions

You need to first create the COVID-19 conda environment by the given
`environment.yml` file, which provides the name and necessary packages for this
tasks. If you have `conda` properly installed, you may create, activate or
deactivate the environment via the following commands:

```
conda env create 
conda activate covid19
conda deactivate
```
OR 

```
conda env create --name NAMEOFYOURCHOICE -f environment.yml
conda activate NAMEOFYOURCHOICE
conda deactivate
```

To view the list of your environments, use the following command:
```
conda env list
```

More useful information about managing environments can be found
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

You may also quickly review the usage of basic Python and Numpy package, if
needed in grading our assignment.

---

To generate our predictions for Part 1, run `python round1.py`. The resulting
predictions will be stored in `./round1.csv`.

To generate our predictions for Part 2, run `python round2.py`. The resulting
predictions will be stored in `./round2.csv`.


## Files

- /environment.yml
  - This file describes the conda environment necessary to run our code. 
- /data
  - test.csv
    - the test csv file that was given to us in the Kaggle competition with
      the test dates filled. Is used as a template to generate our round
      1 submission. 
  - test_round2.csv
    - the initial "test" csv file given to us in the Kaggle competition round
      2. Is used as a template to generate our round 2 submission. 
  - train.csv
    - round 1 training data
  - train_full.csv
    - round1 + round 2 training data as well as data scraped from JHU CSSE
      covid repo (https://github.com/CSSEGISandData/COVID-19)
  - round1.csv 
    - our predictions for round 1.
  - round2.csv 
    - our predictions for round 2. Should be identical to `/Team22.csv`.
- /round1.py
  - Code needed to create round 1 predictions
- /round2.py
  - Code needed to create round 2 predictions
- /Team22.csv
  - our round 2 predictions
