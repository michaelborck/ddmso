# Detecting Distant Maritime Surface Objects

This project uses CNNs to detect distant maritime surface objects when observed from panoramic EO sensors near the sea surface

## Project Organization

    ├── LICENSE
    ├── Makefile           <Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── experiments        <- This is where all the tf-models stored
    │
    ├── weights            <- This is where the model weights are stored (last layer)
    │
    ├── docs               <- Documentation specific to usin this project
    │
    ├── src                <- Python source codes, Jupyter notebooks. 
    │   ├── train.py       <- build a model based on parameters in config.py
    │   ├── eval.py        <- Evaluate a model
    │   ├── predict.py     <- Run a model against unseen data and visualise
    │   ├── dataset.py     <- Edit to change or select a subset of images, day, night etc
    │   └── config.py      <- Edit to change experiement parameters
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`
  
--------

Project based on the cookiecutter data science project template
