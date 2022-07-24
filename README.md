<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a>
    <img src="readme-assets/analytics-3088958-1920.jpg" alt="Logo" width="240" height="240">
  </a>

  <h1 align="center">Data compression using dimensionality reduction</h1>
</div>

Project Organization
------------

    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    ├── README.md               <- The top-level README for developers using this project.
    ├── readme-assets           <- Resources used in README.md.
    ├── data
    │   ├── processed           <- The final, canonical data sets for modeling.
    │   └── raw                 <- The original, immutable data dump.
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures             <- Generated graphics and figures to be used in reporting
    │   └── presentation        <- Presentation for reporting experimental findings
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`
    │
    └── src                     <- Source code for use in this project.
        ├── __init__.py         <- Makes src a Python module
        │
        ├── data                <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── calculations        <- Scripts to calculate statistics
        │   └── calculate.py
        │
        ├── features            <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models              <- Scripts to train models and evaluate models
        │   └── train_and_evaluate_model.py              
        │
        └── visualization       <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
    
--------