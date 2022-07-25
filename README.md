<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a>
    <img src="readme-assets/analytics-3088958-1920.jpg" alt="Logo" width="240" height="240">
  </a>

  <h1 align="center">Compression using Dimensionality Reduction</h1>
</div>

<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/himalayasharma/compression-using-dimensionality-reduction?style=social"> <img alt="GitHub forks" src="https://img.shields.io/github/forks/himalayasharma/compression-using-dimensionality-reduction?style=social"> <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/himalayasharma/compression-using-dimensionality-reduction"> <img alt="GitHub issues" src="https://img.shields.io/github/issues-raw/himalayasharma/compression-using-dimensionality-reduction">


Comparitive analysis of dimensionality reduction techniques for compression of real-world sensor data.

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

Prerequisites
------------
Before you begin, ensure you have met the following requirements:
* You have a `Linux/Mac/Windows` machine.
* You have installed a `python` distribution.
* You have installed `pip`.
* You have installed `make`.

Setup
------------
1. Clone the repository.
	```
	git clone https://github.com/himalayasharma/data-compression-using-dimensionality-reduction.git
	```
2. Traverse into project directory.
3. Create virtual environment.
	```make
	make create_environment
	```
4. Activate virtual environment.
5. Download and install all required packages.
	```make
	make requirements
	```
6. Download and process physiological sensor dataset.
	```make
	make data
	```
7. Build new set of features after dimensionality reduction.
	```make
	make build_features
	```
8. Calculate required statistics (compression ratio, space saving etc).
	```make
	make calculate
	```
9. Train and evaluate models.
	```make
	make train_and_evaluate
	```
10. Generate plots.
	```make
	make plot
	```
   
Contributing
------------
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. If you have a suggestion that would make this better, please fork the repo and create a pull request. Don't forget to give the project a star! Thanks again!

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

License
------------
Distributed under the MIT License. See `LICENSE` for more information.

Ackowledgements
------------
* [Rukshan Pramodhita](https://rukshanpramoditha.medium.com/)
* [Inma Mohino-Herranz](https://pubmed.ncbi.nlm.nih.gov/?term=Mohino-Herranz%20I%5BAuthor%5D)
* [Roberto Gil-Pita](https://pubmed.ncbi.nlm.nih.gov/?term=Gil-Pita%20R%5BAuthor%5D)
* [Manuel Rosa-Zurera](https://pubmed.ncbi.nlm.nih.gov/?term=Rosa-Zurera%20M%5BAuthor%5D)
* [Fernando Seoane](https://pubmed.ncbi.nlm.nih.gov/?term=Seoane%20F%5BAuthor%5D)

References
------------
* Mohino-Herranz I, Gil-Pita R, Rosa-Zurera M, Seoane F. Activity Recognition Using Wearable Physiological Measurements: Selection of Features from a Comprehensive Literature Study. Sensors (Basel). 2019 Dec 13;19(24):0. doi: 10.3390/s19245524. PMID: 31847261; PMCID: PMC6960825.
* Compression ratio. (2022, June 2). In Wikipedia. https://en.wikipedia.org/wiki/Compression_ratio
--------