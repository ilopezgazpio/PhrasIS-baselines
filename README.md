# PhrasIS-baselines
Repository for "PhrasIS: phrase inference and similarity benchmark" paper

PhrasIS is a dataset of Phrase pairs with Inference and Similarity annotations for the evaluation of semantic representations. The dataset is analyzed, showing the relation between inference labels and similarity scores, and is evaluated with several well-known techniques obtaining satisfactory performance.

Requirements
------------
- Python 3
- NumPy
- SciPy
- Pandas
- NLTK
- Seaborn
- heatmapz 

All dependencies in exception of heatmapz are installed when setting the conda environment

To install heatmapz please run  ```pip3 install heatmapz``` 

Usage
-----
Experiments can be run either in Python 3 using the given conda environment or launching the 00.launchColab ipython notebook.


Running the following command will create an environment called **phrasis** with all required dependencies:
```
conda env create -f environment.yml
```

To activate the environment run the following command:
```
conda activate phrasis
```

To deactivate the environment run the following command:
```
conda deactivate
```

To run the ipython notebook file, clone or import the github repo in google colab 
or local installation of jupyter notebook and run the file

Dataset
-------
- PhrasIS dataset can be found in ./dataset

Features
--------
We compute a bunch of lexical and onthology based features, including :
- jaccard overlap
- length differences
- wordnet similarity features: lch, jcn, wup, ...

Models
------
We compute several models, including: 
- Machine Learning models: DecisionTree, LogisticRegression, SVM, ...
- Ensemble methods: Random Forest, ExtraTree, Bagging, ...
- Kernel Ridge

Word Embedding
----------

