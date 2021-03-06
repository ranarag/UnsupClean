# UnsupClean
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/)


## Table of Contents

* [Summary](#summary)
* [Dependencies](#dependencies)
* [Hyperparameters and Options](#hyperparameters-and-options)
* [About](#about)
* [Data for Demo](#data-for-demo)
* [Run Demo](#run-demo)

### Summary
Implementation of the proposed algorithm in the paper **An Unsupervised Normalization Algorithm for Noisy Text: A Case Study for Information Retrieval and Stance Detection** by Anurag Roy, Shalmoli Ghosh, Kripabandhu Ghosh, Saptarshi Ghosh. The proposed unsupervised algorithm finds noisy variants of words in a corpus by finding clusters of its morphological variants. The  flow-chart below gives a schematic overview of the UnsupClean algorithm:
![UnsupClean](UnsupClean.png "Flow-chart of UnsupClean")


### About
UnsupClean is a novel Unsupervised and language independent text normalization algorithm. The algorithm relies on lexical and contextual similarities of words/tokens in a corpus to form clusters of their morphological variants. For the contextual similarities the algorithm relies on word vectors and co-occurrence counts. We have used word2vec embeddings for our experiments. However other word vectors can also be used. The implementation expects word vectors and co-occurrence counts in the form of files(more details in [Hyperparameters and Options](#hyperparameters-and-options)). UnsupClean also depends upon `alpha` -- a threshold based on which it determines the lexical similarity. The implementation also has the provision of stopword removal incase anyone wants to exclude stopwords from the clusters. The output will be stored in a file name passed as `output_fname`. The output format is `<word><||><space separated list of words in cluster>`. An example file entry looks like `blessed<||>bless blessed`

### Dependencies
python version: `python 2.7`

packages: 
- `networkx`
- `gensim`
- `python_louvain`
- `community`
- `scikit_learn`

To install the dependencies run `pip install -r requirements.txt`

### Hyperparameters and Options
Hyperparameters and options in `unsupclean.py`.

- `wordvec_file` Word vectors file containing vector representation for each words in the lexicon in google word2vec text format
- `cooccur_dict` Pickle file of a python dictionary which has word tuples as its key and their co-occurance counts as its corresponding value
- `alpha` The alpha value used in the algorithm  \[0, 1\]
- `output_fname` Name of the output file where the word clusters will be stored
- `nprocs` Number of parallel processes(the algorithm can be run parallely)
- `stopword_list` File containing stopwords with one stopword in each line(optional)

### Data for Demo
For demonstration purposes we have provided the wordvectors and co-occurrence counts of the Sem-Eval Atheism dataset. Stop-word removal is not required for this dataset. Following are the data-files:

1. ![w2v_SemEvalAT_new_100dim_iter100.txt](w2v_SemEvalAT_new_100dim_iter100.txt)
2. ![cooccurData.pkl](cooccurData.pkl)
  
### Run Demo
To generate the clusters, run the following command:

`python2 unsupclean.py -wordvec_file w2v_SemEvalAT_new_100dim_iter100.txt -alpha 0.56 -output_fname word_clusters.txt -nprocs 4 -cooccur_dict cooccurData.pkl`

The word clusters will be generated in the `word_clusters.txt` file. 


