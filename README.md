# UnsupClean

Implementation of the proposed algorithm in the paper **An Unsupervised Normalization Algorithm for Noisy Text: A Case Study for Information Retrieval and Stance Detection** by Anurag Roy, Shalmoli Ghosh, Kripabandhu Ghosh, Saptarshi Ghosh

![UnsupClean](UnsupClean.png "Flow-chart of UnsupClean")

### Dependencies
python version: `python 2.7`

packages: 
- `networkx`
- `gensim`
- `python_louvain`
- `community`
- `scikit_learn`


### Data for demo
1. TODO
  
### Run Demo
`python2 unsupclean.py -w2v_model w2v_risot_iter100.txt -alpha 0.56 -output_fname risot_output_test.txt -nprocs 4 --stopword_list stopwords_list_ben.txt -cooccur_dict cooccurData.pkl`




