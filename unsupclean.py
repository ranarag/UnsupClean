#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright Â© 2016 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.

# coding: utf-8


###IMPORTS###############
import cPickle
import operator
import math
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from multiprocessing import Pool
import community
import networkx as nx
import argparse
##########################

#########GLOBAL VARIABLES###########################
#########GLOBAL VARIABLES###########################
parser = argparse.ArgumentParser()
parser.add_argument('-wordvec_file', type=str, help=' word vectors file with word and its corresponding vectors in google word2vec text format')
parser.add_argument('-cooccur_dict', type=str, help='pickle file containing dict with word pairs tuple as key and their co-occurrence counts as value')
parser.add_argument('-alpha', type=float, help='the alpha value [0,1]')
parser.add_argument('-output_fname', type=str, help='name of the output file where clusters will be stored')


##############Optional Arguments###################
parser.add_argument('-nprocs', type=int, default=1, help='number of parallel process for process pool')
parser.add_argument('--stopword_list', type=str, default='', help='file containing stopwords in each line')


args = parser.parse_args()
model = KeyedVectors.load_word2vec_format(args.wordvec_file, binary=False)

prefix = 0
alpha = args.alpha
model_no = args.output_fname
totWords = model.vocab.keys()


with open(args.cooccur_dict, 'rb') as fid:
    wordDoc = cPickle.load(fid)
        
def editSim(str1, str2):
    """
    Calculates the EditSim(ES).

    Description: The EditSim between two strings str1 and str2, 
    is an edit distance based similarity measure calculated by
    1.0 - (ED(str1, str2)/maximum(length(str1), length(str2))).
    
    Parameters:
    str1(str): The first string
    str2(str): The second string

    Returns:
    float: the ES value
    """
    m = len(str1)
    n = len(str2)    
    dp = [[0 for x in range(n+1)] for x in range(2)]
    b = 0 
    for i in range(m+1):
        b = b^1
        for j in range(n+1):           
            if i == 0:
                dp[b][j] = j    
            elif j == 0:
                dp[b][j] = i    
            elif str1[i-1] == str2[j-1]:
                dp[b][j] = dp[b^1][j-1]            
            else:
                dp[b][j] = 1 + min(dp[b][j-1],        # Insert
                                   dp[b^1][j],        # Remove
                                   dp[b^1][j-1])    # Replace
 
    return 1.0 - (dp[b][n] / float(max(n,m)))



def lcsSim(word1, word2):
    """
    Calculates the BLCSR
    Parameters:
    word1(str): The first word
    word2(str): The seconf word

    Returns:
    float: the BLCSR value
    """
    n = len(word1)
    m = len(word2)
    lcs = [[0]*(m+1) for i in xrange(n+1)]
    for i in xrange(2,n+1):
        for j in xrange(2,m+1):
            if(word1[i-2:i] == word2[j-2:j]):
                lcs[i][j] = 1+lcs[i-1][j-1]
            else:
                lcs[i][j] = max(lcs[i][j-1], lcs[i-1][j])

    try:   
        rt = lcs[n][m] / (float(max(n,m)) - 1.0)
    except:
        rt = 0.0
    return rt

def cooccurDist(w1, w2):
    """
    Calculates the cooccur distance 
    between 2 words w1 and w2.
    """
    global wordDoc
    try:
        return wordDoc[(w1, w2)]
        
    except:
        return 0.0



def find_parent(union_dict, word):
    """ 
    find root of in union find algorithm
    with path compression
    """
    while union_dict[word] != word:
        word = union_dict[word]
    
    return word


def union(union_dict, word1, word2):
    """
    union function in union-find algorithm
    """
    parent1 = find_parent(union_dict, word1)
    parent2 = find_parent(union_dict, word2)
    if len(parent1) < len(parent2):
        union_dict[parent1] = parent2
    else:
        union_dict[parent2] = parent1





def find_match(x, w):
    """
    Function to select the candidate morphemes.
    
    Description:
    Function selects candidate morphemes(w) of a particular word(x)which 
    satisfy two conditions:
    1. length of the word is >= 2
    2. BLCSR(x, w) > alpha

    Parameters:
    x(str): word whose candidate morpheme is required
    w(str): word which may be candidate morpheme

    Returns:
    boolean: True if the conditions are satisfied
    False otherwise
    """  
    global prefix 
    global alpha     
    return ((x[:prefix]==w[:prefix]) and \
    (len(w[prefix:]) >= 2 and len(x[prefix:]) >= 2) and \
(float(lcsSim(w[prefix:], x[prefix:])))) > (alpha)






def get_morphemes(qWord):
    """
    Finds the morphemes of a particular word.

    Parameters:
    QWord(str): the query word whose morphemes are needed 
    to be found

    Returns:
    A tuple(qWord, cluster) where qWord is the query word
    and cluster is the cluster of morphemes of qWord.
    """
    global totWords
    cluster_count = 0
    candidates = [qWord]
    
    for word in totWords:

        flag = True
        for w2 in candidates:

            if find_match(w2,word) == False:
                flag = False                
                break
        if flag:
            candidates.append(word)

    candidates.remove(qWord)

    stem_dict = find_beta_and_clus(candidates) 
    
    maxClus = None
    maxVal = 0.0
    for _, words in stem_dict.iteritems():
        for w in words:
            k = editSim(qWord, w)
            if k > maxVal:
                maxVal = k
                maxClus = words
    
    return (qWord, maxClus)

def find_clus(cluster,beta):
    """
    Breaks the singleton cluster into small clusters.
    """    
    global model
    G = nx.Graph()
    totClus = {}
    n = len(cluster)
    # print n
    for i in range(n-1):
        w1 = cluster[i]        
        for j in range(i+1,n):
            w2 = cluster[j]

            cr = cosine_similarity([model[w1]], [model[w2]])[0][0]

            if cr>= beta:
 
                try:
                    cr = cr * cooccurDist(w1,w2)
                    if cr > 0:
                        G.add_edge(w1,w2,weight=cr)
                        G.add_edge(w2,w1, weight=cr)
                except:
                    pass
                
            
                
            
    
    
    part = community.best_partition(G)
    for k, v in part.iteritems():

        try:
            totClus[v].append(k)
        except:
            totClus[v] = [k]
    
    return totClus


def find_beta_and_clus(c_list):
    """
    calculates beta value and breaks the singleton
    cluster based on that beta value
    """
    global model
    n = len(c_list)
    beta_val = 0.0
    lmbdaSum = 0.0
    for i in range(n-1):
        w1 = c_list[i]
        if len(w1) < 3:
            continue
        for j in range(i+1,n):
            w2 = c_list[j]
            if len(w2) < 3:
                continue
            cosSim = cosine_similarity([model[w1]], [model[w2]])[0][0]
            lSim = lcsSim(w1,w2)
            beta_val += (lSim*cosSim)
            lmbdaSum += lSim

    if lmbdaSum == 0:
        return find_clus(c_list, 1.0)
    else:
        return find_clus(c_list, beta_val/lmbdaSum) # beta 
 









    

if __name__ == '__main__':
    p = Pool(args.nprocs)    
    f = open(model_no+'.txt' ,'w')        
    ResClus = []
    topicWords = model.vocab.keys()    
    topicWords = sorted(topicWords, key=len)
        # while len(topicWords):
    union_dict = {}
    for word in topicWords:
        union_dict[word] = word
    
    tResClus = p.map(get_morphemes, topicWords)
    p.close()
    p.join()
    for word, clusters in tResClus:   
        if clusters is None:
            continue
        for clus in clusters:
            union(union_dict, word,clus)
        
    stem_dict = {}
    for word in topicWords:
        p = find_parent(union_dict, word)
        try:
            stem_dict[p].append(word)
        except:
            stem_dict[p] = [word]
    for word, clus in stem_dict.iteritems():        
        ResClus.append((word, clus))
            # print get_morphemes(word)
            # topicWords.remove(word)
            # if ResClus[-1][1] is not None:
            #     # print ResClus[-1][1]
            #     for words in ResClus[-1][1]:
            #         try:
            #             topicWords.remove(words)
            #         except:
            #             pass
                        # print words

            # print word
        # p.close()
        # p.join()
    wordStemDict = {}
    for qWord, clus in ResClus:
        for c in clus:
            wordStemDict[c] = qWord
        f.write(qWord+'<||>')
        if clus is None:
            f.write(qWord+'\n')
            continue
        for w in clus:
            f.write(w+' ')
        f.write('\n')
    f.close()
    with open(model_no+'.pkl', 'wb') as fid:
        cPickle.dump(wordStemDict, fid)
    print "Done"
                



    








    

        
            

