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
parser.add_argument('-w2v_model', type=str, help='gensimWord2Vec model file')
parser.add_argument('-cooccur_dict', type=str, help='pickle file containing dict with word pairs tuple as key and their co-occurrence counts as value')
parser.add_argument('-alpha', type=float, help='the alpha value [0,1]')
parser.add_argument('-output_fname', type=str, help='name of the output file where clusters will be stored')


##############Optional Arguments###################
parser.add_argument('-nprocs', type=int, default=1, help='number of parallel process for process pool')
parser.add_argument('--stopword_list', type=str, default='', help='file containing stopwords in each line')


args = parser.parse_args()
model = KeyedVectors.load_word2vec_format(args.w2v_model, binary=False)

prefix = 0
alpha = args.alpha
model_no = args.output_fname
totWords = model.vocab.keys()


with open(args.cooccur_dict, 'rb') as fid:
    wordDoc = cPickle.load(fid)
        
def editSim(str1, str2):
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

def cooccurRatio(w1, w2):
    global wordDoc
    try:
        return wordDoc[(w1, w2)]
        
    except:
        return 0.0



def find_parent(union_dict, word):
    
    while union_dict[word] != word:
        word = union_dict[word]
    
    return word


def union(union_dict, word1, word2):
    
    parent1 = find_parent(union_dict, word1)
    parent2 = find_parent(union_dict, word2)
    if len(parent1) < len(parent2):
        union_dict[parent1] = parent2
    else:
        union_dict[parent2] = parent1





def find_match(longer_word, shorter_word):  
    global prefix 
    global alpha     
    return ((longer_word[:prefix]==shorter_word[:prefix]) and \
    (len(shorter_word[prefix:]) >= 2 and len(longer_word[prefix:]) >= 2) and \
(float(lcsSim(shorter_word[prefix:], longer_word[prefix:])))) > (alpha)

''' 
function to calculate lcs similarity(normalized)
of Two words 
'''




def generate_candidate_stems(qWord):
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
''' Function to calculate the gamma value (threshold)
'''

def find_clus(cluster,beta):    
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
                    cr = cr * cooccurRatio(w1,w2)
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
    
    tResClus = p.map(generate_candidate_stems, topicWords)
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
            # print generate_candidate_stems(word)
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
                



    








    

        
            

