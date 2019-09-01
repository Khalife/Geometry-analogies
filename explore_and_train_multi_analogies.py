# coding: utf8
import io
import numpy as np
import pdb
import sys
import load_embeddings

# command : 
# python3 explore_and_train_analogies.py words.txt word2vec embeddings.txt 
dataset = open("./data/questions-words.txt", "r")

##########################################################################################
##########################################################################################
# Word2Vec (2013 -)
embeddings_type = sys.argv[2]

try:
    filename = sys.argv[3]
except:
    if embeddings_type == "word2vec":
        filename = "~/Documents/data/word2vec/GoogleNews-vectors-negative300.bin" 
    if embeddings_type == "fastText":
        filename = "~/Documents/data/fastText"
    if embeddings_type == "glove":
        filename = "~/Documents/data/glove.840B.300d.txt" 


if embeddings_type == "word2vec":
    import gensim
    if ".txt" in filename:
        binaryB = False
    else:
        binaryB = True
    try:
        model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=binaryB)
    except:
        model = load_embeddings.load("fastText", filename)


if embeddings_type == "glove":
# Glove (2014 - )
    # LONG TO LOAD
    def loadGloveModel(gloveFile):
        not_worked = 0
        #print("Loading Glove Model")
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            if len(splitLine) > 2:
                word = splitLine[0]
                try:
                    embedding = np.array([float(val) for val in splitLine[1:]])
                except:
                    not_worked += 1
                    continue
                model[word] = embedding

        if not_worked > 0:
            print(not_worked)
        return model

    #model = loadGloveModel("/home/khalife/ai-lab/data/glove.840B.300d.txt")
    model = loadGloveModel(filename)

# fastText (2015-)
if embeddings_type == "fastText":
    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array([float(tok) for tok in tokens[1:]])
        return data

    model = load_vectors(filename)

##########################################################################################
##########################################################################################
#print("Vectors loaded")
from random import shuffle
from scipy.spatial.distance import cosine 
from sklearn.cluster import KMeans

# 1 - Normaliser
# 2 - Récupérer les différences de paires -> 10^6 vecteurs
# 3 - Entraîner pour la classification d'analogie 

fd = open(sys.argv[1], 'r')
words = set()
for line in fd.readlines():
    words.add(line.strip())
fd.close()
words = sorted([w.lower() for w in words])

shuffle(words)
total_words = words

##########################################################################################
##########################################################################################
#print("Get training data")
pair_differences = []

analogies = []
line_splits = set([])
analogy_to_label = {}
current_label_index = 0
for line in dataset:
    #print(line)
    if ":" in line[:1]:
        current_label = line[2:].strip()
        if current_label == "capital-common-countries": # capital-common-countries is included in capital-world  
            continue
        current_label_index += 1
    else:
        if current_label == "capital-common-countries": # capital-common-countries is included in capital-world  
            continue
        line_split = line.split()
        test_list = sorted(line_split)
        analogy_to_label[" ".join(test_list)] = current_label
        if " ".join(test_list) in line_splits:
            continue
        line_splits.update([" ".join(test_list)])
        try:
            vector1 = model[line_split[0]]
            vector2 = model[line_split[1]]
            vector3 = model[line_split[2]]
            vector4 = model[line_split[3]]
            
            analogies.append((line_split,[vector1, vector2], [vector3, vector4], current_label_index))
        except:
            continue

shuffle(analogies)
#analogies = analogies[:1000]
#print("Shuffled")

# 1 Normalize all analogies
#for i in range(len(analogies)):
#    for j in range(1,3):
#        for k in range(2):
#            analogies[i][j][k] = analogies[i][j][k]/np.linalg.norm(analogies[i][j][k])     

# 2
pair_differences = []
pair_cosines = []
visited_pairs = set([])
quad_positive = []
Y = []
for i in range(len(analogies)):
    #print(i)
    pair_differences_i = []
    pair_cosines_i = []
    quad_positive.append((analogies[i][1][0], analogies[i][1][1], analogies[i][2][0], analogies[i][2][1]))
    Y.append(analogies[i][3]) # multi class 
    #Y.append(1) # for binary classification

# Negative pairs
n_neg = len(quad_positive)
nb_words = len(total_words)
#for i in range(n_neg):
index_negative = 0
quad_negative = []
while index_negative < n_neg:
    random_quad = np.random.randint(0, nb_words, 4).tolist()
    try:
        vector1 = model[total_words[random_quad[0]]]
        vector2 = model[total_words[random_quad[1]]]
        vector3 = model[total_words[random_quad[2]]]
        vector4 = model[total_words[random_quad[3]]]
        #norm1 = np.linalg.norm(vector1)
        #norm2 = np.linalg.norm(vector2)
        #norm3 = np.linalg.norm(vector3)
        #norm4 = np.linalg.norm(vector4)
        #quad_negative.append((vector1*(1/norm1), vector2*(1/norm2), vector3*(1/norm3), vector4*(1/norm4)))
        quad_negative.append((vector1, vector2, vector3, vector4))
        index_negative += 1
        Y.append(0)
    except:
        continue
    
X = quad_positive + quad_negative

#pdb.set_trace()
#X = [np.hstack(qpn) for qpn in X] # no shuffle
# Shuffle order of vectors
XX = []
for x in X:
    perm = np.random.permutation(4).tolist()
    # Unauthorized permutations 
    condition1 = (perm[0] == 1) and (perm[1] == 0) and (perm[2] == 3) and (perm[3] == 2)   
    condition2 = (perm[0] == 2) and (perm[1] == 3) and (perm[2] == 0) and (perm[3] == 1)
    condition3 = (perm[0] == 3) and (perm[1] == 2) and (perm[2] == 1) and (perm[3] == 0)

    while not (condition1 or condition2 or condition3):
        perm = np.random.permutation(4).tolist()
        condition1 = (perm[0] == 1) and (perm[1] == 0) and (perm[2] == 3) and (perm[3] == 2)
        condition2 = (perm[0] == 2) and (perm[1] == 3) and (perm[2] == 0) and (perm[3] == 1)
        condition3 = (perm[0] == 3) and (perm[1] == 2) and (perm[2] == 1) and (perm[3] == 0)

    xx = np.hstack([x[perm[0]], x[perm[1]], x[perm[2]], x[perm[3]]])
    XX.append(xx)

X = XX
X = list(enumerate(X))
shuffle(X)
try:
    shuffle_indices, X = zip(*X)
except:
    pdb.set_trace()
Y = [Y[si] for si in shuffle_indices]


##print("... Supplementary test")
#Y_s_exp = []
#X_s_exp = []
##for qp in quad_positive:
##perms_stest = [[1,0,2,3], [0,1,3,2]]
##perms_stest = [[1,0,3,2], [2,3,0,1], [3,2,1,0]]
##perms_stest = [[2,3, 0,1], [1,0, 3,2]]
#perms_stest = [[1,0, 2,3], [0,1, 3,2]]
#for i in range(len(quad_positive)):
#    if i % 2:
#        Y_s_exp.append(1)
#        X_s_exp.append(np.hstack(quad_positive[i]))
#    else:
#        Y_s_exp.append(0)
#        index_perm = np.random.randint(2)
#        perm_i = perms_stest[index_perm]
#        #perm_i = perm = np.random.permutation(4).tolist() 
#        X_s_exp.append(np.hstack([quad_positive[i][perm_i[0]], quad_positive[i][perm_i[1]], quad_positive[i][perm_i[2]], quad_positive[i][perm_i[3]]]))
#        #X_s_exp.append(np.hstack(quad_positive[i]))
        

##########################################################################################
##########################################################################################

import scipy.spatial.distance as sd
#print("Generating other quadruplets")
def generate_quadruplet(n_test=100):
    test_words_indexes = np.random.randint(0, nb_words, n_test).tolist()
    test_words = [total_words[twi] for twi in test_words_indexes]
    pair_differences = []
    for i in range(n_test):
        #print(i)
        for j in range(n_test):
            if i != j:
                try:
                    pair_norm = np.linalg.norm(model[test_words[i]] - model[test_words[j]])
                    pair_difference = (model[test_words[j]] - model[test_words[i]])/pair_norm
                    #pair_difference = model[test_words[j]] - model[test_words[i]]
                    pair_differences.append((test_words[i], test_words[j], model[test_words[i]], model[test_words[j]], pair_difference))
                except:
                    continue
    return pair_differences

def get_closest_pairs(pair_differences):
    nd = len(pair_differences)
    W  = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(nd):
            if i > j:
                #W[i, j] = np.linalg.norm(pair_differences[i][4] -pair_differences[j][4])
                W[i, j ] = sd.cosine(pair_differences[i][4], pair_differences[j][4])

    W = 0.5*(W + W.T)
    for i in range(W.shape[0]):
        W[i, i] = 100
    
    pairs_to_classify = []
    for line_i in range(W.shape[0]):
        line = W[line_i]
        argmin_i = np.argmin(line)
        #argmax_i = np.argmax(line) 
        pair1 = pair_differences[line_i]
        pair2 = pair_differences[argmin_i]
        pair_to_classify = (pair1[0], pair1[1], pair2[0], pair2[1], pair1[2], pair1[3], pair2[2], pair2[3], W[line_i, argmin_i]) ##### WARNING : VECTOR ORDER, SAME AS TRAIN
        pairs_to_classify.append(pair_to_classify)
    
    return pairs_to_classify 

def quadToPairDiff(x):
    u1 = [x[2] - x[0], x[3] - x[1]]
    u1 = np.array(u1)
   
    u2 = [x[6] - x[4], x[7] - x[5]]
    u2 = np.array(u2)

    return u2 - u1
         
def quadToNormalizedPairDiff(x):
    dimension = int(len(x)/4)
    assert(len(x)/4 == dimension)
    #u1 = [x[2] - x[0], x[3] - x[1]]
    #u1_norm = np.linalg.norm(u1)
    #u1 = np.array(u1)*(1/u1_norm)
   
    #u2 = [x[6] - x[4], x[7] - x[5]]
    #u2_norm = np.linalg.norm(u2)
    #u2 = np.array(u2)*(1/u2_norm)
    u2 = x[dimension*3:dimension*4] - x[dimension*2:dimension*3]
    #if np.linalg.norm(u2) != 0:    
    #    u2 = u2/np.linalg.norm(u2)

    u1 = x[dimension:dimension*2] - x[:dimension]
    #if np.linalg.norm(u1) != 0:
    #    u1 = u1/np.linalg.norm(u1)

    return u2 - u1


def quadToPairs(x):
    dimension = int(len(x)/4)
    assert(len(x)/4 == dimension)
    u2 = x[dimension*3:dimension*4] - x[dimension*2:dimension*3]
    u1 = x[dimension:dimension*2] - x[:dimension]
    return u1, u2

#print("Getting minimum components...")
counter = 0
min_vector_corpus = np.zeros(model["ok"].shape)
#for word in model.vocab:
for word in words:
    counter += 1
    #if counter % 1000 == 0:
        #print(counter)
    try:
        for i in range(300):
            if min_vector_corpus[i] > model[word][i]:
                min_vector_corpus[i] = model[word][i]
    except:
        continue



#########################################################################################################
#################################### Classification from scratch ########################################
if 1:
    d = int(X[0].shape[0]/4)
    tp = {}
    fp = {}
    tn = {}
    fn = {}
    
    X_scratch = [X[i] for i in range(len(X)) if Y[i] > 0]
    Y_scratch = [1 for i in range(len(X_scratch))]
    
    n_neg = len(Y_scratch)
    index_negative = 0
    quad_negative = []
    while index_negative < n_neg:
        #print("index_negative " + str(index_negative)) 
        random_quad = np.random.randint(0, nb_words, 4).tolist()
        try:
            vector1 = model[total_words[random_quad[0]]]
            vector2 = model[total_words[random_quad[1]]]
            vector3 = model[total_words[random_quad[2]]]
            vector4 = model[total_words[random_quad[3]]]
            #quad_negative.append((vector1, vector2, vector3, vector4))
            X_scratch.append(np.hstack((vector1, vector2, vector3, vector4)))
            index_negative += 1
            Y_scratch.append(0)
        except:
            continue
    
    X_sl = list(enumerate(X_scratch))
    shuffle(X_sl)
    indices, X_scratch = zip(*X_sl)
    Y_scratch = [Y_scratch[i] for i in indices]
    
    n_scratch = len(X_scratch)
    
    #taus = np.linspace(0.01, 0.5, 10).tolist() 
    taus = [0.1, 0.2]
    for tau in taus:
        tp[tau] = 0
        fp[tau] = 0
        tn[tau] = 0
        fn[tau] = 0
    
    for i in range(n_scratch):
        #print(i)
        norm_i1 = np.linalg.norm(X_scratch[i][d:2*d] - X_scratch[i][:d])
        norm_i2 = np.linalg.norm(X_scratch[i][3*d:] - X_scratch[i][2*d:3*d])
        if norm_i1 == 0:
            norm_i1 = 1
        if norm_i2 == 0:
            norm_i2 = 1
    
        #vector_i = (1/norm_i1)*(X_scratch[i][d:2*d] - X_scratch[i][:d]) - (1/norm_i2)*(X_scratch[i][3*d:] - X_scratch[i][2*d:3*d])  
        vector_i = X_scratch[i][d:2*d] - X_scratch[i][:d] - (X_scratch[i][3*d:] - X_scratch[i][2*d:3*d])
        norm_i = np.linalg.norm(vector_i)
        upper_bound = min([np.linalg.norm(X_scratch[i][d:2*d] - X_scratch[i][:d]), np.linalg.norm(X_scratch[i][3*d:] - X_scratch[i][2*d:3*d])])
        for tau in taus:
            #if norm_i < tau:
            if norm_i < tau*upper_bound:
                if Y_scratch[i] == 1:
                    tp[tau] += 1
                else:
                    fp[tau] += 1
            else:
                if Y_scratch[i] == 0:
                    tn[tau] += 1
                else:
                    fn[tau] += 1
    
    
    RE = {}
    PR = {}
    F1 = {}
    for tau in taus:
        if tp[tau] == 0:
            RE[tau] = 0.
            PR[tau] = 0.
            F1[tau] = 0.
        else:
            PR[tau] = float(tp[tau])/(tp[tau]+fp[tau])
            RE[tau] = float(tp[tau])/(fn[tau]+tp[tau])
            F1[tau] = 2*PR[tau]*RE[tau]/(PR[tau]+RE[tau])
    
    #solution = sorted([[tau, f1] for tau, f1 in F1.items()], key=lambda x: x[1], reverse=True)

    print("F1 test")
    print(F1.values())
    
    pdb.set_trace()
    sys.exit() 
    assert(0)
#########################################################################################################
#########################################################################################################


#########################################################################################################
########################### Classification with supervised learning #####################################
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
train = True
if train:
    #print("Training...")
    
    from sklearn import datasets, linear_model, svm
    
    percentage_train = 0.5
    n_train = int(percentage_train*len(X))
    #print("Logistic regression")
    
    if 1:

        X_pair_train = []
        X_pair_test = []
        for x in X[:n_train]:
            pairs = quadToPairs(x)
            #X_pair_train.append(pairs[0])
            #X_pair_train.append(pairs[1])
            X_pair_train.append(np.hstack([pairs[0] , pairs[1]]))
        for x in X[n_train:]:
            pairs = quadToPairs(x)
            #X_pair_test.append(pairs[0])
            #X_pair_test.append(pairs[1])
            X_pair_test.append(np.hstack([pairs[0], pairs[1]]))

        n_train = len(X_pair_train)

        Y_ = []
        for y in Y:
            Y_.append(y)
            #Y_.append(y)
        Y = Y_

        print("starting classification")
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        neigh = KNeighborsClassifier(n_neighbors=5)
        #clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_pair_train, Y[:n_train])
        #score = clf.score(X_pair_test, Y[n_train:])
        neigh.fit(X_pair_train, Y[:n_train]) 
        score = neigh.score(X_pair_test, Y[n_train:]) 
        #neigh.fit(X[:n_train], Y[:n_train])
        #score = neigh.score(X[n_train:], Y[n_train:])
        print(score)
        pdb.set_trace()
