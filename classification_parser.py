import numpy as np
import heapq
#import knn_debug
import pdb
from pyspark import SparkContext, SQLContext
from pyspark.conf import SparkConf
import pyspark
import load_embeddings
#from sklearn.neighbors import KNeighborsClassifier

def subject_parse(filepath):
    f1 = open(filepath, "r")
    X = []
    Y = []
    for line in f1:
        label, sentence = line.strip().split("\t")
        binary_label = 1 if label == "objective" else 0
        Y.append(binary_label)
        X.append(sentence.split())
    f1.close()
    return X, Y


def webkb_parse(filepath):
    f1 = open(filepath, "r")
    map_label = {"course": 0, "faculty": 1, "project": 2, "student": 3}
    X = []
    Y = []
    for line in f1:
        line_split = line.strip().split()
        label = line_split[0]
        sentence = line_split[1:]
        new_label = map_label[label]
        Y.append(new_label)
        X.append(sentence)
    f1.close()
    return X, Y

def amazon_parse(filepath):
    f1 = open(filepath, "r")
    X = []
    Y = []
    for line in f1:
        label, sentence = line.strip().split("\t")
        binary_label = 1 if label == "positive" else 0
        Y.append(binary_label)
        X.append(sentence.split())
    f1.close()
    return X, Y

def reuters_parse(subset="test"):
    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset=subset)
    Y = newsgroups_target.tolist()
    X = newsgroups_test.data
    return X, Y


#def trainAndScore(X_train, Y_train, X_test, Y_test):
#    neigh = KNeighborsClassifier(n_neighbors=5)
#    neigh.fit(X_train, Y_train)
#    score = neigh.score(X_test, Y_test)
#    return score
    

def embedCorpus(model, X_s):
    X = []
    for x in X_s:
        xx = []
        for x_ in x:
            try:
                xx.append(model[x_])
                #model[x_]
                #xx.append(x_)
            except:
                continue
        X.append(xx)   
         
    return X

def relaxWMD(X1,X2):
    n1 = len(X1)
    n2 = len(X2)
    W = np.zeros((n1,n2))
    mins_i = []
    mins_j = []

    for i in range(n1):
        min_dij = 100000000
        for j in range(n2):
            dij = np.linalg.norm(X1[i] - X2[j])
            W[i,j] = dij
        
    if 0 in W.shape:
        return 0.
    else:
        cost_i = np.sum(np.min(W, 1))
        cost_j = np.sum(np.min(W, 0))
        return max([cost_i, cost_j])

def KNN(X_train, X_test, Y_train, Y_test):
    Y_predicted = []
    index_t = 0
    accuracy_p = 0
    for xt in X_test:
        index_t += 1
        print(index_t)
        if index_t % 100 == 0:
            print(index_t)
        di = []
        for i in range(len(X_train)):
            #dxti = relaxWMD(xt, X_train[i])
            dxti = D[vocabulary[xt],vocabulary[X_train[i]]]
            #dxti = word_mover_distance(xt, X_train[i], wvmodel, lpFile=None)
            di.append((i,dxti))
                    
        di_sorted_top = sorted(di, key =lambda x : x[1])
        di_indexes_top = [dist[0] for dist in di_sorted_top]
        closest_labels = [Y_train[diit] for diit in di_indexes_top]
        predicted_label = max(set(closest_labels), key = closest_labels.count) 
        Y_predicted.append(predicted_label)
        if predicted_label == Y_test[index_t]:
            accuracy_p += 1
        if index_t % 10 == 0:
            print("accuracy : " + str(accuracy_p))        

    accuracy_score = len([1 for yp, y in zip(Y_predicted, Y_test) if yp == y])
    accuracy_score = accuracy_score/(float(len(Y_test)))

    return accuracy_score

def relaxWMD1(X1,X2): #,D, vocabulary):
    # X1 and X2 2 documents
    n1 = len(X1)
    n2 = len(X2)
    W = np.zeros((n1,n2))
    mins_i = []
    mins_j = []

    for i in range(n1):
        for j in range(n2):
            #dij = np.linalg.norm(X1[i] - X2[j])
            #dij = D.value[vocabulary.value[X1[i]], vocabulary.value[X2[j]]]
            #if X1[i] != X2[j]:
                #key_ij = " ".join(sorted([X1[i], X2[j]]))
                #dij = D.value.get(key_ij, 0.)
                #W[i,j] = dij
            if 1: 
                W[i,j] = np.linalg.norm(X1[i] - X2[j])

    if 0 in W.shape:
        return 0.
    else:
        cost_i = np.sum(np.min(W, 1))
        cost_j = np.sum(np.min(W, 0))
        return max([cost_i, cost_j])


def knnd_spark(x):#, X_train, Y_train, D, vocabulary, k=5):
    # x is a xtest
    dx = []
    for j in range(len(X_train)):
        dj = relaxWMD1(x, X_train[j])#, D, vocabulary)
        #dj = j
        dx.append((j, dj))

    di_indexes_top = heapq.nsmallest(5, dx, lambda x : x[1])
    closest_labels = [Y_train[diit[0]] for diit in di_indexes_top]
    predicted_label = max(set(closest_labels), key = closest_labels.count)
    return predicted_label

def clean_docs_from_model(documents_list, model):
    clean_list = []
    for awt in documents_list:
        clean_doc = []
        for word in awt:
            try:
                #model[awt]
                #clean_doc.append(awt)
                clean_doc.append(model[word])
            except:
                continue
        clean_list.append(clean_doc)

    return clean_list


def define_parameters(X_train, X_test, model):
    all_words_test = list(set(sum(X_test, [])))
    all_words_train = list(set(sum(X_train, [])))

    all_words_test = clean_list_words_from_model(all_words_test, model)
    all_words_train = clean_list_words_from_model(all_words_train, model)
    
    D = {}
    iw1 = 0
    IW1 = len(all_words_test)
    for w1 in all_words_test:
        iw1 += 1
        if iw1 % 100 == 0:
            print(IW1 - iw1)
        for w2 in all_words_train:
            if w2 > w1:
                key12 = " ".join(sorted([w1, w2]))
                D[key12] = np.linalg.norm(model[w1] - model[w2]) 

    return D



def run_knnd_spark(X_train, X_test):
    print("Starting")
    X_test = sc.parallelize(X_test, 500)
    Y_predicted = X_test.map(lambda x: knnd_spark(x)).collect()
    ac = sum([int(y == ytrue) for y, ytrue in zip(Y_predicted, Y_test)])/float(len(Y_test))
    return ac


if __name__ == "__main__":
    import sys
    sc = SparkContext(appName="text-classification")
    type_emb = sys.argv[1]
    filename = sys.argv[2]    
    filewrite1 = filename.split("/")
    filewrite1 = filewrite1[len(filewrite1)-1]
    
    model = load_embeddings.load(type_emb, filename)
    
    #X_train, Y_train = subject_parse("./subject/data/my_subject_train.txt")
    #X_test, Y_test = subject_parse("./subject/data/my_subject_test.txt")  

    #X_train = clean_docs_from_model(X_train, model)
    #X_test = clean_docs_from_model(X_test, model)
    #print("Starting")
    #
    #ac = run_knnd_spark(X_train, X_test)
    #filewrite = open("accuracy_subj_"" + filewrite1, "w")
    #filewrite.write(filename + "\n")
    #filewrite.write("subjectivity " + str(ac) + "\n")
    
    ############################################################################
    ############################################################################
    
    #X_train, Y_train = webkb_parse("./webkb/data/my_WEBKB_train.txt")
    #X_test, Y_test = webkb_parse("./webkb/data/my_WEBKB_test.txt")
    #
    #X_train = X_train[:1000]
    #Y_train = Y_train[:1000]
    #Y_test = Y_test[:500]
    #X_test = X_test[:500]

    #X_train = clean_docs_from_model(X_train, model)
    #X_test = clean_docs_from_model(X_test, model)
    #
    #ac = run_knnd_spark(X_train, X_test)
    #filewrite = open("accuracy_web_" + filewrite1, "w")
    #filewrite.write("web " + str(ac) + "\n")
    
    #############################################################################
    #############################################################################
    #
    X_train, Y_train = amazon_parse("./amazon/data/my_amazon_train.txt")
    X_test, Y_test = amazon_parse("./amazon/data/my_amazon_test.txt")
    
    X_train = X_train[:1000]
    Y_train = Y_train[:1000]
    Y_test = Y_test[:500]
    X_test = X_test[:500]
    X_train = clean_docs_from_model(X_train, model)
    X_test = clean_docs_from_model(X_test, model)

    ac = run_knnd_spark(X_train, X_test)
    filewrite = open("accuracy_amazon_" + filewrite1, "w")
    filewrite.write("amazon " + str(ac) + "\n")
    
    ############################################################################
    ############################################################################
    
    filewrite.close()
