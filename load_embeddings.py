import gensim
import pdb
import numpy as np
import io

def load(embeddings_type, filename):
    model = {}
    if embeddings_type == "word2vec":
        import gensim
        if ".txt" in filename:
            binaryB = False
        else:
            binaryB = True
        model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=binaryB)
    
    
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
        
    return model



def write_word2vec(model, filename):
    vocabulary = [vo for vo in model.vocab.items()]
    file1 = open(filename, "w")
    file1.write(str(len(vocabulary)) + " " + str(model[vocabulary[0][0]].shape[0]) + "\n")
    for voc in vocabulary:
        vector = model[voc[0]].tolist()
        vector = " ".join([str(ve) for ve in vector])
        line = voc[0] + " " + vector
        file1.write(line + "\n")

    file1.close()
    print("Written model")


def write_model(model, filename):
    file1 = open(filename, "w")
    for voc in model.keys():
        vector = model[voc].tolist()
        vector = " ".join([str(ve) for ve in vector])
        line = voc + " " + vector
        file1.write(line + "\n")
                                                      
    file1.close()
    print("Written model")




