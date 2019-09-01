import pdb
#import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# "pairs-questions-words.txt" has been generated before
pair_dataset = open("pairs-questions-words.txt", "r")

G = nx.DiGraph()
words_a = set([])
current_labels = []
pairs1 = {}
pairs2 = {}
edges = []

relation_to_pairs = {}
homonyms = {}

for line in pair_dataset:
    #print(line)
    if ":" in line[:1]:
        current_label = line[2:].strip()
        if current_label == "capital-common-countries":   # capital-common-countries is included in capital-world  
            continue
        current_labels.append(current_label)
    else:
        if current_label == "capital-common-countries":   # capital-common-countries is included in capital-world  
            continue

        ############################################################################################# 
        ###############  Put potential homonyms as first elements of pairs  ######################### 
        #############################################################################################
        if current_label == "currency":  # only relation concerned
            line_split = line.strip().strip()
            line = line_split[1] + " " + line_split[0]

        line_split = line.strip().split()
        ############################################################################################# 
        ############################################################################################# 

        ############################################################################################# 
        ##############################       Handle homonyms        ################################# 
        ############################################################################################# 
        try: 
            test_list = sum([couple.split() for couple in relation_to_pairs[current_label]], [])
        except:
            test_list = []
            new_line = line

        test2 = True
        if line_split[0] in test_list:
            index_1 = test_list.index(line_split[0])
            if  line_split[1] != test_list[index_1+1]:
                try:
                    homonyms[line_split[0]] += 1
                except:                
                    homonyms[line_split[0]] = 1
                new_line = line_split[0] + "_" + str(homonyms[line_split[0]])
                new_line = new_line + " " + line_split[1]        
                test2 = False

        else:
            new_line = line
        ############################################################################################# 
        ############################################################################################# 
    
        ############################################################################################# 
        ##############################   Get relation to pairs    ################################### 
        ############################################################################################# 

        try:                
            relation_to_pairs[current_label].append(new_line.strip()) 
        except:
            relation_to_pairs[current_label] = [new_line.strip()]

        ############################################################################################# 
        ############################################################################################# 

        ############################################################################################# 
        ##################################  Get nodes and edges   ################################### 
        ############################################################################################# 

        line_split = new_line.split()
        for i in range(2):
            words_a.update([line_split[i]])
            edge = line_split
            direction = new_line.strip()
            try:
                edges.append((edge[0], edge[1], current_label, direction))
            except:
                pdb.set_trace()
        
        ############################################################################################# 
        ############################################################################################# 
        
        


G.add_nodes_from(words_a)
G.add_edges_from([(edge[0], edge[1], {"relation":edge[2], "direction": edge[3]}) for edge in edges])

degrees = [(wa, G.degree(wa)) for wa in words_a]



CC = nx.connected_component_subgraphs(G.to_undirected(), copy=True)
G1 = [G.subgraph(C.nodes()) for C in CC]


#pdb.set_trace()
[g for g in G1 if len(g) > 2][0].edges(data=True)


[set([list(ed[2].values())[0] for ed in g.edges(data=True)]) for g in G1 if len(g) > 4]

basis_cycles = [[sc for sc in nx.cycle_basis(g.to_undirected())] for g in G1 if len(g) > 0]

if len(sum(basis_cycles, [])) == 0:
    print("No cycle")



mu_R = {}
d = 40
r_counter = 0
for cl in current_labels:
    mu_r = np.zeros(d)
    mu_r[r_counter] = 1
    mu_R[cl] = mu_r
    r_counter += 1


import queue

def updateProcedure(G, v, model):
    #create a queue Q
    Q = queue.Queue()
    #enqueue v onto Q
    Q.put(v)
    #mark v
    marks = []
    marks.append(v)
    while Q.qsize() > 0: 
        w = Q.get()
        #if w is what we are looking for then
        #    return w
        #for all edges e in G.adjacentEdges(w) do
            #x ← G.adjacentVertex(w, e)
        for x in G.neighbors(w):
            edge_data = G.get_edge_data(w,x)
            direction = edge_data["direction"]
            index_x = direction.split().index(x)

            if x not in marks:
                if index_x > 0:
                    model[x] = model[w] + mu_R[edge_data["relation"]]
                else:
                    model[x] = model[w] - mu_R[edge_data["relation"]]
                #mark x
                marks.append(x)
                #enqueue x onto Q
                Q.put(x)

    return model



def load_model(file_model):
    model = {}
    for line in file_model:
        line_split = line.strip().split()
        word = line_split[0]
        vector = np.array([float(ve) for ve in line_split[1:]])
        model[word] = vector
    return model


pre_existing_model = True
if pre_existing_model:
    file_model = open("model_major_component.txt", "r")
    model = load_model(file_model)
    file_model.close()
else:
    model = {}

# load from existing representations
for g in G1:
    if not pre_existing_model:
        start_offset = np.random.randint(5,10) # arbitrary : we can choose the different start of subgraphs as close (or not) using pointwise information?
        model[start_node] = np.random.multivariate_normal([start_offset for i in range(d)], np.eye(d))

    start_node = list(g.nodes())[0]  # arbritray choice, can be improved?
    try:
        model = updateProcedure(g.to_undirected(), start_node, model)
    except:
        continue 


n = len(model)
R = len(mu_R)
#S = np.zeros((R, n, n))

words_model = list(model.keys())
threshold = 0.5

couples_R = {}
for current_label in current_labels:
    couples_R[current_label] = []

precision = 0
true_negative = 0
true_positive = 0
total_negative = 0
total_positive = 0

false_positives = []
false_negatives = []
for r in range(R):
    print(r)
    S = np.zeros((n, n))
    print(r)
    for i in range(n):
        print(i)
        word_i = words_model[i]
        for j in range(n):
            word_j = words_model[j]
            
            S[i, j] = np.linalg.norm(mu_R[current_labels[r]] - (model[word_j] - model[word_i])) 
            if S[i,j] < threshold:
                couples_R[current_labels[r]].append((words_model[i], words_model[j])) 
                #pdb.set_trace() 
                if " ".join([words_model[i], words_model[j]]) in relation_to_pairs[current_labels[r]]:
                    true_positive += 1
                else:
                    false_positives.append((current_labels[r], word_i, word_j))
                total_positive += 1
            else:
                if " ".join([words_model[i], words_model[j]]) not in relation_to_pairs[current_labels[r]]:
                    true_negative += 1
                else:
                    false_negatives.append((current_labels[r], word_i, word_j))
                total_negative += 1

N = n**2

false_positive = total_positive - true_positive 
false_negative = total_negative - true_negative
PR = float(true_positive)/(true_positive + false_positive)
RE = float(true_positive)/(true_positive + false_negative)
F1 = 2*PR*RE/(PR+RE)


#[g for g in G1 if "impossible" in g.nodes()][0].nodes()


pdb.set_trace()
