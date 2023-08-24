
import numpy as np
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from numpy.linalg import norm
from sklearn.metrics import accuracy_score
import pandas as pd


# Set paths
cbow_50 = "mr/50/cbow/mr-d50-m2-cbow.model"
sg_50 = "mr/50/sg/mr-d50-m2-sg.model"
fasttext_50 = 'mr/50/fasttext/mr-d50-m2-fasttext.model'
glove_50 = "mr/50/glove/vectors.txt"

cbow_100 = "mr/100/cbow/mr-d100-m2-cbow.model"
sg_100 = "mr/100/sg/mr-d100-m2-sg.model"
fasttext_100 = "mr/100/fasttext/mr-d100-m2-fasttext.model"

cbow_200 = "mr/200/cbow/mr-d200-m2-cbow.model"
sg_200 = "mr/200/sg/mr-d200-m2-sg.model"
fasttext_200 = "mr/200/fasttext/mr-d200-m2-fasttext.model"

cbow_300 = "mr/300/cbow/mr-d300-m2-cbow.model"
sg_300 = "mr/300/sg/mr-d300-m2-sg.model"
fasttext_300 = "mr/300/fasttext/mr-d300-m2-fasttext.model"

word_similarity_dataset = "Wordsimilarity_datasets/iiith_wordsim/marathi.txt"




def read_glove():
    with open(glove_50,"r",encoding='utf-8') as file:
        data = file.readlines()
    a = []
    for i in range(len(data)):
        if len(data[i].split()) == 51:
            a.append(data[i])
    data = a
    data = [i.split() for i in data]
    df = pd.DataFrame(data)
    df = df[:815983]
    vocab = df[0].to_list()
    df = df.drop(df.columns.to_list()[0], axis=1)
    df = df.astype(float)
    glove = dict(zip(vocab, df.values))
    return glove


# In[32]:


def cosine_similarity(a,b):
    ''' This function calculate cosine similarity between two vectors   '''
    if (norm(a)*norm(b) != 0):
        return np.dot(a,b)/(norm(a)*norm(b))
    else:
        return -1




def find_similarity(data, model_path, modeltype):
    # load model
    if modeltype == "fasttext":
        model = FastText.load(model_path)
    elif modeltype in  ["cbow","sg"]:
        model = Word2Vec.load(model_path)
    elif modeltype == "glove":
        glove = read_glove()
    
    similarity_score = []
    
    for pair in data:
        word_1 = pair[0]
        word_2 = pair[1]
        try:
            if modeltype == "glove":
                vector_1 = glove[word_1]
                vector_2 = glove[word_2]
            else:
                vector_1 = model.wv[word_1]
                vector_2 = model.wv[word_2]
        except:
            similarity_score.append(-1)
        else:
            similarity_score.append(cosine_similarity(vector_1,vector_2))
    return similarity_score



def get_dataframe(data, score,  labels, acc, threshold, name):
    # save result in csv file
    df_data = np.array(data)
    df_data = np.hstack((df_data, np.atleast_2d(score).T))
    df_data = np.hstack((df_data, np.atleast_2d(labels).T))
    l = np.bitwise_and(np.array(score) > threshold, np.array(labels) > threshold).astype(int)
    df_data = np.hstack((df_data, np.atleast_2d(l).T))
    df = pd.DataFrame(df_data, columns=['word_1','word_2','similarity_score','ground_truth_similarity_score','label'])
    if threshold == 0.4:
        i = 0
    elif threshold == 0.5:
        i = 1
    elif threshold == 0.6:
        i = 2
    elif threshold == 0.7:
        i = 3
    elif threshold == 0.8:
        i = 4
    
    
    
    df.loc[len(df)] = ["Acccuracy",acc[i],np.nan,np.nan,np.nan]
    name = name + str(threshold * 10) + ".csv"
    df.to_csv(name, index = False)


# In[86]:


def get_accurary(label,score):
    # calculaet accuracy with theshold
    accuracy = []
    pred = np.array(score) > 0.4
    accuracy.append(accuracy_score(np.array(label)>0.4, pred))
    pred = np.array(score) > 0.5
    accuracy.append(accuracy_score(np.array(label)>0.5, pred))
    pred = np.array(score) > 0.6
    accuracy.append(accuracy_score(np.array(label)>0.6, pred))
    pred = np.array(score) > 0.7
    accuracy.append(accuracy_score(np.array(label)>0.7, pred))
    pred = np.array(score) > 0.8
    accuracy.append(accuracy_score(np.array(label)>0.8, pred))
    return accuracy
    


# In[34]:


## Load word_similarity_dataset
with open("Wordsimilarity_datasets/iiith_wordsim/marathi.txt","r", encoding='utf-8') as file:
    data = file.readlines()
    data = [i.split("\t") for i in data]
    data = [[word for word in sublist if word!="\n"] for sublist in data]
    labels = [float(i[2].split("\n")[0]) for i in data if len(i) > 0]
    data = [["दहशतवादी" if word=="दहशतवादी\n" else word for word in sublist] for sublist in data]
    data = [i for i in data if len(i)]
    data = [i[:2] for i in data]



# calculate similarity score
print("calculate similarity score")
print("fasttext")
fasttext_50_score  = find_similarity(data,  fasttext_50, "fasttext")
fasttext_100_score = find_similarity(data, fasttext_100, "fasttext")
fasttext_200_score = find_similarity(data, fasttext_200, "fasttext")
fasttext_300_score = find_similarity(data, fasttext_300, "fasttext")
print("cbow")
cbow_50_score  = find_similarity(data,  cbow_50, "cbow")
cbow_100_score = find_similarity(data, cbow_100, "cbow")
cbow_200_score = find_similarity(data, cbow_200, "cbow")
cbow_300_score = find_similarity(data, cbow_300, "cbow")
print("sg")
sg_50_score  = find_similarity(data,  sg_50, "sg")
sg_100_score = find_similarity(data, sg_100, "sg")
sg_200_score = find_similarity(data, sg_200, "sg")
sg_300_score = find_similarity(data, sg_300, "sg")
print("glove")
glove_50_score  = find_similarity(data,  glove_50, "glove")
print("")


# calculate accuracy
print("Evaluating")
fasttext_50_acc = get_accurary(labels,fasttext_50_score)
fasttext_100_acc = get_accurary(labels,fasttext_100_score)
fasttext_200_acc = get_accurary(labels,fasttext_200_score)
fasttext_300_acc = get_accurary(labels,fasttext_300_score)

cbow_50_acc  = get_accurary(labels,cbow_50_score)
cbow_100_acc = get_accurary(labels,cbow_100_score)
cbow_200_acc = get_accurary(labels,cbow_200_score)
cbow_300_acc = get_accurary(labels,cbow_300_score)

sg_50_acc  = get_accurary(labels,sg_50_score)
sg_100_acc = get_accurary(labels,sg_100_score)
sg_200_acc = get_accurary(labels,sg_200_score)
sg_300_acc = get_accurary(labels,sg_300_score)

glove_50_acc = get_accurary(labels,glove_50_score)


# In[93]:

print("saving")
for t in [0.4,0.5,0.6,0.7,0.8]:
	df = get_dataframe(data, fasttext_50_score	, labels, fasttext_50_acc 	, t, "Q1_50_FastText_similarity_")
	df = get_dataframe(data, fasttext_100_score	, labels, fasttext_100_acc	, t, "Q1_100_FastText_similarity_")
	df = get_dataframe(data, fasttext_200_score	, labels, fasttext_200_acc	, t, "Q1_200_FastText_similarity_")
	df = get_dataframe(data, fasttext_300_score	, labels, fasttext_300_acc	, t, "Q1_300_FastText_similarity_")
	df = get_dataframe(data, cbow_50_score 		, labels, cbow_50_acc 		, t, "Q1_50_cbow_similarity_")
	df = get_dataframe(data, cbow_100_score		, labels, cbow_100_acc		, t, "Q1_100_cbow_similarity_")
	df = get_dataframe(data, cbow_200_score		, labels, cbow_200_acc		, t, "Q1_200_cbow_similarity_")
	df = get_dataframe(data, cbow_300_score		, labels, cbow_300_acc		, t, "Q1_300_cbow_similarity_")
	df = get_dataframe(data, sg_50_score 		, labels, sg_50_acc			, t, "Q1_50_sg_similarity_")
	df = get_dataframe(data, sg_100_score		, labels, sg_100_acc		, t, "Q1_100_sg_similarity_")
	df = get_dataframe(data, sg_200_score		, labels, sg_200_acc		, t, "Q1_200_sg_similarity_")
	df = get_dataframe(data, sg_300_score		, labels, sg_300_acc		, t, "Q1_300_sg_similarity_")
	df = get_dataframe(data, glove_50_score		, labels, glove_50_acc		, t, "Q1_50_glove_similarity_")
