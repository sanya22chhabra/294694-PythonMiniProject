import nltk
import json
import pandas as pd
import _curses
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import pickle
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

with open("intents.json")as file:
    data= json.load(file)

try:
    with open("data.pickle","rb")as f:
        words,labels,training,op = pickle.load(f)
except:
    words=[]
    labels=[]
    docs_x=[]
    docs_y=[]

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds=nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    words=[stemmer.stem(w.lower())for w in words if w not in "?"]
    words=sorted(list(set(words)))
    labels=sorted(labels)

    training=[]
    op=[]
    out_empty=[0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag=[]
        wrds=[stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        op_row=out_empty[:]
        op_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        op.append(op_row)

    training=numpy.array(training)
    op=numpy.array(op)

    with open("data.pickle","wb")as f:
        pickle.dump((words,labels,training,op),f)

#modelbuilding
#tensorflow.reset_default_graph()
net= tflearn.input_data(shape=[None,len(training[0])])
net= tflearn.fully_connected(net,8)
net= tflearn.fully_connected(net,8)
net= tflearn.fully_connected(net,len(op[0]), activation="softmax")
net= tflearn.regression(net)
model=tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training,op,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("model.tflearn")

#predictions
def bagwords(s,words):
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower())for word in s_words]

    for se in s_words:
        for i , w in enumerate(words):
            if w==se:
                bag[i]=1
    return numpy.array(bag)

def chatbot():
    print("You can talk to the bot and type quit to stop")
    while True:
        inp=input("You: ")
        if inp == "quit":
            break
        if inp!= "":
            ans = model.predict([bagwords(inp, words)])
            ans_index = numpy.argmax(ans)
            tag=labels[ans_index]

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))

chatbot()