import os
import random
import pandas as pd
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import tarfile
import urllib.request
from abc  import abstractmethod
from os.path import exists



##################################################################################

                    ## Multinomial Naive Bayes classification ##

##################################################################################



def parse_20_news(directory):
        print("Parsing 20_news...")
        paths = []
        names = []

        for filename in os.scandir(directory):
            if filename.is_dir:
                paths.append(filename.path)
        for path in paths:
            name = path.split('.')[-1]
            if name in names:
                names.append(name+"_"+path.split('.')[-2]) 
                continue
            names.append(name)
        DATA = {name:[] for name in names}
        for i in range(len(paths)):
            texts = b''
            for filename in os.scandir(paths[i]):
                if filename.is_file:
                    with open(filename.path, "rb" ) as file:
                            DATA[names[i]].append(file.read())   
        #print("Done")     
        return DATA


    
class Naive_Bayes:
    def __init__(self,vectorizer_type = "Count",test_train_ratio=.2,Data = None):
        self.vectorizer = None
        self.test_train_ratio = test_train_ratio
        self.classifier = MB()
        if vectorizer_type =="Count": self.vectorizer = CountVectorizer(decode_error='ignore')
        #elif vectorizer_type == "Hash": self.vectorizer = HashingVectorizer(decode_error='ignore')
        elif vectorizer_type == "Tfidf": self.vectorizer = TfidfVectorizer(decode_error='ignore')
        else: raise(ValueError("Invalid vectorizer type"))

        if not Data:
            self.init_20_news_data()
            self.Data = parse_20_news('20news-18828')
        else: 
            self.Data = Data
        self.Train_X,self.Train_Y,self.Test_X,self.Test_Y = self.split_data()

    def split_data(self):
        Train = []
        Test = []
        try:
            for key in self.Data:
                random.shuffle(self.Data[key])
                test = self.Data[key][0:int(len(self.Data[key])*self.test_train_ratio)]
                train = self.Data[key][int(len(self.Data[key])*self.test_train_ratio): len(self.Data[key])]
                Train+=([[doc,key] for doc in test])
                Test+=([[doc,key] for doc in train])
        except:
            raise(ValueError("Error in split_data()"))
           # print("issues")
        random.shuffle(Train)
        random.shuffle(Test)
        Train_X = [pair[0] for pair in Train]
        Test_X = [pair[0] for pair in Test]

        Train_Y = [pair[1] for pair in Train]
        Test_Y = [pair[1] for pair in Test]

        return Train_X,Train_Y,Test_X,Test_Y

    def train(self):
        print("Training...")
        try:
            transformed = self.vectorizer.fit_transform(self.Train_X)
            self.classifier.fit(transformed,self.Train_Y)
        except:
            raise(ValueError("Error in train()"))
        #print("Done")
    def test(self):
        print("Testing...")
        try:
            transformed = self.vectorizer.transform(self.Test_X)
            predict = self.classifier.predict(transformed)
        except:
            raise(ValueError("Error in test()"))
        #print("Done")
        return predict
    def confusion(self,prediction):
        return confusion_matrix(self.Test_Y,prediction,labels = list(set(self.Test_Y)))

    def init_20_news_data(self):
        if not exists("20news-18828"):
            print("Downloading 20 news data...")
            URL = "https://figshare.com/ndownloader/files/13356110"
            urllib.request.urlretrieve(URL, "temp_file")
            file = tarfile.open("temp_file")
            file.extractall()
            file.close()
            os.remove("temp_file")
            return
        print("20 news data exists in local directory")
        #print("Done")

##########################################################################

                             ## MAIN ##

###########################################################################



if __name__ == "__main__":
    Model_test = Naive_Bayes()
    Model_test.train()
    prediction =Model_test.test()
    print("Naive Bayes accuracy: ",metrics.accuracy_score(Model_test.Test_Y,prediction))





        





    
