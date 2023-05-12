import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import os
import openai
from pathlib import Path
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

class library:
    def __init__(self):
        openai.api_key = "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"
        self.model = openai.model = "gpt-3.5-turbo"

    def get_transcript(self):
        path = Path(__file__).parents[2]
        path = os.path.join(path, "call_samples", "sample1.mp3")
        audio_file = open("../../call_samples/sample2_german.mp3", "rb")
        transcript = openai.Audio.translate("whisper-1", audio_file)
        return transcript["text"]

    def get_summery(self, transcript):
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "summarize short this cal: " + transcript}
            ]
        )
        return completion.choices[0].message["content"]


#library = library()
#transcript = library.get_transcript()
#print(transcript)
transcript = "my computer does not work"

#Train Set
Train_data=pd.read_csv("Corpus.csv")
Train_data.drop(["SALES", "SUBSCRIPTION", "TECHNICAL", "BILLING", "OTHER"], 1,inplace=True)

Train_data['sentence'].dropna(inplace=True)
Train_data['sentence'] = [entry.lower() for entry in Train_data['sentence']]
Train_data['sentence'] = [word_tokenize(entry) for entry in Train_data['sentence']]
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Train_data['sentence']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    Train_data.loc[index,'text_final'] = str(Final_words)

Train_X=Train_data["text_final"]
Train_Y=Train_data["Label"]
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Train_data['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)

#Test Set

Test_X=pd.DataFrame({"sentence":[transcript]})
Test_Y=[1]
Test_X['sentence'] = [entry.lower() for entry in Test_X['sentence']]
Test_X['sentence']= [word_tokenize(entry) for entry in Test_X['sentence']]
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Test_X['sentence']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    Test_X.loc[index,'text_final'] = str(Final_words)


Test_X_Tfidf = Tfidf_vect.transform(Test_X["text_final"])
Test_Y = Encoder.fit_transform(Test_Y)



#--------------------------------------------------------------
# Predction Models
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y)*100)

# SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)


from keras import Sequential
from keras.layers import Dense
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def main():
    One_hot_test=keras.utils.to_categorical(Test_Y , num_classes=5)
    # one_hot_train = keras.utils.to_categorical(Train_Y, num_classes=5)

    # model = Sequential()
    # model.add(Dense(350, activation='tanh',input_dim=293))
    # model.add(Dense(250, activation='tanh'))
    # model.add(Dense(200, activation='tanh'))
    # model.add(Dense(100, activation='tanh'))
    # model.add(Dense(5, activation='softmax'))
    # model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    #
    # filepath="Final_model.hdf5"
    # checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # model.fit(Train_X_Tfidf, batch_size=32, y=one_hot_train, verbose=1,shuffle=True, epochs=50, callbacks=[checkpointer])
    # model.evaluate(Test_X_Tfidf,One_hot_test)
    # print(model.summary())
    # model.save('my_mode60.h5')

    model2 = load_model('my_mode60.h5')
    model2.evaluate(Test_X_Tfidf,One_hot_test)
    pred=model2.predict(Test_X_Tfidf)

    index={0:"SALES", 1:"SUBSCRIPTION",2:"TECHNICAL", 3:"BILLING", 4:"OTHER"}
    predictions=[]
    Test_actual=[]
    for a in pred:
        predictions.append(index[np.argmax(a)])

    for b in Test_Y:
        Test_actual.append(index[b])


    print("predictions ---> ",predictions)
    #print("Actual values ---> ",Test_actual)
    #Final_Result=pd.DataFrame({"file":Test_file_names,"Class":predictions })
    #Final_Result.to_csv("Not_so_bayesic_I-0SAJ4.csv",header=True)


if __name__ == "__main__":
    main()