from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import nltk
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import os
from django.conf import settings
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,Cyberbullying_Detection_Type,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Tweet_Meesage_Type(request):
    if request.method == "POST":

        if request.method == "POST":
            tweet_text = request.POST.get('tweet_text')

        data = pd.read_csv(os.path.join(settings.BASE_DIR, "Datasets.csv"), encoding='latin-1')

        def clean_text(text):

            text = text.lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            return text

            data['tweet_text'] = data['tweet_text'].apply(lambda x: clean_text(x))

        def apply_results(results):
            if (results == "not_cyberbullying"):
                return 0
            elif (results == "gender"):
                return 1
            elif (results == "religion"):
                return 2
            elif (results == "other_cyberbullying"):
                return 3
            elif (results == "age"):
                return 4
            elif (results == "ethnicity"):
                return 5

        data['Results'] = data['cyberbullying_type'].apply(apply_results)

        x = data['tweet_text']
        y = data['Results']

        cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))

        x = cv.fit_transform(x)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Multinomial Naive Bayes")
        from sklearn.naive_bayes import MultinomialNB

        nb_clf = MultinomialNB()

        nb_clf.fit(X_train, y_train)
        MultinomialNB()
        nb_pred = nb_clf.predict(X_test)
        mnb = accuracy_score(y_test, nb_pred) * 100
        print(mnb)
        print(confusion_matrix(y_test, nb_pred))
        print(classification_report(y_test, nb_pred))
        models.append(('nb_pred', nb_clf))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))



        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        tweet_text1 = [tweet_text]
        vector1 = cv.transform(tweet_text1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)


        if prediction == 0:
                val='not_cyberbullying'
        elif prediction == 1:
                val= 'gender'
        elif prediction == 2:
                val = 'religion'
        elif prediction == 3:
                val = 'other_cyberbullying'
        elif prediction == 4:
                val = 'age'
        elif prediction == 5:
                val = 'ethnicity'

        print(prediction)
        print(val)

        Cyberbullying_Detection_Type.objects.create(Tweet_Message=tweet_text,Prediction=val)

        return render(request, 'RUser/Predict_Tweet_Meesage_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Tweet_Meesage_Type.html')



