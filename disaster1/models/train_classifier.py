import sys
# 基础包
import pandas as pd
import numpy as np
#英文文本处理包
##网页抓取
import requests
from bs4 import BeautifulSoup
##文本清洗
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
##文本特征化
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
#数据准备
from sqlalchemy import create_engine
#机器学习模型
##机器学习模型准备
from sklearn.model_selection import train_test_split,GridSearchCV
##模型选择
from sklearn.linear_model import LogisticRegression 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from xgboost.sklearn import XGBClassifier
##模型管道
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
##模型评估
from sklearn.metrics import classification_report,confusion_matrix

def load_data(database_filepath):
    """
    从数据库导入数据
    输入数据库名：database_filepath
    输出自变量（X）和因变量（Y）
    """
    engine = create_engine('sqlite:///%s' % (database_filepath))
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    X = pd.read_sql('SELECT message FROM DisasterResponse', engine)
    Y = df.drop(['message','original','genre','id'],axis=1) 
    
    return X.iloc[:10],Y.iloc[:10],list(Y.columns)
#     return X,Y,list(Y.columns)


def tokenize(text):
    """
    数据清理
    输入一段文本
    输出分词结果，词性转换，大小写转换
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    用管道构建模型，用网格搜索寻找最佳参数
    使用CountVectorizer，TfidfTransformer将拆分后的文本转换为tf-idf格式
    然后使用MultiOutputClassifier交结果级拆分成二分类预测
    模型拟合选择XGBClassifier
    因为模型速度较慢，所以只选择了一个网格搜索参数
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(XGBClassifier()))
                ])
    parameters = {
        'vect__max_df': [0.75, 0.5]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    模型评估使用classification_report方法，并且根据36组结果的平均值分析效果
    """
    y_pred=model.predict(X_test['message'].tolist())
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns=category_names
    for i in range(0,35):   
        print(classification_report(Y_test[category_names[i]], y_pred[category_names[i]])) 


def save_model(model, model_filepath): 
    """
    将模型model保存在model_filepath中，格式为xx.pkl
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)
    """

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'].tolist(), Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()