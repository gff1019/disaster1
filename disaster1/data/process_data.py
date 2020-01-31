import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ##加载自变量和因变量并且将自变量和因变量合并
    #输入自变量路径：messages_filepath
    #输入因变量路径：categories_filepath
    #返回合并后数据集：df
    messages = pd.read_csv(messages_filepath,dtype=str) #导入自变量
    categories = pd.read_csv(categories_filepath,dtype=str) #导入因变量
    df = pd.merge(messages,categories)#自变量和因变量合并
    return df

def clean_data(df):
    #清洗数据集合
    ##将因变量数据拆分成36个字段，并且替换重复的内容，使其变成数值变量
    ##数据集合排重

    ##变量的按照;拆分
    categories = df['categories'].str.split(';',expand=True)
    ##变量的值替换成数值，按照第一行作为多分类的字段名
    row = categories.iloc[0]
    row1 = row.copy()
    for i in range(0,35):
        row1[i] = row1[i].replace('-1','').replace('-0','')
    category_colnames = row1
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = pd.to_numeric(categories[column])
    ## 替换原始的因变量的字段
    df = df.drop(['categories'],axis=1)
    df=pd.merge(df,categories,left_index=True,right_index=True)
    ##对数据进行去重
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    #将处理后的数据保存到数据库中，数据库名database_filename，对应表名DisasterResponse
    engine = create_engine('sqlite:///%s' % (database_filename))
    df.to_sql('DisasterResponse', engine, index=False) 
    return database_filename

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()