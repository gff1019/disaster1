# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`


3.process_data.py在我的电脑上运行正常，日志在data/process_data运行日志.png

4.输入输出的注释已经修改

###文件介绍

##app包含网页程序和可视化代码

#templates：网页的程序
#run.py：可视化的代码

##data：包含数据文件和数据处理代码
#disaster_categories.csv：因变量数据
#disaster_messages.csv：自变量数据
#process_data.py：数据处理代码
#DisasterResponse.db：数据处理后结果的数据库

##model：包含模型代码和训练后模型的保存文件
#train_classifier.py：模型的训练代码
#classifier.pkl：训练后的模型
