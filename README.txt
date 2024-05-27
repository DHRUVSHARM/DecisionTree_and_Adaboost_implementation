done by : Dhruv Sharma
ds7042@rit.edu

The 2 processes that can be done using this submission
are based on 2 modules :

1) train.py : takes 3 arguments as required in the question and writes
a dt or ada model based on the cmd args.

2) predict.py : then this serialized file (using pickle) is deserialized and detected
as dt or ada and accordingly classification is made and output is displayed. takes 2 args exactly
as required in the question.

3) so for testing dt or ada test case, first run train.py, get the model and use it in predict.py

4) The best model is stores serialized in best.model , which can be directly used with predict

5) Note that for everything to run decision_tree.py and adaboost.py are crucial as they contain
learning method specific functions that are imported in train and predict.

6) examples : the examples I scraped using wikipedia api

7) for reference I also included the scraping and stop word creation code