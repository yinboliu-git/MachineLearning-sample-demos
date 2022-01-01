#! /usr/bin/python3
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
train_data = TabularDataset('train.csv').iloc[0:10,:]
id, label = 'PassengerId', 'Survived'
predictor = TabularPredictor(label=label).fit(train_data.drop(columns=[id]))

test_data = TabularDataset('test.csv')
pred = predictor.predict(test_data.drop(columns=[id]))
sub = pd.DataFrame({id:test_data[id], label:pred})
sub.to_csv('submission.csv', index=False)

