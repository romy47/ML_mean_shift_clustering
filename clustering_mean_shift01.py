from __future__ import division
import matplotlib.pyplot as plot
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import MeanShift
from sklearn import preprocessing, cross_validation
import pandas as pd
import numpy as np

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
def convert_to_numeric(df):
    cols = df.columns.values
    for col in cols:
        text_digit_dic = {}
        def get_int_of_text(text):
            return text_digit_dic[text]
        if df[col].dtype!=np.int64 and df[col].dtype!=np.float64:
            col_values = df[col].values.tolist()
            unique_contents = set(col_values)
            x = 0
            for unique in unique_contents:
                if unique not in text_digit_dic:
                    text_digit_dic[unique] = x
                    x+=1
            df[col] = list(map(get_int_of_text, df[col]))
    return df;
df = convert_to_numeric(df)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = MeanShift()
clf.fit(X)
cluster_centers = clf.cluster_centers_
labels = clf.labels_
original_df['cluster_group'] = np.nan
for i in range(len(original_df)):
    original_df['cluster_group'].iloc[i] = labels[i]
survival_rates = {}
for i in range(len(np.unique(labels))):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_clusters = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_clusters)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
for i in range(len(survival_rates)):
    print ('Cluster: ' + str(i) + ', Survival rate of passengers of this cluster: ' + str(survival_rates[i]))

print('\n')
print('cluster with higher survival rate of titanic passengers are comprised of passengers who were mainly female and had first class ticket. These features grouped them into this cluster.')