import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

features = df[['total_bill', 'size']]
target = df['tip']

print('Features:\n', features.head())
print('Target:\n', target.head())

X_train , X_test , Y_Train, Y_test = train_test_split(features , target , test_size=0.2 , random_state=42)
print('\nX_train:\n', X_train.shape)
print('\nX_test:\n', X_test.shape)

#Visualise dataset 
sns.pairplot(df, x_vars=['total_bill', 'size'], y_vars='tip', height=5, aspect=0.8, kind='scatter')
plt.show()