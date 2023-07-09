# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Import data
train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
train.head()

# Clean data
df = train.dropna(inplace=True)
df = train.drop('Name', axis=1)
df.head()

# Create list of column names and check data type
cols = list(df.columns)
dtypes = df.dtypes
print(dtypes)

# Turn passenger ID into float
def pid_to_float(d):
    ks = []
    vs = []
    for ix, row in d.iterrows():
        val = row['PassengerId']
        ks.append(val)
        nval = val.replace('_', '')
        vs.append(nval)
    kvar = np.asarray(vs, dtype=np.float64)
    passid = {ks[i]: kvar[i] for i in range(len(ks))}
    return passid

passid = pid_to_float(df)
df['PassengerId'] = df['PassengerId'].map(passid)

# Non-float columns into floats
nf_cols = []
for col in cols:
    if df[col].dtype != np.float64:
        nf_cols.append(col)

def sb_to_float(d, n):
    vdl = []
    for c in nf_cols:
        col = d[c]
        vals = list(col[0:])
        v_dict = {} 
        i = float(n)
        for v in vals:
            if v in list(v_dict.keys()):
                continue
            else:
                v_dict[v] = i
                i += 1
        vdl.append(v_dict)
    return vdl

vdl = sb_to_float(df, 0)      
print(vdl[1])

# Change values in columns
for i in range(len(vdl)):
    cl = nf_cols[i]
    df[cl] = df[cl].map(vdl[i])

df.head

# Split data
x_train = df.drop('Transported', axis=1)
y_train = df['Transported']

# Normalize data
import tensorflow as tf

norm = tf.keras.layers.Normalization()
norm.adapt(x_train)
# Model
def build_and_compile_model(norm):
  model = tf.keras.Sequential([
      norm,
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(64, activation='sigmoid'),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(16, activation='sigmoid'),
      tf.keras.layers.Dense(16, activation='relu'),
      tf.keras.layers.Dense(8, activation='sigmoid'),
      tf.keras.layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adamax(0.0055))
  return model

linear_model = build_and_compile_model(norm)

history = linear_model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_split = 0.085)
# Plot data
import matplotlib.pyplot as plt

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 0.5])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history)
# Test results
test_results = {}

test_results['linear_model'] = linear_model.evaluate(
    x_train,
    y_train, verbose=0)

print(test_results)
# Convert test data
t = test
df2 = t.drop('Name', axis=1)

pid2 = pid_to_float(test)
cpid = df2['PassengerId']
df2['PassengerId'] = df2['PassengerId'].map(pid2)

def conv_df2():
    v2l = []
    for c in nf_cols:
        if c != 'Transported':
            col = df2[c]
            n = len(col) + 1
            vl = list(col)
            v2k = []
            v2v = []
            for val in col:
                v2k.append(val)
                v2v.append(n)
                n += 1
            v2vf = np.array(v2v, dtype=np.float64)
            v2 = {v2k[i]: v2vf[i] for i in range(len(v2k))}
            v2l.append(v2)
    return v2l

cv2 = conv_df2()

def merge_two_dicts(d1, d2):
    d3 = d1.copy()
    for key, value in d2.items():
        if key not in list(d1.keys()):
            d3[key] = value
    return d3

md = []
for i in range(len(cv2)):
    m2 = merge_two_dicts(vdl[i], cv2[i])
    md.append(m2)

for n in range(len(cv2)):
    col = nf_cols[n]
    for val in df2[col]:
        i = 0
        if val in list(md[n].keys()):
            nv = md[n].get(val)
            df2 = df2.replace(to_replace=val, value=nv)
            
# Normalize test data
norm.adapt(df2)
print(norm(df2))

# Make predictions
predictions = linear_model.predict(df2)
pcol = []
for p in predictions:
    pl = round(p[0])
    pcol.append(pl)
len(pcol)

# Create and save submission dataframe
Transported = pd.Series(pcol)
pcdc = {0: False, 1: True}
submit = pd.concat([cpid, Transported], axis=1)
submit = submit[:len(pcol)]
submit.columns = [submit.columns[0], 'Transported']
submit['Transported'] = submit['Transported'].map(pcdc)
submit.to_csv('/kaggle/working/submission.csv', index=False)
submit.head()
