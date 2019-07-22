import pandas as pd
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

# Date collection functions.
def convert_to_bag_of_words(text):
    list_of_letters = list(text)
    word = ''
    dictionary = dict()
    letters_list = list(string.ascii_letters)

    for i in range(len(list_of_letters)):
        if list_of_letters[i] in letters_list:
            word = word + list_of_letters[i]
        else:
            if word != 'I':
                word = word.lower()
            if word not in dictionary and word != '':
                dictionary[word] = 1
            elif word != '':
                dictionary[word] += 1
            word = ''
    return dictionary

def merge(a, b):
    for x in b.keys():
        if x in a.keys():
            a[x] = a[x] + b[x]
        else:
            a[x] = b[x]
    return a

def f(df, col_1, col_2):
    a = list(df[col_1])
    b = list(set(a))
    for x in b:
        a.remove(x)
    c = list(set(a))
    for j in c:
        a = list(df.loc[df[col_1] == j][col_2])

        for i in range(len(a)):
            if i != 0:
                a[0] = merge(a[0], a[i])

        d = list(df.loc[df[col_1] == j].index)

        for x in d:
            if d.index(x) != 0:
                df.drop(x, inplace=True)

def clean(c):
    d = list(c.values())
    limit = max(d) * 0.05

    dead_list = []

    for x in c.items():
        if x[1] <= limit:
            dead_list.append(x[0])

    for x in dead_list:
        c.pop(x)

# RNN type feature shaping functions.

# Convert all the text to a list of words.
def convert_to_list_of_words(text):
    list_of_letters = list(text)
    word = ''
    word_list = []
    letters_list = list(string.ascii_letters)

    for i in range(len(list_of_letters)):
        if list_of_letters[i] in letters_list:
            word = word + list_of_letters[i]
        else:
            if word != 'I':
                word = word.lower()
            if word != '':
                word_list.append(word)
            word = ''
    return word_list

# Construct the matrix for one text.
def convert_to_matrix(max_lenght, word_list, sentence):
    s = convert_to_list_of_words(sentence)
    b = word_list
    line = list(np.zeros((max_lenght+1), dtype=int))
        
    matrix = []
    for check in word_list:
        defualt_line = line[:]
        for index in range(0,len(s)):
            if check == s[index]:
                defualt_line[index] = 1
        matrix.append(defualt_line)
    return matrix

# Find the max length.
def find_max_length(text_list):
    max_lenght = 0
    for item in text_list:
        words = convert_to_list_of_words(item)
        lenght = len(words)
        if lenght > max_lenght:
            max_lenght = lenght
    return max_lenght

# Main function.
def create_RNN_feature(df, word_list):
    feature = []
    text_list = np.array(df['TEXT'])
    max_lenght = find_max_length(text_list)
    
    for item in list(df['TEXT']):
        m = convert_to_matrix(max_lenght, word_list, item)
        feature.append(m)
    return feature


# Load noteevents data.
df = pd.read_csv('NOTEEVENTS.csv', nrows=4000)
df.drop(columns=[
    'ROW_ID',
    'CHARTTIME',
    'CHARTDATE',
    'STORETIME',
    'CATEGORY',
    'DESCRIPTION',
    'CGID',
    'ISERROR'], inplace=True)
print('Finish loading', len(df.index), 'lines of NOTEEVENTS data.')


# Problem: Too slow...
# Convert notes to bag of words (very slow)...
bag_of_words = []

for i in range(len(df.TEXT)):
    bag_of_words.append(convert_to_bag_of_words(df.TEXT[i]))

df['bag_of_words'] = bag_of_words
f(df, 'HADM_ID', 'bag_of_words')
print('Converting to bag of words, finished.')
print('\n')
print(df)


# Load admissions data.
df1 = pd.read_csv('ADMISSIONS.csv', nrows=4000)
df1.drop(columns=['ROW_ID', 'ADMITTIME', 'DISCHTIME',
       'DEATHTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION',
       'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION',
       'MARITAL_STATUS', 'ETHNICITY', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS',
        'HAS_CHARTEVENTS_DATA'], inplace=True)
df1.dropna(inplace=True)
print('Finish loading', len(df1.index), 'lines of Admissions data.')


# Check how many data match.
check = df1['HADM_ID'].tolist()
good_data = []

for HI in df['HADM_ID']:
    if HI in check:
        good_data.append(HI)

print('Totally', len(good_data), 'data match!')


# Filter matched data for admission (relatively slow).
for HI in df1['HADM_ID']:
    if HI not in good_data:
         df1.drop(df1.loc[df1['HADM_ID'] == HI].index, inplace=True)
df1.set_index('HADM_ID', inplace=True)
df1.sort_index(inplace=True)
df1.drop(columns=['SUBJECT_ID'], inplace=True)


# Filter matched data for noteevents (relatively slow).
for HI in df['HADM_ID']:
    if HI not in good_data:
         df.drop(df.loc[df['HADM_ID'] == HI].index, inplace=True)
df.set_index('HADM_ID', inplace=True)
df.sort_index(inplace=True)


# Clip and find all words will be used.
total = dict()

for x in df['bag_of_words']:
    total = merge(total, x)

clean(total)


# Join data.
df.drop(columns=['bag_of_words'], inplace=True)
full_data = df.join(df1)

############################################# Data dispose over. #####################################################
######################################################################################################################


# Create word list.
word_list = list(total.keys())


# Create feature and label.
Y = list(full_data['HOSPITAL_EXPIRE_FLAG'])
feature_data = full_data.drop(columns=['HOSPITAL_EXPIRE_FLAG'])
X = np.array(create_RNN_feature(feature_data, word_list))

X_train = X
y_train = Y

# Training
model = Sequential()

model.add(LSTM(1, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=False))

model.add(Dense(1, input_shape = X.shape[1:]))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3)