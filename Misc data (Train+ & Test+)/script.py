import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

#surveyData[cols_to_norm] = StandardScaler().fit_transform(surveyData[cols_to_norm])

def normalize_standard(df):
    result = df.copy()
    result[result.columns[1:]] = StandardScaler().fit_transform(result[result.columns[1:]])
    return result

def normalize_minmax(df):
    result = df.copy()
    result[result.columns[1:]] =  MinMaxScaler().fit_transform(result[result.columns[1:]])
    return result

def normalize_power(df):
    result = df.copy()
    result[result.columns[1:]] =  PowerTransformer().fit_transform(result[result.columns[1:]])
    return result

############################################################ TESTE
###############################################
df = pd.read_csv("KDDTest+.csv")
# numerossss
df['class']  = df['class'].replace({k:i for i,k in enumerate(df['class'].unique(), 1)})
# colocar a class no começo
columns_fim = [df.columns[-1]]
columns_fim.extend(df.columns[:-1])
df = df.reindex(columns=columns_fim)

# 3 tipos de normalização.
df_standard_test = normalize_standard(df)
df_minmax_test = normalize_minmax(df)
df_power_test = normalize_power(df)
df_test = df
########################################## TREINOOOOOOOOOOOOOOOOOOOOOOOOOO
##########################################
df = pd.read_csv("KDDTrain+.csv")
# numerossss
df['class']  = df['class'].replace({k:i for i,k in enumerate(df['class'].unique(), 1)})
# colocar a class no começo
columns_fim = [df.columns[-1]]
columns_fim.extend(df.columns[:-1])
df = df.reindex(columns=columns_fim)

# 3 tipos de normalização.
df_standard_train = normalize_standard(df)
df_minmax_train = normalize_minmax(df)
df_power_train = normalize_power(df)
df_train = df
######################################### normz ##########################

df_standard_test.to_csv("kdd_test_standard.csv", sep=" ", index=False)
kdd_file = open("kdd_test_standard.csv", "r")
kdd_file.readline()
first_line=1
samples, features  = df_standard_test.shape
classes = 2
with open('kdd_test_standard.fem', 'w') as f:
    if first_line:
        f.write(f"{classes} {samples} {features-1}\n")
        #opf format header
        #f.write(f"{samples} {classes} {features-1}\n")
        first_line=0
    for i in kdd_file.readlines():
        f.write(i)

df_minmax_test.to_csv("kdd_test_minmax.csv", sep=" ", index=False)
kdd_file = open("kdd_test_minmax.csv", "r")
kdd_file.readline()
first_line=1
with open('kdd_test_minmax.fem', 'w') as f:
    if first_line:
        f.write(f"{classes} {samples} {features-1}\n")
        #opf format header
        #f.write(f"{samples} {classes} {features-1}\n")
        first_line=0
    for i in kdd_file.readlines():
        f.write(i)

df_power_test.to_csv("kdd_test_power.csv", sep=" ", index=False)
kdd_file = open("kdd_test_power.csv", "r")
kdd_file.readline()
first_line=1
with open('kdd_test_power.fem', 'w') as f:
    if first_line:
        f.write(f"{classes} {samples} {features-1}\n")
        #opf format header
        #f.write(f"{samples} {classes} {features-1}\n")
        first_line=0
    for i in kdd_file.readlines():
        f.write(i)

df_standard_train.to_csv("kdd_train_standard.csv", sep=" ", index=False)
kdd_file = open("kdd_train_standard.csv", "r")
kdd_file.readline()
first_line=1
samples, features  = df_standard_train.shape
classes = 2
with open('kdd_train_standard.fem', 'w') as f:
    if first_line:
        f.write(f"{classes} {samples} {features-1}\n")
        #opf format header
        #f.write(f"{samples} {classes} {features-1}\n")
        first_line=0
    for i in kdd_file.readlines():
        f.write(i)

df_minmax_train.to_csv("kdd_train_minmax.csv", sep=" ", index=False)
kdd_file = open("kdd_train_minmax.csv", "r")
kdd_file.readline()
first_line=1
with open('kdd_train_minmax.fem', 'w') as f:
    if first_line:
        f.write(f"{classes} {samples} {features-1}\n")
        #opf format header
        #f.write(f"{samples} {classes} {features-1}\n")
        first_line=0
    for i in kdd_file.readlines():
        f.write(i)

df_power_train.to_csv("kdd_train_power.csv", sep=" ", index=False)
kdd_file = open("kdd_train_power.csv", "r")
kdd_file.readline()
first_line=1
with open('kdd_train_power.fem', 'w') as f:
    if first_line:
        f.write(f"{classes} {samples} {features-1}\n")
        #opf format header
        #f.write(f"{samples} {classes} {features-1}\n")
        first_line=0
    for i in kdd_file.readlines():
        f.write(i)
#############################################################################
classes=2

df_minmax10_test = df_minmax_test.sample(frac=0.4,random_state=28122014)
df_minmax10_test = df_minmax10_test.reset_index(drop=True)
df_minmax10_test.loc[df_minmax10_test['class'] != 1, 'class'] = 2
samples, features  = df_minmax10_test.shape
df_minmax10_test.to_csv("kdd_test_minmax10.csv", sep=" ")
kdd_file = open("kdd_test_minmax10.csv", "r")
kdd_file.readline()
first_line=1
with open('kdd_test_minmax10.opf', 'w') as f:
    if first_line:
        #f.write(f"{classes} {samples} {features-1}\n")
        #opf format header
        f.write(f"{samples} {classes} {features-1}\n")
        first_line=0
    for i in kdd_file.readlines():
        f.write(i)


df_minmax10_train = df_minmax_train.sample(frac=0.4,random_state=28122014)
df_minmax10_train = df_minmax10_train.reset_index(drop=True)
df_minmax10_train.loc[df_minmax10_train['class'] != 1, 'class'] = 2
samples, features  = df_minmax10_train.shape
df_minmax10_train.to_csv("kdd_train_minmax10.csv", sep=" ")
kdd_file = open("kdd_train_minmax10.csv", "r")
kdd_file.readline()
first_line=1
with open('kdd_train_minmax10.opf', 'w') as f:
    if first_line:
        #f.write(f"{classes} {samples} {features-1}\n")
        #opf format header
        f.write(f"{samples} {classes} {features-1}\n")
        first_line=0
    for i in kdd_file.readlines():
        f.write(i)


df_notnorm = df_test.sample(frac=0.4,random_state=28122014)
df_notnorm = df_notnorm.reset_index(drop=True)
df_notnorm.loc[df_notnorm['class'] != 1, 'class'] = 2
samples, features  = df_notnorm.shape
df_notnorm.to_csv("kdd_test_minmax10.csv", sep=" ")
kdd_file = open("kdd_test_minmax10.csv", "r")
kdd_file.readline()
first_line=1
with open('kdd_test_minmax10.opf', 'w') as f:
    if first_line:
        #f.write(f"{classes} {samples} {features-1}\n")
        #opf format header
        f.write(f"{samples} {classes} {features-1}\n")
        first_line=0
    for i in kdd_file.readlines():
        f.write(i)


df_notnorm = df_train.sample(frac=0.4,random_state=28122014)
df_notnorm = df_notnorm.reset_index(drop=True)
df_notnorm.loc[df_notnorm['class'] != 1, 'class'] = 2
samples, features  = df_notnorm.shape
df_notnorm.to_csv("kdd_train_minmax10.csv", sep=" ")
kdd_file = open("kdd_train_minmax10.csv", "r")
kdd_file.readline()
first_line=1
with open('kdd_train_minmax10.opf', 'w') as f:
    if first_line:
        #f.write(f"{classes} {samples} {features-1}\n")
        #opf format header
        f.write(f"{samples} {classes} {features-1}\n")
        first_line=0
    for i in kdd_file.readlines():
        f.write(i)
