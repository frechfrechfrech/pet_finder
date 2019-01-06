import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('max_columns',500)
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def dummy_breeds(X, breed_lookup):
    '''
    Take in dataframe of all datapoints and return df with columns for all
    possible breed values where each row represents the breed options of that animal
    X: df from train.csv or test.csv
    breed_lookup: df from breed_labels.csv
    returns breed_dummies: df of dummies describing breed of animal
    '''
    breed_dummies = pd.DataFrame(np.zeros((X.shape[0],breed_lookup.shape[0])),\
                        columns = breed_lookup.BreedID, index = X.index)
    breed_dummies[0] = 0.0
    breed_dummies[X.Breed1.unique()] = breed_dummies[X.Breed1.unique()] + pd.get_dummies(X['Breed1'])
    breed_dummies[X.Breed2.unique()] = breed_dummies[X.Breed2.unique()] + pd.get_dummies(X['Breed2'])
    breed_dummies = breed_dummies.replace(2,1)
    breed_dummies.loc[breed_dummies.sum(axis=1)>1,0] = 0
    breed_dummies.columns = 'breed_'+breed_dummies.columns.astype('str')

    return breed_dummies

def dummy_colors(X, color_lookup):
    '''
    Take in dataframe of all datapoints and return df with columns for all
    possible breed values where each row represents the breed options of that animal
    X: df from train.csv or test.csv
    color_lookup: df from color_labels.csv
    returns color_dummies: df of dummies describing color of animal
    '''
    color_dummies = pd.DataFrame(np.zeros((X.shape[0],color_lookup.shape[0])),\
                        columns = color_lookup.ColorID, index = X.index)
    color_dummies[0] = 0.0
    color_dummies[np.sort(X.Color1.unique())] = color_dummies[np.sort(X.Color1.unique())] + pd.get_dummies(X['Color1'])
    color_dummies[np.sort(X.Color2.unique())] = color_dummies[np.sort(X.Color2.unique())] + pd.get_dummies(X['Color2'])
    color_dummies[np.sort(X.Color3.unique())] = color_dummies[np.sort(X.Color3.unique())] + pd.get_dummies(X['Color3'])

    color_dummies = color_dummies.replace(2,1).replace(3,1)
    color_dummies.loc[color_dummies.sum(axis=1)>1,0] = 0
    color_dummies.columns = 'color_'+color_dummies.columns.astype('str')

    return color_dummies


def preprocess_data_basic(X, breed_lookup, color_lookup):
    breed_dummies = dummy_breeds(X, breed_lookup)
    color_dummies = dummy_colors(X, color_lookup)
    dummies = color_dummies.merge(breed_dummies, left_index=True, right_index=True)
    X = X.loc[:,(np.logical_not(X.columns.str.contains('Breed')))&\
                (np.logical_not(X.columns.str.contains('Color')))]
    X_w_dummies = pd.concat([X,dummies], axis=1)

    return X_w_dummies

if __name__=='__main__':
    breed_lookup = pd.read_csv('data/breed_labels.csv')
    color_lookup = pd.read_csv('data/color_labels.csv')
    train = pd.read_csv('data/train.csv')

    # random_assignment_score
    y = train['AdoptionSpeed'].values
    y_pred = np.random.randint(0,5,size=y.shape)
    random_score = cohen_kappa_score(y, y_pred)
    # baseline_score = 0.00608257794913003
    print('Just Guessing Train Score:{:.2f}'.format(random_score))
    # numeric_only

    X = train.drop('AdoptionSpeed', axis=1).select_dtypes('int','float') # only use numbers
    X_w_dummies = preprocess_data_basic(X, breed_lookup, color_lookup)


    X_train, X_test, y_train, y_test = train_test_split(X_w_dummies, y, test_size=0.2, random_state=42)


    rf = RandomForestClassifier(n_estimators =100, oob_score=True)
    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    rf_train_score = cohen_kappa_score(y_train, y_pred_train)
    rf_test_score = cohen_kappa_score(y_test, y_pred_test)

    print('Random Forest Train Score:{:.2f}'.format(rf_train_score))
    print('Random Forest Test Score:{:.2f}'.format(rf_test_score))

    test = pd.read_csv('data/test/test.csv')
