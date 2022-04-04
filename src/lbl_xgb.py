import pandas as pd
from sklearn import preprocessing, metrics
import xgboost as xgb
import joblib
import itertools
def feature_engineering(df, cat_cols):
    #add new features to the categorical columns
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[
            :,
            c1 + "_" + c2
        ] = df[c1].astype(str) + "_" + df[c2].astype(str)

    return df
#no dropping the numerical columns
def run(fold):
    df =  pd.read_csv("../input/adult_folds.csv")
    #train a model without numerical columnbs
    num_cols = ['age', 'fnlwgt', 'capital.gain','education.num','capital.loss','hours.per.week']
    #mapping the target
    target_mapping = {
        "<=50K": 0,
        ">50K":1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    #categorical variables
    cat_cols = [
        c for c in df.columns if c not in num_cols
        and c not in ("kfold", "income")
    ]
    #add new features
    #df =  feature_engineering(df, cat_cols)

    features = [
        f for f in df.columns if f not in ("income", "kfold")
    ]
    #fill na values
    for col in features:
        #do not encode numerical columns
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")
    #label encoding
    for col in features:
        if col in cat_cols:
            lbl = preprocessing.LabelEncoder()
            #fit
            lbl.fit(df[col])
            #transform
            df.loc[:, col] = lbl.transform(df[col])
    #get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    #get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    #standardize the training data
    minVec = df_train[features].min().copy()
    maxVec = df_train[features].max().copy()
    df_train[features] = (df_train[features] - minVec/(maxVec - minVec))

    #x_train
    x_train = df_train[features].values
    #x_validation
    x_validation = df_valid[features].values
    
    #model intialization
    model = xgb.XGBClassifier(n_jobs=-1, max_depth=7)
    #model fit
    model.fit(x_train, df_train.income.values)
    #model predict proability of 1s = income >50k
    valid_preds = model.predict_proba(x_validation)[:,1]
    #area under the curve metri
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    #print results auc
    print(f"Folds = {fold}, AUC = {auc}")
    #save model with joblib
    filename = '../models/lbl_xgb_model.sav'
    joblib.dump(model, filename)

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
