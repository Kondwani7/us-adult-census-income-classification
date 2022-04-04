import pandas as pd
from sklearn import linear_model, preprocessing, metrics
import joblib
from keras.models import model_from_json

def run(fold):
    df =  pd.read_csv("../input/adult_folds.csv")
    #train a model without numerical columnbs
    num_cols = ['age', 'fnlwgt', 'capital.gain','education.num','capital.loss','hours.per.week']
    cat_cols = ['workclass', 'education', 'marital.status','occupation','relationship','race','sex','native.country']
    df = df.drop(num_cols, axis=1)
    #mapping the target
    target_mapping = {
        "<=50K": 0,
        ">50K":1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)

    features = [
        f for f in df.columns if f not in ("income", "kfold")
    ]
    #fill na values
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    #training data with folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    #validation data with folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    #one hot enncoding
    ohe = preprocessing.OneHotEncoder()
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data[features])
    #training data
    x_train = ohe.transform(df_train[features])
    #validation data
    x_valid = ohe.transform(df_valid[features])
    #initalize logistic regression model
    model = linear_model.LogisticRegression()
    #fit model
    model.fit(x_train, df_train.income.values)
    #predict the probability of 1s, income >50k with x_validation
    valid_preds = model.predict_proba(x_valid)[:, 1]
    #area under the curve
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    #print results auc
    print(f"Folds = {fold}, AUC = {auc}")
    #save model with joblib
    filename = '../models/ohe_logres_model.sav'
    joblib.dump(model, filename)
    
    #serialize model to json #neural network model
    '''
     model_json = model.to_json()
    with open("../models/ohe_logres_json", "w") as json_file:
        json_file.write(model_json)
    #serialize model weights to HDF5
    model.save_weights("ohe_logres_weights.h5")
    print("saved model")
    '''
   


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
