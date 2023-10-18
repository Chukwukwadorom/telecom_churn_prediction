import numpy as np
import joblib
from keras.optimizers import Adam
from keras.layers import Input, Embedding, Concatenate, Dense, Flatten, Dropout, LeakyReLU
from keras.models import Model

def feature_engineer(train_df):

    #engineering some features

    ## categorical features
    train_df.drop("network_age", axis=1, inplace=True)
    train_df['Consistent_competitor'] = np.where(
        train_df['Most_Loved_Competitor_network_in_Month_1'] == train_df['Most_Loved_Competitor_network_in_Month_2'],
        'yes',
        'no'
    )
    train_df['Network_Upgrade'] = "same"

    train_df.loc[(train_df['Network_type_subscription_in_Month_1'] == '2G') &
                (train_df['Network_type_subscription_in_Month_2'] == '3G'), 'Network_Upgrade'] = "upgrade"

    train_df.loc[(train_df['Network_type_subscription_in_Month_1'] == '3G') &
                (train_df['Network_type_subscription_in_Month_2'] == '2G'), 'Network_Upgrade'] = "downgrade"

    # will enginner two numerical features. to ensure numerical stability i will add a small number, and use the log:
    small_number =  0.01
    Offnet_Over_Onnet= list(np.log(train_df['Total_Offnet_spend'] +small_number)/ np.log(train_df['Total_Onnet_spend_'] + small_number))
    SMS_Over_Data = list(np.log(train_df['Total_SMS_Spend'] + small_number) /np.log(train_df['Total_Data_Spend'] + small_number))

    ls_of_cols = list(train_df.columns)
    idx = ls_of_cols.index('Total_Call_centre_complaint_calls')

    train_df.insert(idx, "Offnet_Over_Onnet" , Offnet_Over_Onnet)
    train_df.insert(idx, "SMS_Over_Data" , SMS_Over_Data)


    return train_df




scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
def scale_and_transform(df):

    categorical_features = ["Network_type_subscription_in_Month_1", "Network_type_subscription_in_Month_2", "Most_Loved_Competitor_network_in_Month_1", "Most_Loved_Competitor_network_in_Month_2", 'Consistent_competitor','Network_Upgrade']
    numerical_features = [col for col in list(df.columns) if col not in categorical_features]  

    print("from scale_and_trans")
    print(categorical_features)
    print(numerical_features)
    df[numerical_features] = scaler.transform(df[numerical_features])

    for col in categorical_features:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    return df, categorical_features, numerical_features 

    


def create_model(df, categorical_features, numerical_features ):

  cat_emb = []
  input_layers = []
  num_nos_featues = len(list(df[numerical_features].columns))
  num_input_layer = Input(shape=(num_nos_featues,))

  input_layers.append(num_input_layer)

  embedding_size =  8
  for col in categorical_features:
    cat_input_layer = Input(shape=(1,))
    le = label_encoders[col]

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    embedding_layer = Embedding(input_dim=len(mapping), output_dim=8)(cat_input_layer)

    flatten_layer = Flatten()(embedding_layer)

    cat_emb.append(flatten_layer)
    input_layers.append(cat_input_layer)

  concatenated = Concatenate()([num_input_layer] + cat_emb)

  dense_layer = Dense(128, activation=LeakyReLU(alpha=0.1))(concatenated)
  x = Dropout(0.3)(dense_layer)
  output_layer = Dense(1, activation='sigmoid')(x)

  model = Model(inputs=input_layers, outputs=output_layer)
  model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

  return model
