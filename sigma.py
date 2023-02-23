import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat
from deepctr_torch.models import DeepFM

def RecommenderDeepFM(data, sparse_features, dense_features, sequence_features, target, train_length, w_hat_train, w_hat_test):

    def split(x):
        key_ans = x.split(',')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    # 1.Label Encoding for sparse features,and process sequence features

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[sequence_features] = data[sequence_features].fillna('0', )
    data.dropna(inplace=True) # drop NA in dense features

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    # embedding size is hypyer, can be remove
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    #SequencefeaturesDict = {}
    for feat in sequence_features:
        #SequencefeaturesDict[feat] = Sequencefeatures(feat, data)
        # preprocess the sequence feature
        key2index = {}
        templist = list(map(split, data[feat].values))
        length = np.array(list(map(len, templist)))
        max_len = max(length)
        # Notice : padding=`post`
        templist = pad_sequences(templist, maxlen=max_len, padding='post', )

        use_weighted_sequence = False
        if use_weighted_sequence:
            varlen_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=len(key2index) + 1,
                                                                  embedding_dim=4), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature
        else:
            varlen_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=len(key2index) + 1,
                                                                  embedding_dim=4), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature

        linear_feature_columns += varlen_feature_columns
        dnn_feature_columns += varlen_feature_columns
        data[feat] = list(templist)

        # 3.generate input data for model
    train = data[0:train_length]
    test = data[train_length-1:-1]
    y_train =  train[target].values
    y_test = test[target].values

    var_train = (y_train-w_hat_train) ** 2
    var_test = (y_test-w_hat_test) ** 2

    train_model_input = {name: train[name] for name in data.columns}
    train_model_input['usertag'] = np.stack(np.asarray(train_model_input['usertag']), axis=0)

    test_model_input = {name: test[name] for name in data.columns}
    test_model_input['usertag'] = np.stack(np.asarray(test_model_input['usertag']), axis=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DeepFMModel = DeepFM(linear_feature_columns, dnn_feature_columns,
                         dnn_hidden_units=(128, 128, 128), task='regression',  dnn_activation='relu', device=device)

    DeepFMModel.compile("adam", "mse", metrics=['mse'], )

    DeepFMModel.fit(train_model_input, var_train, batch_size=512, epochs=30, verbose=2, validation_split=0.2, )
    return var_train, var_test, train_model_input, test_model_input, DeepFMModel



def main():
    data1 = pd.read_csv('./train.log.txt', sep='\t')
    train_length =data1.shape[0]
    data2 = pd.read_csv('./test.log.txt', sep='\t')
    data = pd.concat([data1, data2])
    data.drop(columns=['bidprice', 'nconversation', 'keypage',
                       'logtype', 'timestamp', 'urlid', 'nclick'], inplace=True)

    dense_features = ['slotprice', 'slotwidth', 'slotheight']
    sparse_features = ['bidid', 'ipinyouid', 'weekday', 'hour', 'adexchange', 'slotid',
                       'slotvisibility', 'slotformat', 'useragent', 'domain', 'IP', 'city', 'region', 'creative', 'url']
    sequence_features = ['usertag']

    # input the "payprice" to obtain sigma
    target = ['payprice']

    w_hat_train = np.loadtxt('w_hat_train.csv')
    w_hat_test = np.loadtxt('w_hat_test.csv')

    var_train, var_test, strain_model_input, stest_model_input, sigma_model = RecommenderDeepFM(data, sparse_features, dense_features, sequence_features, target, train_length, w_hat_train, w_hat_test)
    var_hat_train = sigma_model.predict(strain_model_input)
    var_hat_test = sigma_model.predict(stest_model_input)
    print("train mse", round(mean_squared_error(var_train, var_hat_train), 4))
    print("train r2", round(r2_score(var_train, var_hat_train), 4))

    print("")
    print("test mse", round(mean_squared_error(var_test, var_hat_test), 4))
    print("test r2", round(r2_score(var_test, var_hat_test), 4))

    torch.save(sigma_model.state_dict(), 'sigma_model.h5')
    # w_model.load_state_dict(torch.load('w_model.h5'))
    l=np.quantile(var_hat_train, 0.9, axis=0)
    new_var_train=np.clip(np.abs(var_hat_train), 0.01, l)
    new_var_test=np.clip(np.abs(var_hat_test), 0.01, l)

    sigma_w_hat = np.sqrt(new_var_train)
    sigma_w_hat_test = np.sqrt(new_var_test)

    np.savetxt('sigma_w_hat.csv',sigma_w_hat)
    np.savetxt('sigma_w_hat_test.csv',sigma_w_hat_test)

if __name__ == '__main__':
    main()
