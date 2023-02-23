import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, log_loss, roc_auc_score
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat
from deepctr_torch.models import DeepFM


def RecommenderDeepFM(data, sparse_features, dense_features, sequence_features, target, train_length):

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
    data.dropna(inplace=True)  # drop NA in dense features

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
    train = data[:train_length]
    test = data[train_length:]
    y_train = train[target].values.flatten()
    y_test = test[target].values.flatten()

    train_model_input = {name: train[name] for name in data.columns}
    train_model_input['usertag'] = np.stack(
        np.asarray(train_model_input['usertag']), axis=0)

    test_model_input = {name: test[name] for name in data.columns}
    test_model_input['usertag'] = np.stack(
        np.asarray(test_model_input['usertag']), axis=0)

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    DeepFMModel = DeepFM(linear_feature_columns, dnn_feature_columns,
                         task='binary', l2_reg_embedding=1e-5, seed=0, device=device)

    DeepFMModel.compile("adagrad", "binary_crossentropy",
                        metrics=["binary_crossentropy", "auc"], )

    DeepFMModel.fit(train_model_input, y_train, batch_size=256,
                    epochs=10, verbose=2, validation_split=0.2)

    return y_train, y_test, train_model_input, test_model_input, DeepFMModel


def main():
    data1 = pd.read_csv('./train.log.txt', sep='\t')
    train_length = data1.shape[0]
    data2 = pd.read_csv('./test.log.txt', sep='\t')
    data = pd.concat([data1, data2])
    data.drop(columns=['bidprice', 'nconversation', 'keypage',
                       'logtype', 'timestamp', 'urlid', 'nclick'], inplace=True)

    dense_features = ['slotprice', 'slotwidth', 'slotheight']
    sparse_features = ['bidid', 'ipinyouid', 'weekday', 'hour', 'adexchange', 'slotid',
                       'slotvisibility', 'slotformat', 'useragent', 'domain', 'IP', 'city', 'region', 'creative', 'url']
    sequence_features = ['usertag']
    target = ['click']

    c_train, c_test, ctrain_model_input, ctest_model_input, c_model = RecommenderDeepFM(
        data, sparse_features, dense_features, sequence_features, target, train_length)
    c_hat_train = c_model.predict(ctrain_model_input)
    print("")
    print("train logloss", round(log_loss(c_train, c_hat_train), 4))
    print("train auc", round(roc_auc_score(c_train, c_hat_train), 4))
    print("train r2", round(r2_score(c_train, c_hat_train), 4))

    c_hat_test = c_model.predict(ctest_model_input)
    print("")
    print("test logloss", round(log_loss(c_test, c_hat_test), 4))
    print("test auc", round(roc_auc_score(c_test, c_hat_test), 4))
    print("test r2", round(r2_score(c_test, c_hat_test), 4))
    torch.save(c_model.state_dict(), 'c_model.h5')
    # w_model.load_state_dict(torch.load('w_model.h5'))
    np.savetxt("./c_train.csv", c_train)
    np.savetxt("./c_test.csv", c_test)
    np.savetxt("./c_hat_train.csv", c_hat_train)
    np.savetxt("./c_hat_test.csv", c_hat_test)


if __name__ == '__main__':
    main()
