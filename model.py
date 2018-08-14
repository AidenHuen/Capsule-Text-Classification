import pickle,config
# from Capsule_Keras import *
from keras.layers import *
from keras.models import *
from keras.regularizers import l2
from keras_utils import Capsule, AttentionWithContext,Attention
import dataProcess

def get_cnn(embedding):
    '''
    CNN which concatenate avg_poolong, max_pooling and attention
    :return: Model
    '''

    word_input = Input(shape=(config.padding_length,), dtype="int32")
    embed = Embedding(input_dim=embedding.shape[0],
        weights=[embedding],
        output_dim=embedding.shape[1],
        mask_zero=False,
        trainable=False
        )(word_input)

    conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=64, kernel_size=3, padding="valid")(embed)))


    # max_pool = GlobalMaxPool1D()(conv)
    # max_dropfeat = Dropout(0.3)(max_pool)
    #
    # avg_pool = GlobalAveragePooling1D()(conv)
    # avg_dropfeat = Dropout(0.3)(avg_pool)
    att = AttentionWithContext()(conv)
    att_dropfeat = Dropout(0.3)(att)
    # feat = concatenate([max_dropfeat, avg_dropfeat, att_dropfeat])

    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(att_dropfeat)))
    output = Dense(config.num_classes, activation="softmax")(fc)
    # output = Dense(y_dev.shape[1], activation="sigmoid")(fc)
    model = Model(inputs=word_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model


def get_grnn(embedding,n_recurrent=128, dropout_rate=0.2, l2_penalty=0.0001, mask_zero = True):
    '''
    GRU-based RNN
    :return: Model
    '''

    word_input = Input(shape=(None,), dtype="int32")
    embed = Embedding(input_dim=embedding.shape[0],
                      weights=[embedding],
                      output_dim=embedding.shape[1],
                      mask_zero=False,
                      trainable=True
                      )(word_input)

    x = SpatialDropout1D(dropout_rate)(embed)
    x = Bidirectional(
        CuDNNGRU(n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x)

    conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=64, kernel_size=3, padding="valid")(embed)))
    conv_caps = Capsule(
        num_capsule=10, dim_capsule=16,
        routings=3, share_weights=True)(conv)
    conv_flat = Flatten()(conv_caps)

    # x = Bidirectional(
    #     GRU(n_recurrent, return_sequences=True))(x)
    # feature = GlobalMaxPool1D()(x) # acc 65%
    feature1 = AttentionWithContext()(x)  # acc 64%
    # feature2 = GlobalAveragePooling1D(x)
    # feat = concatenate([feature,feature1, feature2])
    feat = concatenate([feature1, conv_flat])
    outputs = Dense(config.num_classes, activation="softmax")(feat)
    model = Model(inputs=word_input, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=['accuracy'])
    model.summary()
    return model

def get_cnn_capnet(embedding, n_capsule = 30, n_routings = 5, capsule_dim = 100, dropout_rate=0.2):
    """
    Conv_CapNet
    :param n_capsule:
    :param n_routings:
    :param capsule_dim:
    :param dropout_rate:
    :return: Model
    """

    word_input = Input(shape=(None,), dtype="int32")
    embed = Embedding(input_dim=embedding.shape[0],
              weights=[embedding],
              output_dim=embedding.shape[1],
              mask_zero=False,
              trainable=False
              )(word_input)
    conv_3 = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=128, kernel_size=3, padding="valid")(embed)))
    x_3 = Capsule(
        num_capsule=n_capsule, dim_capsule=capsule_dim,
        routings=n_routings, share_weights=True)(conv_3)
    x = Flatten()(x_3)
    # x = concatenate([x_3, x_4, x_5], axis=1)
    x = Dropout(dropout_rate)(x)
    # outputs = Dense(5, activation='sigmoid')(x)
    outputs = Dense(config.num_classes, activation='softmax')(x)
    model = Model(inputs=word_input, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='nadam',metrics=['accuracy'])
    model.summary()
    return Model

def get_grnn_capnet(embedding,n_capsule = 15, n_routings = 5, capsule_dim = 16, n_recurrent=100, dropout_rate=0.2, l2_penalty=0.0001):
    """
    RNN-CapNet
    :param n_capsule:
    :param n_routings:
    :param capsule_dim:
    :param n_recurrent:
    :param dropout_rate:
    :param l2_penalty:
    :return:
    """

    print(embedding)
    word_input = Input(shape=(None,), dtype="int32")
    embed = Embedding(input_dim=embedding.shape[0],
              weights=[embedding],
              output_dim=embedding.shape[1],
              mask_zero=False,
              trainable=False
              )(word_input)

    x = SpatialDropout1D(dropout_rate)(embed)
    # x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Bidirectional(
        CuDNNGRU(n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x)
    x = Capsule(
        num_capsule=n_capsule, dim_capsule=capsule_dim,
        routings=n_routings, share_weights=True)(x)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    # outputs = Dense(5, activation='sigmoid')(x)
    outputs = Dense(config.num_classes, activation='softmax')(x)
    model = Model(inputs=word_input, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='nadam',metrics=['accuracy'])
    model.summary()
    return Model

def train_han(mask_zero=False, n_recurrent=64, dropout_rate=0.2, l2_penalty=0.0001):
    x_train = pickle.load(open(config.train_pk3, "rb"))
    x_dev = pickle.load(open(config.dev_pk3, "rb"))
    y_train = dataProcess.get_Y(config.train_path)
    y_dev = dataProcess.get_Y(config.dev_path)
    embedding = pickle.load(open(config.word_embed_pk3,"rb"))


    """sentence_encoding"""
    # print(y_train,y_dev)
    sentence_input = Input(shape=(x_train.shape[2],), dtype="int32",name='word_input')
    x = Embedding(
        name= "embeding",
        input_dim = embedding.shape[0],
        weights = [embedding],
        output_dim = embedding.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )(sentence_input)

    x = Bidirectional(GRU(64, return_sequences=True))(x)
    # x = Capsule(
    #     num_capsule=10, dim_capsule=16,
    #     routings=5, share_weights=True)(x)

    x = Attention(x_train.shape[2])(x)
    sent_encode = Model(sentence_input, x, name='sent_encode')

    """doc_encoding"""
    doc_input = Input(shape=(x_train.shape[1], x_train.shape[2]), dtype="int32", name="sent_input")
    x = TimeDistributed(sent_encode)(doc_input)
    # x = SpatialDropout1D(dropout_rate)(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Capsule(
        num_capsule=5, dim_capsule=16,
        routings=5, share_weights=True)(x)
    # x = Attention(x_train.shape[1])(x)
    doc_encode = Flatten()(x)
    doc_encode = Activation(activation="relu")(BatchNormalization()(Dense(128)(doc_encode)))
    output = Dense(config.num_classes, activation='softmax')(doc_encode)

    model = Model(doc_input, output, name='doc_encode')
    model.compile(loss='categorical_crossentropy', optimizer="nadam", metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              validation_data=(x_dev, y_dev)
              )

# han:0.6594
if __name__ == "__main__":
    # train_han()
    x_train, y_train, x_dev, y_dev, embedding = dataProcess.load_dataset()
    model = get_grnn(embedding)
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              validation_data=(x_dev, y_dev)
              )
