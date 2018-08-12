import pickle,config
# from Capsule_Keras import *
from keras.layers import *
from keras.models import *
from keras.regularizers import l2
from keras_utils import Capsule
def train_cnn():
    x_train, y_train = pickle.load(open(config.train_pk, "rb"))
    x_dev, y_dev = pickle.load(open(config.dev_pk,"rb"))
    embedding = pickle.load(open(config.word_embed_pk,"rb"))
    print(embedding)
    word_input = Input(shape=(None,), dtype="int32")
    embed = Embedding(input_dim=embedding.shape[0],
              weights=[embedding],
              output_dim=embedding.shape[1],
              mask_zero=False,
              trainable=False
              )(word_input)
    conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=64, kernel_size=3, padding="valid")(embed)))
    # conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=128, kernel_size=3, padding="valid")(conv)))
    pool = GlobalMaxPool1D()(conv)
    dropfeat = Dropout(0.3)(pool)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(config.num_classes, activation="softmax")(fc)
    # output = Dense(y_dev.shape[1], activation="sigmoid")(fc)
    model = Model(inputs=word_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    model.fit(x_train,y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              validation_data=(x_dev, y_dev)
              )

def train_cnn_capnet(n_capsule = 15, n_routings = 5, capsule_dim = 32,dropout_rate=0.2):
    x_train, y_train = pickle.load(open(config.train_pk, "rb"))
    x_dev, y_dev = pickle.load(open(config.dev_pk,"rb"))
    embedding = pickle.load(open(config.word_embed_pk,"rb"))
    print(embedding)
    word_input = Input(shape=(None,), dtype="int32")
    embed = Embedding(input_dim=embedding.shape[0],
              weights=[embedding],
              output_dim=embedding.shape[1],
              mask_zero=False,
              trainable=False
              )(word_input)
    conv_3 = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=128, kernel_size=3, padding="valid")(embed)))
    conv_4 = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=128, kernel_size=4, padding="valid")(embed)))
    conv_5 = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=128, kernel_size=5, padding="valid")(embed)))
    x_3 = Capsule(
        num_capsule=n_capsule, dim_capsule=capsule_dim,
        routings=n_routings, share_weights=True)(conv_3)
    x_4 = Capsule(
        num_capsule=n_capsule, dim_capsule=capsule_dim,
        routings=n_routings, share_weights=True)(conv_4)
    x_5 = Capsule(
        num_capsule=n_capsule, dim_capsule=capsule_dim,
        routings=n_routings, share_weights=True)(conv_5)
    x_3 = Flatten()(x_3)
    x_4 = Flatten()(x_4)
    x_5 = Flatten()(x_5)
    x = concatenate([x_3, x_4, x_5], axis=1)
    x = Dropout(dropout_rate)(x)
    # outputs = Dense(5, activation='sigmoid')(x)
    outputs = Dense(config.num_classes, activation='softmax')(x)
    model = Model(inputs=word_input, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='nadam',metrics=['accuracy'])
    model.summary()
    model.fit(x_train,y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              validation_data=(x_dev, y_dev)
              )

def train_rnn_capnet(n_capsule = 15, n_routings = 5, capsule_dim = 16,n_recurrent=100, dropout_rate=0.2, l2_penalty=0.0001):
    x_train, y_train = pickle.load(open(config.train_pk, "rb"))
    x_dev, y_dev = pickle.load(open(config.dev_pk,"rb"))
    embedding = pickle.load(open(config.word_embed_pk,"rb"))
    print(embedding)
    word_input = Input(shape=(None,), dtype="int32")
    embed = Embedding(input_dim=embedding.shape[0],
              weights=[embedding],
              output_dim=embedding.shape[1],
              mask_zero=False,
              trainable=False
              )(word_input)

    x = SpatialDropout1D(dropout_rate)(embed)
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
    model.fit(x_train,y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              validation_data=(x_dev, y_dev)
              )


train_cnn_capnet()