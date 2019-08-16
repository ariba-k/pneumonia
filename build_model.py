import tensorflow as tf
import datetime as dt
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications import InceptionV3, Xception, InceptionResNetV2, DenseNet121
from tensorflow.python.keras.layers import Conv2D, Dense, AveragePooling2D, BatchNormalizationV2, Dropout, MaxPool2D, Softmax, Flatten
from tensorflow.python.keras import Model, Sequential, Input
from tensorflow.python.keras.layers import Average
from augment_data import *
from load_data import DataGenFile, DataGenH5, LABEL_NORM_MAP, h5file_name
import os
import pickle


def select_gpu(gpu_id=-1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id != -1 else '0,1'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('RUNNING ON GPU #{}'.format(gpu_id))


# config
img_dim = (512, 512)
lab_dim = len(LABEL_NORM_MAP)
input_shape = img_dim + (3,)
valid_ratio = 0.2
augment_chances = {stretch_image: 0.05, flip_image: 0.2, rotate_image: 0.1}
load_from = 'h5'  # file|h5; h5 is like 10x faster, it's ^quite^ dashing
base_dir = '/mnt/data/pneumonia/'
augment_b4_h5 = True  # augment and then save to hdf5
augment_b4_training = False  # xor augment_b4_h5

nimages = -1  # -1 == all
gpu_id = -1
batch_size = 16
epochs = 50
lr = 1e-3
shuffle_size = 2048  # FIXME 16384
dtype = tf.float32
npdtype = np.float32

select_gpu(gpu_id)
tf.keras.backend.set_floatx('float32' if dtype == tf.float16 else 'float32')

if __name__ == '__main__':

    h5file = None
    if load_from == 'file':
        with open(base_dir + 'assembled.pickle', 'rb') as f:
            assembled = pickle.load(f)

        nvalid = int(len(assembled) * valid_ratio)
        train_src = assembled[nvalid:]
        valid_src = assembled[:nvalid]

        train_dg = DataGenFile(train_src, augment_chances, zscore_norm, lab_dim, img_dim, augment_b4_h5)
        valid_dg = DataGenFile(valid_src, augment_chances, zscore_norm, lab_dim, img_dim, augment_b4_h5)

    else:  # h5
        h5file_path = base_dir + h5file_name(nimages, img_dim, lab_dim, augment_b4_h5, augment_chances, valid_ratio)
        train_dg = DataGenH5(h5file_path, augment_chances, zscore_norm, training=True, augment=augment_b4_training, dtype=npdtype)
        valid_dg = DataGenH5(h5file_path, augment_chances, zscore_norm, training=False, augment=augment_b4_training, dtype=npdtype)
    ds_img_train = tf.data.Dataset.from_generator(train_dg, (dtype, dtype), (tf.TensorShape(img_dim), tf.TensorShape(lab_dim))) \
        .map(lambda img, lab: (tf.expand_dims(img, axis=-1), lab)) \
        .map(lambda img, lab: (tf.tile(img, tf.constant([1, 1, 3])), lab)) \
        .shuffle(shuffle_size) \
        .batch(batch_size, drop_remainder=True)
    ds_img_valid = tf.data.Dataset.from_generator(valid_dg, (dtype, dtype), (tf.TensorShape(img_dim), tf.TensorShape(lab_dim))) \
        .map(lambda img, lab: (tf.expand_dims(img, axis=-1), lab)) \
        .map(lambda img, lab: (tf.tile(img, tf.constant([1, 1, 3])), lab)) \
        .shuffle(shuffle_size) \
        .batch(batch_size, drop_remainder=True)

    # Inception  # runs with batch size of 32
    # print('Inception')
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     base_model_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    #     x0 = Flatten()(base_model_inception.get_output_at(-1))
    #     x0 = Dense(32, activation='relu')(x0)
    #     output0 = Dense(lab_dim, activation='sigmoid')(x0)
    #     model_inception = Model(base_model_inception.input, output0)
    #     model_inception.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
    #                              loss=tf.nn.sigmoid_cross_entropy_with_logits, metrics=['accuracy'])
    #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    #     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=lr)
    #     file_name = 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
    #     save_model = tf.keras.callbacks.ModelCheckpoint('{}'.format(file_name), monitor='val_loss')
    #     log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #     model_inception.fit(ds_img_train, epochs=epochs,
    #                         callbacks=[reduce_lr, early_stopping, save_model, tensorboard],
    #                         validation_data=ds_img_valid)

    # Xception  # runs with batch size of <32
    # print('Xception')
    # base_model_xception = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    # x1 = Flatten()(base_model_xception.get_output_at(-1))
    # x1 = Dense(32, activation='relu')(x1)
    # output1 = Dense(lab_dim, activation='sigmoid')(x1)
    # model_xception = Model(base_model_xception.input, output1)
    # model_xception.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
    #                          loss=tf.nn.sigmoid_cross_entropy_with_logits, metrics=['accuracy'])
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=lr)
    # file_name = 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
    # save_model = tf.keras.callbacks.ModelCheckpoint('{}'.format(file_name), monitor='val_loss')
    # log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # model_xception.fit(ds_img_train, epochs=epochs,
    #                    callbacks=[reduce_lr, early_stopping, save_model, tensorboard], validation_data=ds_img_valid)


    # Resnet  # runs with batch size of 32
    # print('Resnet')
    # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # x2 = Flatten()(base_model.get_output_at(-1))
    # x2 = Dense(32, activation='relu')(x2)
    # output2 = Dense(lab_dim, activation='sigmoid')(x2)
    # model_resnet = Model(base_model.input, output2)
    # model_resnet.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
    #                          loss=tf.nn.sigmoid_cross_entropy_with_logits, metrics=['accuracy'])
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=lr)
    # file_name = 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
    # save_model = tf.keras.callbacks.ModelCheckpoint('{}'.format(file_name), monitor='val_loss')
    # log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # model_resnet.fit(ds_img_train, epochs=epochs,
    #                  callbacks=[reduce_lr, early_stopping, save_model, tensorboard],
    #                  validation_data=ds_img_valid)


    print('InceptionResNetV2')  # runs with batch size of 32
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        x2 = Flatten()(base_model.get_output_at(-1))
        x2 = Dense(32, activation='relu')(x2)
        output2 = Dense(lab_dim, activation='sigmoid')(x2)
        model_resnet = Model(base_model.input, output2)
        model_resnet.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                             loss=tf.nn.sigmoid_cross_entropy_with_logits, metrics=['accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=lr, verbose=True)
        file_name = 'models/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
        save_model = tf.keras.callbacks.ModelCheckpoint('{}'.format(file_name), monitor='val_loss')
        log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model_resnet.fit(ds_img_train, epochs=epochs,
                         callbacks=[reduce_lr, early_stopping, save_model, tensorboard],
                         validation_data=ds_img_valid, validation_steps=len(train_dg) // batch_size)


    # print('DenseNet121')  # runs with batch size of <32
    # base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    # x2 = Flatten()(base_model.get_output_at(-1))
    # x2 = Dense(32, activation='relu')(x2)
    # output2 = Dense(lab_dim, activation='sigmoid')(x2)
    # model_resnet = Model(base_model.input, output2)
    # model_resnet.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
    #                          loss=tf.nn.sigmoid_cross_entropy_with_logits, metrics=['accuracy'])
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=lr)
    # file_name = 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
    # save_model = tf.keras.callbacks.ModelCheckpoint('{}'.format(file_name), monitor='val_loss')
    # log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # model_resnet.fit(ds_img_train, epochs=epochs,
    #                  callbacks=[reduce_lr, early_stopping, save_model, tensorboard],
    #                  validation_data=ds_img_valid)


def model_pretraining(model_name, num_labels):
    if model_name.lower() == 'resnet50':
        restnet_base_model = ResNet50(weights='imagenet', include_top=False,input_shape=input_shape)
        x = Flatten()(restnet_base_model.get_output_at(-1))
        x = Dense(64, activation='relu')(x)
        output = Dense(num_labels, activation='sigmoid')(x)
        model = Model(restnet_base_model.input, output, name='resnet')
    elif model_name.lower() == 'inceptionv3':
        inception_base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        x = Flatten()(inception_base_model.get_output_at(-1))
        x = Dense(64, activation='relu')(x)
        output = Dense(num_labels, activation='sigmoid')(x)
        model = Model(inception_base_model.input, output, name='inception')
    elif model_name.lower() == 'xception':
        xception_base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        x = Flatten()(xception_base_model.get_output_at(-1))
        x = Dense(64, activation='relu')(x)
        output = Dense(num_labels, activation='sigmoid')(x)
        model = Model(xception_base_model.input, output, name='xception')
    elif model_name.lower() == 'densenet':
        xception_base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        x = Flatten()(xception_base_model.get_output_at(-1))
        x = Dense(64, activation='relu')(x)
        output = Dense(num_labels, activation='sigmoid')(x)
        model = Model(xception_base_model.input, output, name='densenet')
    elif model_name.lower() == 'inceptionresnet':
        xception_base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        x = Flatten()(xception_base_model.get_output_at(-1))
        x = Dense(64, activation='relu')(x)
        output = Dense(num_labels, activation='sigmoid')(x)
        model = Model(xception_base_model.input, output, name='inceptionresnet')
    else:
        raise ValueError('Invalid model name!')
    return model


models = []
resnet = model_pretraining('resnet50', 15)
models.append(resnet)
inception = model_pretraining('inceptionv3', 15)
models.append(inception)
xception = model_pretraining('xception', 15)
models.append(xception)


def ensemble_training(models_list):
    outputs = [model.outputs[-1] for model in models]
    final_input = [model.input for model in models]
    final_output = Average()(outputs)
    ens_model = Model(final_input, final_output)
    return ens_model

ens_model = ensemble_training(models)

