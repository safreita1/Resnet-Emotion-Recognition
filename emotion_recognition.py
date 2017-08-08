import tflearn
import numpy as np
from numpy import genfromtxt
from tflearn.data_preprocessing import ImagePreprocessing


if __name__ == "__main__":
    n = 5

    # Data loading and pre-processing
    X = np.asarray(genfromtxt('data/Training_Data.csv', delimiter=' ',  skip_header=1,  dtype=float))
    Y = np.asarray(genfromtxt('data/Training_Labels.csv', delimiter=',', skip_header=1, dtype=int))

    X_test = np.asarray(genfromtxt('data/Test_Data.csv', delimiter=' ',  skip_header=1,  dtype=float))
    Y_test = np.asarray(genfromtxt('data/Test_Labels.csv', delimiter=',', skip_header=1, dtype=int))
    predict_value = np.asarray(genfromtxt('test_image.csv', delimiter=',', dtype=float))

    predict_value = predict_value.reshape([-1, 48, 48, 1])

    # Reshape the images into 48x4
    X = X.reshape([-1, 48, 48, 1])
    X_test = X_test.reshape([-1, 48, 48, 1])

    # One hot encode the labels
    Y = tflearn.data_utils.to_categorical(Y, 7)
    Y_test = tflearn.data_utils.to_categorical(Y_test, 7)

    # Real-time preprocessing of the image data
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()

    # Building Residual Network
    net = tflearn.input_data(shape=[None, 48, 48, 1], data_preprocessing=img_prep, data_augmentation=img_aug)
    net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.residual_block(net, n, 16)
    net = tflearn.residual_block(net, 1, 32, downsample=True)
    net = tflearn.residual_block(net, n-1, 32)
    net = tflearn.residual_block(net, 1, 64, downsample=True)
    net = tflearn.residual_block(net, n-1, 64)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)

    # Regression
    net = tflearn.fully_connected(net, 7, activation='softmax')
    mom = tflearn.Momentum(learning_rate=0.1, lr_decay=0.0001, decay_step=32000, staircase=True, momentum=0.9)
    net = tflearn.regression(net, optimizer=mom,
                             loss='categorical_crossentropy')
    # Training
    model = tflearn.DNN(net, checkpoint_path='models/model_resnet_emotion',
                        max_checkpoints=20, tensorboard_verbose=0,
                        clip_gradients=0.)

    model.load('model_resnet_emotion-42000')

    model.fit(X, Y, n_epoch=150, snapshot_epoch=False, snapshot_step=500,
              show_metric=True, batch_size=128, shuffle=True, run_id='resnet_emotion')

    score = model.evaluate(X_test, Y_test)
    print'Test accuarcy: ', score

    #model.save('model.tfl')
    #prediction = model.predict(predict_value)
    #print prediction