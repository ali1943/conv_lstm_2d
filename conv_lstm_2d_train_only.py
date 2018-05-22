''' Demonstrates the use of a convolutional LSTM network
    Based off of F. Chollet's amazing:
    https://github.com/keras-team/keras/blob/master/examples/conv_lstm.py
    For any application this needs to be heavily modified!!!
'''

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
import numpy as np


def frames_to_input_output(frames, n_frames=15, n_samples=1000, start_idx=0):
    ''' Format 3D stack of images to 5D arrays for LSTM '''
    input_frames = np.zeros((n_samples, n_frames, frames.shape[1], frames.shape[2], 1),
                            dtype=np.float)
    output_frames = np.zeros((n_samples, n_frames, frames.shape[1], frames.shape[2], 1),
                             dtype=np.float)

    for i in range(n_samples):
        for t in range(n_frames):
            input_frames[i, t, :, :, 0] = frames[i + t + start_idx, :, :]
            output_frames[i, t, :, :, 0] = frames[i + t + start_idx + 1, :, :]
    return input_frames, output_frames


def add_noise_to_frames(frames, noise_level=0.00):
    ''' Add noise to observations '''
    noisy_frames = frames + noise_level * np.random.rand(frames.shape[0],
                                                         frames.shape[1],
                                                         frames.shape[2])
    return noisy_frames


def build_model(frames):
    ''' Assemble Keras convLSTM2D mode; '''
    n_conv_filters = 30
    model = Sequential()
    model.add(ConvLSTM2D(filters=n_conv_filters,
                         kernel_size=(3, 3),
                         input_shape=(None, frames.shape[1], frames.shape[2], 1),
                         padding='same',
                         return_sequences=True))
    model.add(ConvLSTM2D(filters=n_conv_filters,
                         kernel_size=(3, 3),
                         padding='same',
                         return_sequences=True))
    model.add(ConvLSTM2D(filters=n_conv_filters,
                         kernel_size=(3, 3),
                         padding='same',
                         return_sequences=True))
    model.add(ConvLSTM2D(filters=n_conv_filters,
                         kernel_size=(3, 3),
                         padding='same',
                         return_sequences=True))
    model.add(Conv3D(filters=1,
                     kernel_size=(3, 3, 3),
                     activation='sigmoid',
                     padding='same',
                     data_format='channels_last'))
    model.compile(loss='mae',
                  optimizer='sgd')
    return model


def main():
    ''' Take as input, time-dependent frames of shape
        (n_frames, AR frames, width, height, channels)
        and returns a movie of identical shape.  Mask as required.
    '''

    # Generate snthetic data.  Your read in should replace this
    frames = np.random.rand(1000, 28, 28) # MNIST-style shapes

    # Format data
    _noise_frames = add_noise_to_frames(frames,
                                        noise_level=0.02)
    input_frames, output_frames = frames_to_input_output(frames,
                                                         n_frames=15,
                                                         n_samples=6000,
                                                         start_idx=0)

    # Build and train the model
    model = build_model(frames)
    model.fit(input_frames,
              output_frames,
              batch_size=10,
              epochs=10,
              validation_split=0.05)

    # Save trained model
    model.save('conv_lstm_2d_train_only.h5')


if __name__ == '__main__':
    main()
