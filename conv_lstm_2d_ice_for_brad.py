''' Demonstrates the use of a convolutional LSTM network
    Based off of F. Chollet's amazing:
    https://github.com/keras-team/keras/blob/master/examples/conv_lstm.py
    For any application this needs to be heavily modified!!!
'''

import copy
import matplotlib.pylab as plt
import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D


def style_subplot():
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim([0, 0.75])
    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(False)


def frames_to_input_output(frames, n_frames=15, n_samples=50, start_idx=0):
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
    MODEL_FILE_NAME = 'conv_lstm_2d_train_only.h5'

	# Load data from a .npz file
    with np.load('wais.npz', 'r') as data:
        frames = data['arr_0']

    # Normalize frames from 0 to 1
    frames = frames - frames.min()
    frames = frames / frames.max()

    # Format data
    _noise_frames = add_noise_to_frames(frames,
                                        noise_level=0.02)
    input_frames, output_frames = frames_to_input_output(frames,
                                                         n_frames=15,
                                                         n_samples=5000,
                                                         start_idx=0)

    # Build and train the model
    # model = build_model(frames)
    # model.fit(input_frames,
    #           output_frames,
    #           batch_size=2,
    #           epochs=10,
    #           validation_split=0.05)

    # # Save trained model
    # model.save(MODEL_FILE_NAME)

    # Load saved model and predict
    model = load_model(MODEL_FILE_NAME)

    # Predict some new frames
    frames_modified = copy.deepcopy(frames)
    start_frame = 6000
    n_predict_frames = 20
    n_rolling_frames = 15
    
    true_frames = frames_modified[start_frame:start_frame + n_predict_frames, :, :]
    predict_frames, _ = frames_to_input_output(frames_modified, n_frames=n_rolling_frames, n_samples=n_rolling_frames+n_predict_frames, start_idx=start_frame)
    predict_frames = predict_frames[0, :, :, :, :] # Delete leading dimension

    for i in range(n_predict_frames + 1):
        new_pos = model.predict(predict_frames[np.newaxis, :, :, :, :])
        new = new_pos[:, -1, :, :, :] # Get last frame of prediction
        predict_frames = np.concatenate((predict_frames, new), axis=0)

    # Visualize prediction vs. true
    true_magnitude = list()
    residual_magnitude = list()
    correlation = list()
    frame_index = list()

    plt.close('all')
    for i in range(n_predict_frames):
        fig = plt.figure(figsize=(15, 3), facecolor='w')
  
        ax = fig.add_subplot(1, 5, 1)
        plt.title('TRUE, time step: ' + str(start_frame + i))
        # plt.contour(true_frames[i, :, :], linewidth=0.5, colors='w', levels=contour_levels)
        plt.imshow(true_frames[i, :, :], interpolation='none', cmap='viridis')
        style_subplot()

        ax = fig.add_subplot(1, 5, 3)
        if i >= n_rolling_frames:
            plt.title('PREDICTIONS, time step: ' + str(start_frame + i), color='red')
        else:
            plt.title('AR STAGE, time step: ' + str(start_frame + i))
        # plt.contour(predict_frames[i, :, :, 0], linewidth=0.5, colors='w', levels=contour_levels)
        plt.imshow(predict_frames[i, :, :, 0], interpolation='none', cmap='viridis')
        style_subplot()

        ax = fig.add_subplot(1, 5, 5)
        plt.title('RESIDUALS, time step: ' + str(start_frame + i))
        # plt.contour(np.abs(true_frames[i, :, :] - predict_frames[i, :, :, 0]), linewidth=0.5, colors='w', levels=contour_levels)
        plt.imshow(np.abs(true_frames[i, :, :] - predict_frames[i, :, :, 0]), interpolation='none', cmap='viridis')
        style_subplot()

        # Calculate residual
        frame_index.append(start_frame + i)
        true_magnitude.append(np.sum((true_frames[i, :, :])**2))
        residual_magnitude.append(np.sum((true_frames[i, :, :] - predict_frames[i, :, :, 0])**2))
        correlation.append(np.corrcoef(true_frames[i, :, :].flatten(), predict_frames[i, :, :, 0].flatten())[0][1])

    # Plot residuals as a function of timestep
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    true_magnitude = np.array(true_magnitude)
    residual_magnitude = np.array(residual_magnitude)
    plt.plot(frame_index, residual_magnitude / true_magnitude * 100)
    plt.xlabel('frame index')
    plt.ylabel('% error')
    plt.title('Prediction error through time')
    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(False)

    plt.subplot(2, 1, 2)
    plt.plot(frame_index, np.array(correlation))
    plt.xlabel('frame index')
    plt.ylabel('correlation coefficient')
    plt.title('Prediction correlation through time')
    plt.ylim(0, 1)
    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(False)

    plt.show(block=False)



if __name__ == '__main__':
    main()
