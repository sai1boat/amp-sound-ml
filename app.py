import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

import os.path
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from util import load, spectrogram


class AmpSoundML:
    def __init__(self):
        self.model = None

    def load_wave(self, file_path) -> tf.Tensor:
        return load(file_path)

    def prepare_files_for_one_species(self, species_name,
                                      directory_pos, directory_neg) -> tf.data.Dataset:
        pos = tf.data.Dataset.list_files(os.path.join(directory_pos, "*.wav"))
        pos = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
        neg = tf.data.Dataset.list_files(os.path.join(directory_neg, "*.wav"))
        neg = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
        return pos.concatenate(neg), species_name

    def run_data_pipeline(self, m_data):
        m_data = m_data.map(spectrogram)
        m_data = m_data.cache()
        m_data = m_data.shuffle(buffer_size=1000)
        m_data = m_data.batch(16)
        m_data = m_data.prefetch(8)
        return m_data

    def build_model(self, training_data, test_data=None):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(186, 257, 1)))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile('Adam', loss='BinaryCrossentropy',
                      metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        model.summary()
        print("Training model...")
        hist = model.fit(training_data, epochs=4, validation_data=test_data, steps_per_epoch=2, verbose=1)
        plt.title('Model Training Metrics')
        plt.plot(hist.history['loss'], label='loss')
        if 'val_loss' in hist.history:
            plt.plot(hist.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

        return model


    def build_stft_data(self, m_data):
            return self.run_data_pipeline(m_data)

if __name__ == "__main__":
    app = AmpSoundML()
    fox_pos_dir = "sounds/john/fox_pos"
    fox_neg_dir = "sounds/john/fox_neg"
    all_fox_filepaths_and_bools, species_name = app.prepare_files_for_one_species("fox", fox_pos_dir, fox_neg_dir)
    # spec = spectrogram(all_fox_filepaths_and_bools.as_numpy_iterator().next())
    # plot_spectrogram(spec[0])
    stft_data = app.build_stft_data(all_fox_filepaths_and_bools)
    print(f"Total dataset size: {len(stft_data)}")

    # Split data: ~67% training, ~33% testing
    # First unbatch to work with individual samples, not batches
    all_samples = stft_data.unbatch()
    train_samples = all_samples.take(14)  # Take 14 samples for training
    test_samples = all_samples.skip(14)   # Skip 14, take remaining samples

    # Re-batch each split
    train = train_samples.batch(16).repeat()  # Repeat training data across epochs
    test = test_samples.batch(16)  # Test data (no repeat needed)

    print("Training dataset spec:")
    print(train.element_spec)
    print("\nTest dataset spec:")
    print(test.element_spec)

    model = app.build_model(train, test)

    # Get test data for evaluation
    try:
        # Use the actual test dataset we created
        X_test, y_test = [], []
        for spec, label in test_samples:  # Use test_samples instead of stft_data.skip(14)
            X_test.append(spec.numpy())
            y_test.append(label.numpy())

        if X_test and y_test:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            print(f"\nEvaluating on {len(X_test)} test samples...")
            y_hat = model.predict(X_test)
            y_hat = [1 if prediction > 0.5 else 0 for prediction in y_hat]
            print(f"Predictions: {y_hat}")
            print(f"True labels: {y_test}")
        else:
            print("No test data available")
    except Exception as e:
        print(f"Error during evaluation: {e}")

