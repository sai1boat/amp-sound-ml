import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

def load(fname) -> tf.Tensor:
    binary_data = tf.io.read_file(fname)
    wav, sample_rate = tf.audio.decode_wav(binary_data, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def _spectrogram_eager(file_path, label):
    """Eager version of spectrogram processing"""
    wav = load(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], axis=0)
    spectro = tf.signal.stft(wav, frame_length=512, frame_step=256)
    spectro = tf.abs(spectro)
    spectro = tf.cast(spectro, tf.float32)
    spectro = tf.expand_dims(spectro, axis=-1)
    return spectro, label

def spectrogram(file_path, label):
    """Wrapper for spectrogram that handles tf.py_function for file I/O"""
    spectro, label_out = tf.py_function(
        _spectrogram_eager,
        [file_path, label],
        [tf.float32, tf.float32]  # Fixed: returns float32, not complex64
    )
    # Set shapes manually since py_function doesn't infer them
    # With frame_length=512, frame_step=256, and 48000 samples:
    # Time frames = (48000 - 512) // 256 + 1 = 186
    # Frequency bins = 512/2 + 1 = 257
    spectro.set_shape([186, 257, 1])  # Fixed: correct output shape
    label_out.set_shape([])
    return spectro, label_out

def plot(wav):
    plt.plot(wav)

def plot_spectrogram(spec):
    plt.figure(figsize=(30,20))
    plt.imshow(tf.transpose(spec)[0])
    plt.show()