import tensorflow as tf
import numpy as np

# Dummy input: batch_size=1, sequence_length=10
sample_input = tf.random.uniform((1, 10), dtype=tf.int32, minval=0, maxval=5000)

# Initialize encoder and decoder
vocab_inp_size = 5000
vocab_tar_size = 5000
embedding_dim = 256
units = 512

encoder = Encoder(vocab_inp_size, embedding_dim, units)
decoder = Decoder(vocab_tar_size, embedding_dim, units)

# Run encoder
enc_output, enc_hidden_h, enc_hidden_c = encoder(sample_input)
print("Encoder output shape:", enc_output.shape)  # (1, 10, 512)

# Prepare decoder input (start token)
start_token = tf.constant([[1]])  # shape: (1, 1)

# Run decoder
decoder_output, dec_hidden_h, dec_hidden_c, attention_weights = decoder(
    start_token, enc_hidden_h, enc_output
)

print("Decoder output shape:", decoder_output.shape)         # (1, 5000)
print("Attention weights shape:", attention_weights.shape)   # (1, 10, 1)
<img width="312" height="69" alt="Screenshot 2025-10-08 115242" src="https://github.com/user-attachments/assets/0ee852c2-681e-4112-b80a-21848e5cfd00" />
