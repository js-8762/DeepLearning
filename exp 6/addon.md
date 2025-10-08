from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Dropout, Activation

# Example constants
max_len = 100       # Maximum sequence length
n_words = 5000      # Vocabulary size
n_tags = 17         # Number of unique tags (classes)

# Input
input = Input(shape=(max_len,))

# Embedding layer
model = Embedding(input_dim=n_words, output_dim=64, input_length=max_len)(input)

# BiLSTM layer
model = Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1))(model)

# Optional dropout
model = Dropout(0.1)(model)

# TimeDistributed Dense layer with softmax for classification
output = TimeDistributed(Dense(n_tags, activation="softmax"))(model)

# Build model
model = Model(inputs=input, outputs=output)

# Compile model with categorical crossentropy
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Model summary (optional)
model.summary()
<img width="1071" height="312" alt="Screenshot 2025-10-08 110022" src="https://github.com/user-attachments/assets/1465761f-64cc-4b4f-992c-74c0f2544d6b" />
