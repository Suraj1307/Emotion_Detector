import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Input, Embedding,
    Bidirectional, LSTM,
    Dense, Dropout,
    Conv1D, MaxPooling1D
)
from tensorflow.keras.models import Model


# ==========================================
# ATTENTION LAYER
# ==========================================
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[-1]

        self.W = self.add_weight(
            shape=(hidden_dim, hidden_dim),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            shape=(hidden_dim,),
            initializer="zeros",
            trainable=True
        )

        self.u = self.add_weight(
            shape=(hidden_dim, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super().build(input_shape)

    def call(self, x):
        score = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1),
            axis=1
        )

        context_vector = attention_weights * x
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector


# ==========================================
# BUILD MODEL
# ==========================================
def build_model(
    vocab_size,
    embedding_dim,
    max_len,
    num_classes,
    embedding_matrix
):

    inputs = Input(shape=(max_len,))

    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True  # 🔥 Unfrozen embeddings
    )(inputs)

    x = Conv1D(64, 3, activation="relu", padding="same")(x)
    x = MaxPooling1D(2)(x)

    x = Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.4)
    )(x)

    x = AttentionLayer()(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
