import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense

def build_model(num_users, num_courses, embedding_size=16):
    user_input = Input(shape=(1,))
    course_input = Input(shape=(1,))

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
    course_embedding = Embedding(input_dim=num_courses, output_dim=embedding_size)(course_input)

    user_vec = Flatten()(user_embedding)
    course_vec = Flatten()(course_embedding)

    merged = Concatenate()([user_vec, course_vec])
    x = Dense(128, activation='relu')(merged)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)

    model = Model(inputs=[user_input, course_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model
