import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Flatten, Dense, Concatenate
from keras.callbacks import EarlyStopping
import joblib

# ------------------ Load & Encode -------------------
df = pd.read_csv('../dataset/RAW_interactions.csv')

user_to_index = {user: idx for idx, user in enumerate(df['user_id'].unique())}
recipe_to_index = {recipe: idx for idx, recipe in enumerate(df['recipe_id'].unique())}
df['user_idx'] = df['user_id'].map(user_to_index)
df['recipe_idx'] = df['recipe_id'].map(recipe_to_index)
df['rating'] = df['rating'] / 5.0  # Normalize

n_users = len(user_to_index)
n_recipes = len(recipe_to_index)

# ------------------ Train-Test Split -------------------
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# ------------------ Model Inputs -------------------
user_input = Input(shape=(1,))
recipe_input = Input(shape=(1,))

embedding_size = 50

user_embedding = Embedding(n_users, embedding_size)(user_input)
recipe_embedding = Embedding(n_recipes, embedding_size)(recipe_input)

# Combine
user_vec = Flatten()(user_embedding)
recipe_vec = Flatten()(recipe_embedding)
merged = Concatenate()([user_vec, recipe_vec])

# Predict rating
dense1 = Dense(64, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(dense1)

model = Model(inputs=[user_input, recipe_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# ------------------ Train -------------------
model.fit(
    [train_df['user_idx'], train_df['recipe_idx']], train_df['rating'],
    validation_data=([test_df['user_idx'], test_df['recipe_idx']], test_df['rating']),
    epochs=10, batch_size=512,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=1
)

# ------------------ Save -------------------
model.save('fast_rec_model.h5')
joblib.dump(user_to_index, 'user_to_index.pkl')
joblib.dump(recipe_to_index, 'recipe_to_index.pkl')

print("✅ Fast training complete.")
