import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

# Load models and data
scaler = joblib.load('../CB_with_clustering/scaler_clustered.pkl')
cf_model = tf.keras.models.load_model('../New_CF/embedding_cf_model.h5', compile=False)
cf_model.compile(optimizer='adam', loss='mse')

user_to_index = joblib.load('../New_CF/fast_user_to_index.pkl')
recipe_to_index = joblib.load('../New_CF/fast_recipe_to_index.pkl')
index_to_recipe = {v: k for k, v in recipe_to_index.items()}

df = pd.read_csv('../CB_with_clustering/final_meal_dataset.csv')
interactions_df = pd.read_csv('../dataset/RAW_interactions.csv')
raw_df = pd.read_csv('../dataset/RAW_recipes.csv')
raw_df = raw_df[['id', 'ingredients', 'steps']]

nutrition_cols = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbs']

goal_weights = {
    "gain_muscle": np.array([2, 1, 0.5, 0.5, 3, 1, 1]),
    "lose_weight": np.array([2, 2, 1.5, 2, 2, 1.5, 1]),
    "healthy_eating": np.array([1, 1, 1, 1, 1, 1, 1]),
    "gain_weight": np.array([3, 2, 1, 1, 2, 1, 1])
}

category_ratios = {
    'breakfast': 0.20, 'main': 0.35, 'side': 0.15,
    'snack': 0.10, 'dessert': 0.10, 'drink': 0.05, 'other': 0.05
}

def calculate_bmr(weight, height, age, gender):
    return 10 * weight + 6.25 * height - 5 * age + (5 if gender.lower() == "male" else -161)

def adjust_for_activity(bmr, activity_level):
    return bmr * {
        "sedentary": 1.2, "light": 1.375, "moderate": 1.55,
        "active": 1.725, "super_active": 1.9
    }.get(activity_level, 1.2)

def adjust_for_goal(calories, goal):
    return calories + {
        "lose_weight": -500, "gain_muscle": 250, "gain_weight": 500, "healthy_eating": 0
    }.get(goal, 0)

def get_daily_target(user):
    bmr = calculate_bmr(user["weight"], user["height"], user["age"], user["gender"])
    tdee = adjust_for_activity(bmr, user["activity_level"])
    adjusted_calories = adjust_for_goal(tdee, user["goal"])
    macro_ratios = {
        "gain_muscle": {"protein": 0.60, "carbs": 0.20, "fat": 0.20},
        "lose_weight": {"protein": 0.35, "carbs": 0.30, "fat": 0.35},
        "healthy_eating": {"protein": 0.20, "carbs": 0.50, "fat": 0.30},
        "gain_weight": {"protein": 0.25, "carbs": 0.45, "fat": 0.30}
    }
    extras = {
        "gain_muscle": {"sugar": 8, "sodium": 350, "saturated_fat": 4},
        "lose_weight": {"sugar": 3, "sodium": 250, "saturated_fat": 2},
        "healthy_eating": {"sugar": 5, "sodium": 300, "saturated_fat": 3},
        "gain_weight": {"sugar": 10, "sodium": 350, "saturated_fat": 4}
    }
    ratio = macro_ratios[user["goal"]]
    extra = extras[user["goal"]]
    return {
        "calories": adjusted_calories,
        "protein": (adjusted_calories * ratio["protein"]) / 4,
        "carbs": (adjusted_calories * ratio["carbs"]) / 4,
        "total_fat": (adjusted_calories * ratio["fat"]) / 9,
        "sugar": extra["sugar"] * 4,
        "sodium": extra["sodium"] * 4,
        "saturated_fat": extra["saturated_fat"] * 4
    }

def contains_excluded_ingredient(recipe_id):
    entry = raw_df[raw_df['id'] == recipe_id]
    if entry.empty:
        return False
    ingredients = eval(entry.iloc[0]['ingredients'])  # list from string
    return any(bad in ing.lower() for ing in ingredients for bad in excluded_list)

# -------- Streamlit UI --------
st.title("🥗 Smart Meal Recommender")

with st.form("user_info"):
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["male", "female"])
    activity_level = st.selectbox("Activity Level", ["sedentary", "light", "moderate", "active", "super_active"])
    goal = st.selectbox("Health Goal", ["gain_muscle", "lose_weight", "healthy_eating", "gain_weight"])
    excluded_ingredients = st.text_input("Excluded Ingredients (comma-separated)", placeholder="e.g. carrot, cheese")
    excluded_list = [x.strip().lower() for x in excluded_ingredients.split(",") if x.strip()]
    submitted = st.form_submit_button("Get My Meal Plan")

if submitted:
    user_info = {
        "weight": weight, "height": height, "age": age,
        "gender": gender, "activity_level": activity_level, "goal": goal
    }

    daily_target = get_daily_target(user_info)
    goal_weight = goal_weights[goal]
    user_id = interactions_df['user_id'].value_counts().index[3]  # Change logic if needed
    user_idx = user_to_index[user_id]

    st.subheader("📊 Daily Nutritional Target")
    st.json(daily_target)

    st.subheader("🍽️ Recommendations")
    for category in df['category'].unique():
        df_cat = df[df['category'] == category].reset_index(drop=True)
        df_cat = df_cat[~df_cat['id'].apply(contains_excluded_ingredient)].reset_index(drop=True)
        if df_cat.empty:
            continue

        ratio = category_ratios.get(category, 0.05)
        target = {k: v * ratio for k, v in daily_target.items()}
        query_scaled = scaler.transform(pd.DataFrame([target])[nutrition_cols])
        X_scaled = scaler.transform(df_cat[nutrition_cols])

        knn = NearestNeighbors(n_neighbors=min(20, len(df_cat)))
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(query_scaled)

        cb_candidates = df_cat.iloc[indices[0]].copy()
        cb_candidates['cb_distance'] = distances[0]

        cf_scores = []
        for rid in cb_candidates['id']:
            if rid in recipe_to_index:
                r_idx = recipe_to_index[rid]
                pred = cf_model.predict([np.array([user_idx]), np.array([r_idx])], verbose=0)[0][0]
                cf_scores.append(pred)
            else:
                cf_scores.append(0.0)
        cb_candidates['cf_score'] = cf_scores

        top_cb = cb_candidates.sort_values(by='cb_distance').head(10)
        top_cf_reranked = top_cb.sort_values(by='cf_score', ascending=False).head(5)

        if not top_cf_reranked.empty:
            selected = top_cf_reranked.sample(1).iloc[0]
            st.markdown(f"### 🔹 {category.title()}")
            st.write(f"📌 **{selected['name']}**")
            st.write(f"⚖️ MSE: `{selected['cb_distance']:.4f}` | 🌟 CF Score: `{selected['cf_score']*5:.2f}`")
            with st.expander(f"📌 {selected['name']} (click to view instructions)"):
                recipe_id = selected['id']
                row = raw_df[raw_df['id'] == recipe_id]
                if not row.empty:
                    steps = eval(row.iloc[0]['steps'])
                    for i, step in enumerate(steps):
                        st.markdown(f"Step {i+1}: {step}")
                else:
                    st.warning("No instructions found.")
