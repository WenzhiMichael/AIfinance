import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# 1) Fetch a Recent Time Window
# -------------------------------
ticker = "AAPL"
start_date = "2022-01-01"
end_date = "2025-01-01"  # or today's date for the most recent data

df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    raise ValueError("No data fetched. Check your ticker symbol or date range.")

# We'll keep only the 'Close' price
df = df[["Close"]].copy()
df.reset_index(inplace=True)


# -------------------------------
# 2) Fetch and Compute News Sentiment for Each Day
#    (Example: We'll just fetch the *current* sentiment and apply it,
#     but in production you might gather daily news and match dates.)
# -------------------------------
def get_news_sentiment():
    """
    Fetches the latest business headlines using NewsAPI,
    processes them with FinBERT to compute an average sentiment score.
    Maps: Positive => +score, Negative => -score, Neutral => 0.
    Returns a single aggregated sentiment score.
    """
    news_api_key = "b46a45e14c1a4277a846d352721640d9"  # Replace with your actual NewsAPI key
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={news_api_key}"

    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        raise Exception("Error fetching news: " + data.get("message", "Unknown error"))

    articles = data.get("articles", [])
    if not articles:
        return 0.0  # No news => neutral sentiment

    # Initialize the FinBERT sentiment pipeline
    model_name = "ProsusAI/finbert"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

    total_score = 0.0
    count = 0

    for article in articles:
        headline = article.get("title")
        if headline:
            sentiment = sentiment_pipeline(headline)[0]
            label = sentiment["label"].lower()
            score = sentiment["score"]
            # Map sentiment to numeric value
            if label == "positive":
                total_score += score
            elif label == "negative":
                total_score -= score
            # neutral => 0
            count += 1

    return total_score / count if count else 0.0


# For demonstration: get today's aggregated sentiment and repeat for all days
sentiment_score = get_news_sentiment()
df["Sentiment"] = sentiment_score

# -------------------------------
# 3) Time-based Split (Train vs. Test)
#    We'll split at 2023-07-01 for demonstration
# -------------------------------
split_date = "2023-07-01"
train_df = df[df["Date"] < split_date].copy()
test_df = df[df["Date"] >= split_date].copy()

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError("Check your split_date or data range; got empty train/test set.")

# -------------------------------
# 4) Scale the Data
# -------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))

# We'll combine train/test to fit the scaler once,
# then transform separately to avoid data leakage.
full_data = pd.concat([train_df, test_df]).reset_index(drop=True)

# We have 2 features: [Close, Sentiment]
scaled_full = scaler.fit_transform(full_data[["Close", "Sentiment"]])
scaled_full_df = pd.DataFrame(scaled_full, columns=["Close", "Sentiment"])

# Re-split after scaling
scaled_train = scaled_full_df.iloc[: len(train_df)]
scaled_test = scaled_full_df.iloc[len(train_df):]

# -------------------------------
# 5) Create Time-Series Sequences
#    e.g., a 60-day window to predict the next day's close
# -------------------------------
sequence_length = 60


def create_sequences(scaled_df, seq_length):
    x, y = [], []
    data_array = scaled_df.values
    for i in range(seq_length, len(data_array)):
        # i-th row is the "today" that we want to predict
        x.append(data_array[i - seq_length: i])  # previous 60 rows
        y.append(data_array[i, 0])  # "Close" is 0th col
    return np.array(x), np.array(y)


train_x, train_y = create_sequences(scaled_train, sequence_length)
test_x, test_y = create_sequences(scaled_test, sequence_length)

print("Train X shape:", train_x.shape, "Train Y shape:", train_y.shape)
print("Test  X shape:", test_x.shape, "Test  Y shape:", test_y.shape)

# -------------------------------
# 6) Build a GRU Model
# -------------------------------
from tensorflow.keras import backend as K

K.clear_session()  # Just to be safe if you're running multiple times

inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
gru_out = GRU(50, return_sequences=False)(inputs)
dense1 = Dense(25, activation='relu')(gru_out)
outputs = Dense(1)(dense1)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# -------------------------------
# 7) Train with Early Stopping
# -------------------------------
es = EarlyStopping(monitor="val_loss", mode="min", patience=3, restore_best_weights=True)

history = model.fit(
    train_x,
    train_y,
    validation_split=0.2,  # use 20% of the train set for validation
    epochs=50,  # up to 50 epochs
    batch_size=32,
    callbacks=[es],
    verbose=1,
)

# -------------------------------
# 8) Evaluate on the Test Set
# -------------------------------
test_preds_scaled = model.predict(test_x)
# We need to inverse transform only the 'Close' column
# We'll create a dummy array with 2 columns, because the scaler expects 2 features
dummy = np.zeros((test_preds_scaled.shape[0], 2))
dummy[:, 0] = test_preds_scaled[:, 0]
test_preds = scaler.inverse_transform(dummy)[:, 0]  # unscale the "Close"

# Ground truth (unscale)
dummy_test = np.zeros((test_y.shape[0], 2))
dummy_test[:, 0] = test_y
test_true = scaler.inverse_transform(dummy_test)[:, 0]

# Let's see how well we did on the final portion
df_results = pd.DataFrame({
    "Date": test_df["Date"].iloc[sequence_length:].values,
    "Actual_Close": test_true,
    "Predicted_Close": test_preds
})
print(df_results.tail())

# -------------------------------
# 9) Predict the Most Recent Day (Final Value)
# -------------------------------
# Use the last 60 days from your test set to predict the next day.
last_60 = scaled_test.values[-sequence_length:]  # shape = (60, 2)
last_60 = last_60.reshape((1, sequence_length, 2))

final_pred_scaled = model.predict(last_60)
dummy_final = np.zeros((1, 2))
dummy_final[:, 0] = final_pred_scaled[0]
final_pred_price = scaler.inverse_transform(dummy_final)[:, 0][0]

print("Predicted Next-Day Closing Price:", final_pred_price)
