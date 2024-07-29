import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('building_occupancy_data.csv')

# Feature engineering
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Prepare the models dictionary
models = {}
days_of_week = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

# Train a separate model for each day of the week
for day in days_of_week.keys():
    # Filter data for the specific day of the week
    day_df = df[df['day_of_week'] == day]

    X_day = day_df[['hour', 'is_weekend', 'is_holiday']]
    y_day = day_df['occupancy']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_day, y_day, test_size=0.2, random_state=42)
    
    # Train the model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Store the trained model
    models[day] = rf
    
    # Predict and evaluate
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{days_of_week[day]} Model - Mean Squared Error: {mse}')

# Forecast future values for the next Monday (as an example)
next_day = (df['datetime'].iloc[-1] + pd.Timedelta(days=1)).date()
next_day_of_week = next_day.weekday()
print(f"Next day's weekday: {days_of_week[next_day_of_week]}")

# Prepare data for the forecast
future_hours = pd.DataFrame({
    'hour': list(range(0, 24)),
    'is_weekend': [1 if next_day_of_week in [5, 6] else 0] * 24,
    'is_holiday': [0] * 24  # Adjust based on your data for holidays
})

# Select the model for the next day's weekday
selected_model = models[next_day_of_week]
future_predictions = selected_model.predict(future_hours)

# Print the forecast
print(future_predictions)

# Plot the forecast
sns.set(style="whitegrid")
plt.figure(figsize=(14, 7))

plt.plot(future_hours['hour'], future_predictions, marker='o', linestyle='-', color='seagreen', markersize=8)

plt.title(f'Predicted Building Occupancy for Next {days_of_week[next_day_of_week]}', fontsize=18, weight='bold')
plt.xlabel('Hour of the Day', fontsize=14)
plt.ylabel('Predicted Occupancy', fontsize=14)

plt.xticks(range(0, 24), fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Adding annotations closer to the points
for hour, pred in enumerate(future_predictions):
    plt.annotate(f'{pred:.0f}', xy=(hour, pred), xytext=(0, 3),
                 textcoords='offset points', ha='center', fontsize=10, color='black')

# Show the plot
plt.show()
