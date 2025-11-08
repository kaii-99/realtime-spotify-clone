import requests
import pandas as pd

# Fetch from your backend API
url = "http://localhost:5000/api/export/allsong"
response = requests.get(url)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data)
print(df.head())

# Save locally for model training
df.to_csv("listening_data.csv", index=False)
print("✅ Data saved to listening_data.csv")
