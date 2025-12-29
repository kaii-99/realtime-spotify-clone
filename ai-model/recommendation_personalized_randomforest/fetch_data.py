import requests
import pandas as pd
import os

# testuser1 -> 690f61b5822864ba799ef31d
# testuser2 -> 690f6586822864ba799ef3a3
# testuser3 -> 690f66bf822864ba799ef3fc

USER_ID = "690f6586822864ba799ef3a3"

url = f"http://localhost:5000/api/export/listenhistory_mood/{USER_ID}"

response = requests.get(url)

if response.status_code != 200:
    print("Error:", response.text)
    exit()

data = response.json()

if len(data) == 0:
    print("No data returned for this user.")
    exit()

df = pd.DataFrame(data)
print(df.head())

# ✅ Create folder if not exists
folder_name = "listening_history_data"
os.makedirs(folder_name, exist_ok=True)

# ✅ Save inside the new folder
file_path = os.path.join(folder_name, f"{USER_ID}_listening_data.csv")
df.to_csv(file_path, index=False)

print(f"✅ Data saved to {file_path}")
