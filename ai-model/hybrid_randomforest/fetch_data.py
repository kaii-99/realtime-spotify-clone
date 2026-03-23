import requests
import pandas as pd
import os

# testuser1 -> 690f61b5822864ba799ef31d
# testuser2 -> 690f6586822864ba799ef3a3
# testuser3 -> 690f66bf822864ba799ef3fc

USER_IDS = [
    "690f61b5822864ba799ef31d",
    "690f6586822864ba799ef3a3",
    "690f66bf822864ba799ef3fc"
]

BASE_URL = "http://localhost:5000/api/export/listenhistory_mood"

# Create folder if not exists
folder_name = "listening_history_data"
os.makedirs(folder_name, exist_ok=True)

for USER_ID in USER_IDS:
    print(f"📥 Fetching data for user: {USER_ID}")

    url = f"{BASE_URL}/{USER_ID}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error for {USER_ID}: {response.text}")
        continue

    data = response.json()

    if len(data) == 0:
        print(f"⚠️ No data returned for {USER_ID}")
        continue

    df = pd.DataFrame(data)

    file_path = os.path.join(
        folder_name,
        f"{USER_ID}_listening_data.csv"
    )
    df.to_csv(file_path, index=False)

    print(f"Saved {len(df)} rows → {file_path}")

print("All users processed.")
