import requests
import pandas as pd
import os

# testuser1 -> 690f61b5822864ba799ef31d
# testuser2 -> 690f6586822864ba799ef3a3
# testuser3 -> 690f66bf822864ba799ef3fc

#USER_ID = "690f6586822864ba799ef3a3"
#url = f"http://localhost:5000/api/export/listenhistory_mood/{USER_ID}"

#all users
USER_IDS = {
    "testuser1": "690f61b5822864ba799ef31d",
    "testuser2": "690f6586822864ba799ef3a3",
    "testuser3": "690f66bf822864ba799ef3fc"
}

BASE_URL = "http://localhost:5000/api/export/listenhistory_mood"

# Create folder if not exists
folder_name = "listening_history_data"
os.makedirs(folder_name, exist_ok=True)

for username, user_id in USER_IDS.items():
    url = f"{BASE_URL}/{user_id}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching {username}: {response.text}")
        continue

    data = response.json()

    if not data:
        print(f"No data for {username}")
        continue

    df = pd.DataFrame(data)

    # Save inside the new folder
    file_path = os.path.join(folder_name, f"{user_id}_listening_data.csv")
    df.to_csv(file_path, index=False)

    print(f"Saved {user_id} data to {file_path}")