import requests

API_KEY = "af766ad1aced6c38b4b19efe57a2a85e"
CITY = "Singapore" 

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code != 200:
        print("Error fetching weather data:", data.get("message", "Unknown error"))
        return None
    
    weather_main = data["weather"][0]["main"]
    temperature = data["main"]["temp"]
    print(f"🌤 Weather: {weather_main}, 🌡 Temp: {temperature}°C")
    return weather_main, temperature

def weather_to_mood(weather):
    mapping = {
        "Clear": "happy",
        "Clouds": "chill",
        "Rain": "sad",
        "Thunderstorm": "intense",
        "Drizzle": "romantic",
        "Snow": "peaceful",
        "Mist": "relaxed",
        "Fog": "relaxed"
    }
    return mapping.get(weather, "neutral")

def mood_to_genres(mood):
    mapping = {
        "happy": ["Pop", "Dance"],
        "chill": ["Lo-Fi", "Jazz"],
        "sad": ["Ballad", "Acoustic"],
        "intense": ["Rock", "Metal"],
        "romantic": ["R&B", "Soul"],
        "peaceful": ["Classical", "Instrumental"],
        "relaxed": ["Lo-Fi", "Ambient"],
        "neutral": ["Pop", "Indie"]
    }
    return mapping.get(mood, ["Pop"])

def detect_mood_from_weather(city):
    weather_data = get_weather(city)
    if not weather_data:
        return None
    
    weather_main, _ = weather_data
    mood = weather_to_mood(weather_main)
    genres = mood_to_genres(mood)
    
    return {
        "weather": weather_main,
        "mood": mood,
        "recommended_genres": genres
    }

if __name__ == "__main__":
    result = detect_mood_from_weather(CITY)
    print(result)
