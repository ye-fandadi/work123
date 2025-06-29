import requests
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class WeatherAPI:
    def __init__(self):
        self.api_key = config.get("weather_api_key")

    def get_weather(self, city: str):
        url = f"http://api.weatherapi.com/v1/current.json?key={self.api_key}&q={city}&aqi=no"
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            condition = data['current']['condition']['text']
            temp_c = data['current']['temp_c']
            return f"{city}现在的天气是{condition}，气温{temp_c}℃。"
        else:
            return "无法获取天气信息。"

