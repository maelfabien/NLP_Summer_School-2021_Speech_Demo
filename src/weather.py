import requests, json
import streamlit as st

with open("src/key.txt", "r") as f:
    api_key = f.readlines()[0]
    
def find_temperature(city_name):

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]
        z = x["weather"]

        current_temperature = y["temp"] - 273.15 # convert kelvin to celsius
        #current_pressure = y["pressure"]
        #current_humidiy = y["humidity"]
        weather_description = z[0]["description"]
      
        return current_temperature, weather_description
      
    else:
        return "0", "None"



