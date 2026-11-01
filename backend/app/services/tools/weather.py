"""Weather tool, backed by Open-Meteo: free, no API key, so it works out of the box.

Two calls: geocoding (city name -> coordinates), then the forecast. The formatting
is a separate pure function, so it can be tested without network access.
"""

import httpx

from app.services.tools import Tool, ToolOutput

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10

# WMO weather codes used by Open-Meteo, grouped into plain descriptions
_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "violent showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


def describe(code: int | None) -> str:
    return _WEATHER_CODES.get(code, "unknown conditions") if code is not None else "unknown"


def format_weather(place: dict, forecast: dict) -> str:
    """Turn Open-Meteo's JSON into a short text the model can read."""
    name = ", ".join(part for part in (place.get("name"), place.get("country")) if part)
    current = forecast.get("current", {})
    lines = [
        f"Weather in {name} (local time {current.get('time', '?')}):",
        f"- now: {current.get('temperature_2m')} °C, feels like "
        f"{current.get('apparent_temperature')} °C, {describe(current.get('weather_code'))}, "
        f"wind {current.get('wind_speed_10m')} km/h, humidity {current.get('relative_humidity_2m')}%",
    ]
    daily = forecast.get("daily", {})
    days = daily.get("time", [])
    for i, day in enumerate(days):
        lines.append(
            f"- {day}: {daily['temperature_2m_min'][i]} to {daily['temperature_2m_max'][i]} °C, "
            f"{describe(daily['weather_code'][i])}, "
            f"rain probability {daily['precipitation_probability_max'][i]}%"
        )
    return "\n".join(lines)


def _run(arguments: dict) -> ToolOutput:
    city = str(arguments.get("city", "")).strip()
    if not city:
        raise ValueError("the 'city' argument is required")

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        geo = client.get(
            GEOCODING_URL, params={"name": city, "count": 1, "language": "en", "format": "json"}
        )
        geo.raise_for_status()
        results = geo.json().get("results") or []
        if not results:
            raise ValueError(f"no city found for '{city}'")
        place = results[0]

        response = client.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,weather_code",
                "timezone": "auto",
                "forecast_days": 3,
            },
        )
        response.raise_for_status()

    return ToolOutput(content=format_weather(place, response.json()))


weather_tool = Tool(
    name="weather",
    description=(
        "Current weather and a 3-day forecast for a city: temperature, conditions, "
        "wind, rain probability. Use it for any question about weather."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Paris' or 'Lyon, France'"}
        },
        "required": ["city"],
    },
    run=_run,
)
