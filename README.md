To fetch data:
```python
import requests

response = requests.get(
    url="https://api.predicthq.com/v1/events",
    headers={
        "Authorization": "Bearer access_token",
        "Accept": "text/csv"
    },
    params={
        "category": "concerts,festivals,performing-arts,sports",
        "country": "US",
        "label": "art,community,concert,entertainment,family,festival,holiday-hebrew,music"
    }
)

print(response.json())
```