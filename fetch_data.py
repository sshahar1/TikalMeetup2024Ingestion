import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

access_token = os.getenv("PREDICTHQ_ACCESS_TOKEN")
offset = 0
results = []

while True:
    response = requests.get(
        url="https://api.predicthq.com/v1/events",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        params={
            "category": "concerts,festivals,performing-arts,sports",
            "country": "US",
            "label": "art,community,concert,entertainment,family,festival,holiday-hebrew,music",
            "limit": 50,
            "offset": offset,
        }
    )

    content = json.loads(response.content.decode("utf-8"))
    results.extend(content['results'])
    print(len(results))
    if content['next'] is None:
        result_string = json.dumps(results)
        text_file = open("Output.json", "w")

        text_file.write(result_string)

        text_file.close()
        exit(0)
    else:
        offset += 50
