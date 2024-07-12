from pandas import read_json

PREDICT_HQ_DESCRIPTION = 'Sourced from predicthq.com'
DESCRIPTION_PREFIX = f'{PREDICT_HQ_DESCRIPTION} - '
prefix_len = len(DESCRIPTION_PREFIX)


def edit_description(description):
    if description == PREDICT_HQ_DESCRIPTION:
        return ''
    elif description.startswith(DESCRIPTION_PREFIX):
        return description[prefix_len:]
    else:
        return description


def extract_geo_info(geo):
    type_info = geo['geometry']['type']
    coordinates = geo['geometry']['coordinates']
    formatted_address = geo['address']['formatted_address']
    text = f"{type_info} {coordinates} {formatted_address}"
    return text


def generate_langgraph_embedding(text):
    # This should be replaced with the actual code to generate embeddings using LangGraph
    return [0.0, 0.1, 0.2]


df = read_json("Output.json")

# Filter out rows where 'private' is True
df_filtered = df[df['private'] != True]

df_filtered['description'] = df_filtered['description'].apply(edit_description)
df_filtered['geo_info'] = df_filtered['geo'].apply(extract_geo_info)
df_filtered['geo_embeddings'] = df_filtered['geo_info'].apply(generate_langgraph_embedding)


df_filtered.to_json('for_collection.json', orient='records', lines=True)
