from langchain_openai import OpenAIEmbeddings
from pandas import read_json

from dotenv import load_dotenv

load_dotenv()

PREDICT_HQ_DESCRIPTION = 'Sourced from predicthq.com'
DESCRIPTION_PREFIX = f'{PREDICT_HQ_DESCRIPTION} - '
prefix_len = len(DESCRIPTION_PREFIX)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


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
    if 'formatted_address' in geo['address']:
        formatted_address = geo['address']['formatted_address']
    else:
        if geo['address'].get('region') is not None:
            formatted_address = f"{geo['address']['region']} "
        else:
            formatted_address = ''
        formatted_address = f"{formatted_address}{geo['address']['country_code']}"
    text = f"{formatted_address}"
    return text


def generate_langgraph_embedding_text(text):
    return embeddings_model.embed_query(text)


def flatten_array(x):
    if type(x) is list:
        return ', '.join(x)
    else:
        return ''


def created_embeddings(row):
    text = f"Geo Info: {row['geo_info']}"

    return embeddings_model.embed_query(text)


df = read_json("Output.json")

df['alternate_titles_flat'] = df['alternate_titles'].apply(flatten_array)
df['description'] = df['description'].apply(edit_description)
df['geo_info'] = df['geo'].apply(extract_geo_info)
df['labels_flat'] = df['labels'].apply(flatten_array)
df['phq_attendance_str'] = df['phq_attendance'].apply(lambda x: f'{x}')

df['embeddings'] = df.apply(created_embeddings, axis=1)

df.to_json('for_collection.json', orient='records', lines=True)
