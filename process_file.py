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
        formatted_address = ''
    text = f"{type_info} {coordinates} {formatted_address}"
    return text


def generate_langgraph_embedding_text(text):
    return embeddings_model.embed_query(text)


def flatten_array(x):
    if type(x) is list:
        return ', '.join(x)
    else:
        return ''


df = read_json("Output.json")

df['title_embeddings'] = df['title'].apply(generate_langgraph_embedding_text)
df['alternate_titles_flat'] = df['alternate_titles'].apply(flatten_array)
df['alternate_titles_embeddings'] = df['alternate_titles_flat'].apply(generate_langgraph_embedding_text)
df['description'] = df['description'].apply(edit_description)
df['description_embeddings'] = df['description'].apply(generate_langgraph_embedding_text)
df['geo_info'] = df['geo'].apply(extract_geo_info)
df['geo_embeddings'] = df['geo_info'].apply(generate_langgraph_embedding_text)
df['category_embeddings'] = df['category'].apply(generate_langgraph_embedding_text)
df['labels_flat'] = df['labels'].apply(flatten_array)
df['labels_embeddings'] = df['labels_flat'].apply(generate_langgraph_embedding_text)
df['phq_attendance_str'] = df['phq_attendance'].apply(lambda x: f'{x}')
df['phq_attendance_embeddings'] = df['phq_attendance_str'].apply(generate_langgraph_embedding_text)

df.to_json('for_collection.json', orient='records', lines=True)
