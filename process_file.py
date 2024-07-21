from langchain_openai import OpenAIEmbeddings
from pandas import read_json

from dotenv import load_dotenv

load_dotenv()

PREDICT_HQ_DESCRIPTION = 'Sourced from predicthq.com'
DESCRIPTION_PREFIX = f'{PREDICT_HQ_DESCRIPTION} - '
prefix_len = len(DESCRIPTION_PREFIX)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def edit_description(description):
    """
    Edit event descriptions based on whether there is description other then the predictHq prefix.

    Parameters:
    description (str): The original event description.

    Returns:
    str: The edited event description. If the description matches the predictHq prefix,
         it will be returned as an empty string. If the description starts with the predictHq prefix,
         the prefix will be removed. Otherwise, the original description will be returned as is.
    """
    if description == PREDICT_HQ_DESCRIPTION:
        return ''
    elif description.startswith(DESCRIPTION_PREFIX):
        return description[prefix_len:]
    else:
        return description


def extract_geo_info(geo):
    """
    Extract geographical information from the given event data.

    This function takes a dictionary representing event data and extracts the geographical information
    such as type, coordinates, and formatted address. If the formatted address is not available, it constructs
    it using the available region and country code.

    Parameters:
    geo (dict): A dictionary containing event data with the following structure:
        {
            'geometry': {
                'type': str,
                'coordinates': list
            },
            'address': {
                'formatted_address': str,
                'region': str,
                'country_code': str
            }
        }

    Returns:
    str: A formatted string containing the geographical information.
    """
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


def flatten_array(x):
    """
    Flattens a list into a comma-separated string.

    This function takes an input `x` and checks if it is a list. If `x` is a list,
    it joins all the elements of the list into a single string separated by commas.
    If `x` is not a list, it returns an empty string.

    Parameters:
    x (any): The input to be flattened. It can be a list or any other type.

    Returns:
    str: A comma-separated string representation of the input list. If the input is not a list,
         an empty string is returned.
    """
    if type(x) is list:
        return ', '.join(x)
    else:
        return ''


def created_embeddings(row):
    """
    Create embeddings for the event description using the OpenAI embeddings model.

    Parameters:
    row (pandas Series): A row of the dataframe containing event data.

    Returns:
        list: A list of embeddings for the geo information.
    """
    text = f"Geo Info: {row['geo_info']}"

    return embeddings_model.embed_query(text)


df = read_json("Output.json")

df['alternate_titles_flat'] = df['alternate_titles'].apply(flatten_array)
df['description'] = df['description'].apply(edit_description)
df['geo_info'] = df['geo'].apply(extract_geo_info)
df['labels_flat'] = df['labels'].apply(flatten_array)
df['phq_attendance_str'] = df['phq_attendance'].apply(lambda x: f'{x}')

df['embedding'] = df.apply(created_embeddings, axis=1)

df.to_json('for_collection.json', orient='records', lines=True)
