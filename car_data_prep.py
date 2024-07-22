import pandas as pd
import numpy as np 
import re
import requests
import json
import pickle
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from scipy.stats.mstats import winsorize
from scipy import stats

def prepare_data(dataframe, encoder_path='encoder.pkl', scaler_path='scaler.pkl', fit=False):
    # Drop Duplicate Rows
    dataframe.drop_duplicates(keep='first', inplace=True)
    
    # Dropping Irrelevant Columns
    columns_to_drop = ['Area', 'Pic_num', 'Cre_date', 'Repub_date', 'Test']
    dataframe.drop(columns=columns_to_drop, inplace=True)
    
    # Cleaning and Standardizing Model Names
    def clean_model_column(row):
        manufactor = row['manufactor']
        model = row['model']
        model = re.sub(r'\b' + re.escape(manufactor) + r'\b', '', model, flags=re.IGNORECASE)
        model = re.sub(r'\(\d{4}\)', '', model)
        model = re.sub(r'\b\d{4}\b', '', model)
        model = re.sub(r',', '', model)
        return model.strip()
    dataframe['model'] = dataframe.apply(clean_model_column, axis=1)
    
    # Additional Model Cleaning
    dataframe['model'] = dataframe['model'].str.replace('לקסוס', '', regex=False).str.strip()
    dataframe['capacity_Engine'] = dataframe['capacity_Engine'].str.replace(',', '', regex=True)
    dataframe['Km'] = dataframe['Km'].str.replace(',', '', regex=True)
    
    # Normalizing Values
    replacements = {
        'manufactor': {'Lexsus': 'לקסוס'},
        'Engine_type': {'היברידי': 'היבריד'},
        'Gear': {'אוטומט': 'אוטומטית'},
        'model': {
            'סיוויק הייבריד': 'סיוויק',
            'סוויפט החדשה': 'סויפט',
            'CIVIC': 'סיוויק',
            'ACCORD': 'אקורד',
            'פלואנס חשמלי': 'פלואנס'
        },
        'City': {
            'Tel aviv': 'תל אביב',
            'Rishon LeTsiyon': 'ראשון לציון',
            'Tzur Natan': 'צור נתן',
            'jeruslem': 'ירושלים',
            'Rehovot': 'רחובות',
            'ashdod': 'אשדוד'
        }
    }
    for column, replace_dict in replacements.items():
        dataframe[column] = dataframe[column].replace(replace_dict)
    
    # Replace "לא מוגדר" with NaN 
    dataframe.replace('לא מוגדר', np.nan, inplace=True)
    
    # Handling Electric Vehicles
    dataframe.loc[dataframe['Engine_type'] == 'חשמלי', 'capacity_Engine'] = 0
    
    # Validate and clean numeric inputs
    for col in ['Year', 'Hand', 'capacity_Engine', 'Km']:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    
    # Filling Missing Values with Mode for Categorical Features
    def fill_missing_with_mode(df, columns):
        for col in columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        return df
    
    columns_to_fill = ['Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Color']
    dataframe = fill_missing_with_mode(dataframe, columns_to_fill)
    
    # Filling Missing 'Km' with Median
    def fill_missing_with_median(df):
        df['Km'] = df['Km'].replace('None', np.nan)
        df['Km'].fillna(df['Km'].median(), inplace=True)
        return df
    dataframe = fill_missing_with_median(dataframe)
    
    # Fetching and Filtering API Data
    url = 'https://data.gov.il/api/3/action/datastore_search?resource_id=5e87a7a1-2f6f-41c1-8aec-7216d52a6cf6'
    web = requests.get(url)
    dict_data = json.loads(web.text)
    
    api_list = []
    for number in range(len(dict_data["result"]["records"])):
        tozar = dict_data["result"]["records"][number]["tozar"]
        degem = dict_data["result"]["records"][number]["kinuy_mishari"]
        shnaton = dict_data["result"]["records"][number]["shnat_yitzur"]
        dictio = {'tozar': tozar, 'degem': degem, 'shnaton': shnaton}
        api_list.append(dictio)
    car_gov_data = pd.DataFrame(api_list)
    car_gov_data = car_gov_data.rename(columns={'tozar': 'manufactor', 'degem': 'model', 'shnaton': 'Year'})
    
    replace_dict = {'מזדה': 'מאזדה', 'ב מ וו': 'ב.מ.וו', 'וולבו': 'וולוו'}
    car_gov_data['manufactor'] = car_gov_data['manufactor'].replace(replace_dict)
    
    to_keep = ['סובארו', 'הונדה', 'יונדאי', "פיג'ו", 'רנו', 'סקודה', 'קיה',
               'אופל', 'פולקסווגן', 'מיצובישי', 'פורד', 'מאזדה', 'סיטרואן',
               'ניסאן', 'אאודי', 'טויוטה', 'ב.מ.וו', 'סוזוקי', 'שברולט', 'מרצדס',
               'לקסוס', 'דייהטסו', 'מיני', 'קרייזלר', 'אלפא רומיאו', 'וולוו']
    filtered_api_data = car_gov_data[car_gov_data['manufactor'].isin(to_keep)]
    
    def replace_model_name(df, filtered_api_data):
        manufacturers = df['manufactor'].unique()
        for manufacturer in manufacturers:
            model_api = filtered_api_data[filtered_api_data['manufactor'] == manufacturer]
            model_df = df[df['manufactor'] == manufacturer]
        
            model_replacements = {df_model: api_model for df_model in model_df['model'].unique() 
                                  for api_model in model_api['model'].unique()
                                  if df_model.lower() == api_model.lower()}
            df.loc[df['manufactor'] == manufacturer, 'model'] = \
                df[df['manufactor'] == manufacturer]['model'].replace(model_replacements)
    
        return df
    dataframe = replace_model_name(dataframe, filtered_api_data)
    
    # Calculating Supply Score
    def supply_score(df, model_api):
        similar_rows_count = model_api.groupby(['manufactor', 'model', 'Year']).size().reset_index(name='supply')
        count_dict = {(row['manufactor'], row['Year'], row['model']): row['supply'] for idx, row in similar_rows_count.iterrows()}
        def get_supply(row):
            return count_dict.get((row['manufactor'], row['Year'], row['model']), 0)
        df['Supply_score'] = df.apply(get_supply, axis=1)
    supply_score(dataframe, filtered_api_data)
    
    # Creating 'No_accident' Column
    pattern = r'(שמור(?:ה)?|בלי תאונות|ללא תאונות|לא תאונות|לא עברה תאונות|לא עבר תאונות|שמיר)'
    def check_keyword_in_description(description):
        return 1 if re.search(pattern, description) else 0
    dataframe['No_accident'] = dataframe['Description'].apply(lambda x: check_keyword_in_description(str(x)))
    dataframe.drop(columns=['Description'], inplace=True)
    
    # Converting Data Types
    dataframe['Year'] = dataframe['Year'].astype(int, errors='ignore')
    dataframe['Hand'] = dataframe['Hand'].astype(int, errors='ignore')
    dataframe['Gear'] = dataframe['Gear'].astype('category')
    dataframe['capacity_Engine'] = dataframe['capacity_Engine'].astype(int, errors='ignore')
    dataframe['Engine_type'] = dataframe['Engine_type'].astype('category')
    dataframe['Prev_ownership'] = dataframe['Prev_ownership'].astype('category')
    dataframe['Curr_ownership'] = dataframe['Curr_ownership'].astype('category')
    dataframe['City'] = dataframe['City'].astype('string')
    dataframe['Color'] = dataframe['Color'].astype('string')
    dataframe['Km'] = dataframe['Km'].astype(int, errors='ignore')
    dataframe['Supply_score'] = dataframe['Supply_score'].astype(int)
    dataframe['No_accident'] = dataframe['No_accident'].astype(bool)
    
    # Winsorize the data
    columns_to_winsorize = ['capacity_Engine', 'Km', 'Hand', 'Year']
    for col in columns_to_winsorize:
        dataframe[col] = winsorize(dataframe[col], limits=[0.05, 0.05])
    
    # Remove outliers using IQR method
    def remove_outliers_iqr(df, columns):
        Q1 = df[columns].quantile(0.25)
        Q3 = df[columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_entries = ((df[columns] >= lower_bound) & (df[columns] <= upper_bound)).all(axis=1)
        return df[filtered_entries]
    dataframe = remove_outliers_iqr(dataframe, columns_to_winsorize)
    
    # One-hot encode remaining categorical features
    categorical_features = ['manufactor', 'model', 'City', 'Color', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'No_accident']

    if fit:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = encoder.fit_transform(dataframe[categorical_features])
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
    else:
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        encoded_features = encoder.transform(dataframe[categorical_features])
    
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    dataframe = dataframe.drop(categorical_features, axis=1).reset_index(drop=True)
    dataframe = pd.concat([dataframe, encoded_df], axis=1)
    
    # Scale numerical features
    numerical_features = ['Year', 'Hand', 'capacity_Engine', 'Km', 'Supply_score']
    print(dataframe['Year'])
    if fit:
        scaler = RobustScaler()
        dataframe[numerical_features] = scaler.fit_transform(dataframe[numerical_features])
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        dataframe[numerical_features] = scaler.transform(dataframe[numerical_features])
    
    return dataframe
