import os
from tools.download_data import download_data
from tools.preprocess_functions import del_duplicates, del_columns, to_lower_str, mapping_category, del_empty_text, text_gluing, preprocess_text, cleaning_custom_stopwords, balancing_data, del_big_text, category_encoding

import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import aiohttp
from navec import Navec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import shutil


def preprocess_data(file_path):
    tqdm.pandas()

    is_cuda = tf.test.is_built_with_cuda()
    if is_cuda and len(tf.config.list_physical_devices('GPU')) >= 0:
        print(f"GPU is available.")
    else:
        print("GPU not available, CPU used")

    print("Считывание csv файла...")
    df = pd.read_csv(file_path)

    print("Удаление дубликатов...")
    df = del_duplicates(df)

    print("Удаление не нужных колонок...")
    df = del_columns(df)

    print("Преобразование текста к нижнему регистру...")
    df = to_lower_str(df)

    print("Преобразование категорий...")
    df = mapping_category(df)

    print("Удаление пустых строк с текстом...")
    df = del_empty_text(df)

    print("Склеивание текста...")
    df = text_gluing(df)

    print("Предобработка текста...")
    df = preprocess_text(df)

    print("Очистка от кастомных стопслов...")
    df = cleaning_custom_stopwords(df)

    print("Балансировка данных...")
    df = balancing_data(df)

    print("Балансировка длины текста...")
    df = del_big_text(df)

    print("Преобразование категорий к числовому значению...")
    df = category_encoding(df)

    return df


async def download_file(url, dest):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded_size = 0

                with open(dest, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        if total_size > 0:
                            print(
                                f"Downloaded {downloaded_size} of {total_size} bytes", end='\r')
                print(f"Файл скачан: {dest}")
            else:
                print(f"Ошибка при скачивании: {response.status}")


def text_to_vector(text, model):
    words = text.split()
    word_vectors = []

    for word in words:
        try:
            word_vectors.append(model[word])
        except KeyError:
            continue

    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


async def train(df):
    file_path = "navec/navec_hudlit_v1_12B_500K_300d_100q.tar"
    url = "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar"

    temp_dir = "navec"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if not os.path.exists(file_path):
        print("Файл не найден, начинаем скачивание...")
        await download_file(url, file_path)
    else:
        print("Уже скачан")

    navec = Navec.load(file_path)

    X = np.array([text_to_vector(text, navec) for text in df['text']])
    y = df['category_encoded'].astype(int)

    classifier = RandomForestClassifier(
        n_estimators=100, criterion='entropy', random_state=0).fit(X, y)

    dump(classifier, 'model/model.joblib')

    print("Модель обучена и готова к использованию")


async def train_model():
    data_path = 'data/data.csv'

    if not os.path.exists(data_path):
        print("Data not found")
        print("Starting data loading")
        try:
            await download_data(data_path)
            print("Starting the model training process")
        except Exception as e:
            print(e)

    df = preprocess_data(data_path)

    await train(df)
