import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Загрузка и подготовка данных
data = pd.read_csv("../data/messages.csv")
data['text'] = data['text'].fillna("Photo")
data['date'] = pd.to_datetime(data['date'])
data['msg_length'] = data['text'].apply(lambda x: len(x) if x != "Photo" else 0)
stopwords = {
    'в', 'на', 'с', 'из', 'у', 'по', 'к', 'о', 'об', 'за', 'до', 'от', 'для', 'при', 'через', 'над', 'под', 'без',
    'и', 'а', 'но', 'или', 'что', 'как', 'если', 'то', 'потому', 'либо', 'тоже', 'чтобы', 'да', 'же',
    'не', 'ни', 'ли', 'бы', 'же', 'пусть', 'даже', 'ведь', 'разве', 'уж', 'точно', 'именно', 'почти', 'я'
}

all_text = ' '.join(data['text'].astype(str))

filtered_text = [word for word in all_text.split() if len(word) > 3 and word.lower() not in stopwords]

# Боковое меню навигации
st.sidebar.title('Навигация')
page = st.sidebar.radio("Разделы:", ["Главная", "Данные", "EDA", "Тренды", "Выводы"])

if page == "Главная":
    st.title("Анализ Телеграмм канала по матанализу")
    st.markdown("Данные взяты из Телеграмм канала по матанализу (далее ТГК по матану), с помощью метода подключения к API"
                "через библиотеке ***telethon***. Ссылка на ТГК **https://t.me/+kZ9Y3KuqGwU2NmY6**, были скачены"
                "конкретно комментарии к постам, дата написания их и id пользователя написавшего комментарий.")

    image = Image.open("2025-05-12 17.02.59.jpg")
    st.image(image, use_container_width=True)

elif page == "Данные":
    st.header("Обзор данных")
    # Счетчики метрик
    col1, col2, col3 = st.columns(3)
    col1.metric("Сообщений", len(data))
    col2.metric("Без текста", (data['text']=="Photo").sum())
    col3.metric("Уник. пользователей", data['sender_id'].nunique())
    # Поиск и таблица
    search = st.text_input("Поиск по сообщениям:")
    if search:
        subset = data[data['text'].str.contains(search, case=False)]
    else:
        subset = data
    n_rows = st.slider("Сколько строк показать:", min_value=5, max_value=len(data), value=5)
    st.dataframe(subset.head(n_rows))
    # График распределения по типам
    data['msg_type'] = data['text'].apply(lambda x: "Photo" if x=="Photo" else "Text")
    counts = data['msg_type'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values, color=['#1f77b4','#ff7f0e'])
    ax.set_title("Сообщения: текст vs фото")
    st.pyplot(fig)



elif page == "EDA":
    st.header("Разведочный анализ (EDA)")
    # Тип данных
    st.markdown("### Типы данных по столбцам:")
    for col, dtype in data.dtypes.items():
        st.write(f"**{col}**: {dtype}")

    # Статистики по длине
    st.write("**Статистика длины сообщений:**")
    st.table(data['msg_length'].describe())
    # Топ 20 слов
    st.write("**Топ-20 слов:**")
    word_counts = Counter(filtered_text)
    common_words = word_counts.most_common(20)
    words, counts = zip(*common_words)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts), ax=ax)
    ax.set_title('Топ 20 самых частых слов')
    ax.set_ylabel('Частота')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Подсчёт топ-10 активных пользователей
    top_senders = data["sender_id"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_senders.index.astype(str), y=top_senders.values, ax=ax)
    ax.set_title("Топ 10 самых активных пользователей")
    ax.set_xlabel("ID пользователя")
    ax.set_ylabel("Количество сообщений")
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif page == "Тренды":
    st.header("Тренды и закономерности")
    # Облако слов

    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords)
    wc_img = wc.generate(" ".join(filtered_text))
    fig, ax = plt.subplots()
    ax.imshow(wc_img)
    ax.axis("off")
    st.pyplot(fig)
    # Длинна слова
    max_len = int(data['msg_length'].max())
    min_val, max_val = st.slider("Диапазон длины сообщений", 0, max_len, (0, max_len))
    lengths = data[(data['msg_length'] >= min_val) & (data['msg_length'] <= max_val)]['msg_length']
    fig, ax = plt.subplots()
    sns.histplot(lengths, bins=30, ax=ax, kde=True)
    ax.set_xlabel("Длина сообщения")
    ax.set_ylabel("Частота")
    st.pyplot(fig)

    # Тепловая карта активности
    # Подготовка данных и создания фильтров
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce")

    st.markdown("### Фильтр по дате")
    date_range = st.date_input("Выберите диапазон", [data["date"].dt.date.min(), data["date"].dt.date.max()])

    start_date = pd.to_datetime(date_range[0]).tz_localize("UTC")
    end_date = pd.to_datetime(date_range[1]).tz_localize("UTC")

    filtered = data[(data["date"] >= start_date) & (data["date"] <= end_date)]

    filtered["weekday"] = filtered["date"].dt.day_name()
    filtered["hour"] = filtered["date"].dt.hour


    pivot = filtered.pivot_table(index='weekday', columns='hour', values='text', aggfunc='count', fill_value=0)
    pivot = pivot.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    # Построение тепловой карты
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, cmap='magma', ax=ax)
        ax.set_title("Активность по дням и часам")
        st.pyplot(fig)
    else:
        st.warning("Нет данных для отображения в выбранном диапазоне.")

else:
    st.header("Выводы и рекомендации")
    st.markdown("""
    ### Анализ комментариев учебного ТГК

    Анализ комментариев учебного ТГК показал ряд интересных закономерностей как в содержании сообщений, так и во временной активности студентов. Были построены визуализации:

    - Облако слов  
    - Гистограммы самых частых слов (топ-20 и топ-40)  
    - Тепловая карта активности по дням недели и часам  

    Эти визуализации позволили не просто представить данные, но и сделать конкретные выводы.

    ---

    ### Вежливость в общении

    Во-первых, стиль общения студентов оказался вежливым — среди самых частых слов встречаются:

    - «здравствуйте»  
    - «добрый»  
    - «спасибо»  
    - «пожалуйста»  

    Это говорит о культурной манере общения даже в стрессовых учебных ситуациях.

    ---

    ### Частые темы обращений

    По содержанию комментариев выделено **три основные группы проблем**:

    1. **Проблемы с Геолином** (система электронного обучения)  
    2. **Проблемы с системой оценивания** (баллы, ППА, коллоквиумы)  
    3. **Недостаток информации о курсе** (вопросы по расписанию, щитам, лекциям)

    ---

    ### Временная активность

    - **Чаще пишут после 18:00**, особенно вечером  
    - **Наиболее активные дни** — *вторник* и *четверг*  
    - Это совпадает с публикацией постов преподавателей, вызывая всплески активности  

    > Гипотеза: активность студентов связана со временем публикации постов.

    ---

    ### Активность преподавателей

    Анализ времени публикации самих постов показал:

    - Посты чаще публикуются в **рабочие дни**
    - Чаще — **вечером**, что логично с учётом рабочего графика преподавателей

    Это подтверждает связь между активностью преподавателей и студентов.

    ---

    ### Вывод

    Работа позволила:

    - Визуализировать текстовые данные
    - Выявить паттерны поведения студентов
    - Определить основные болевые точки

    **Практическая ценность**: это может помочь улучшить коммуникацию и сократить число повторяющихся вопросов.
    """)

