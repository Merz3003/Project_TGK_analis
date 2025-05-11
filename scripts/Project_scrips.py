import asyncio
from telethon.sync import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from dotenv import load_dotenv
import os

async def main():
    load_dotenv()

    api_id = int(os.getenv("API_ID"))
    api_hash = os.getenv("API_HASH")

    client = TelegramClient('my_session.session', api_id, api_hash)
    await client.start()

    entity = await client.get_entity('https://t.me/+kZ9Y3KuqGwU2NmY6')
    full_entity = await client(GetFullChannelRequest(entity))
    channel_username = full_entity.full_chat.linked_chat_id

    all_messages = []

    async for message in client.iter_messages(channel_username, reverse=True):
        all_messages.append({
            'id': message.id,
            'date': message.date,
            'text': message.text,
            'sender_id': message.sender_id
        })

    data = pd.DataFrame(all_messages)
    data.to_csv('messages.csv', index=False)

    data['text'] = data['text'].fillna('Photo')
    data['sender_id'] = data['sender_id'].fillna('Unknown')

    post_count = (data['sender_id'] == -1002163089538.0).sum()
    print(f"Количество сообщений от ТГК (закрепленных постов): {post_count}")

    data = data[data['sender_id'] != -1002163089538.0]
    data['date'] = pd.to_datetime(data['date'])

    stopwords_custom = {
        'в', 'на', 'с', 'из', 'у', 'по', 'к', 'о', 'об', 'за', 'до', 'от', 'для', 'при', 'через', 'над', 'под', 'без',
        'и', 'а', 'но', 'или', 'что', 'как', 'если', 'то', 'потому', 'либо', 'тоже', 'чтобы', 'да', 'же',
        'не', 'ни', 'ли', 'бы', 'же', 'пусть', 'даже', 'ведь', 'разве', 'уж', 'точно', 'именно', 'почти', 'я'
    }

    all_text = ' '.join(data['text'].astype(str))
    filtered_words = [word for word in all_text.split() if len(word) > 3 and word.lower() not in stopwords_custom]
    filtered_text = ' '.join(filtered_words)

    # Облако слов
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='black',
        max_words=100,
        colormap='autumn',
        stopwords=stopwords_custom
    ).generate(filtered_text)

    plt.figure(figsize=(14, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Облако слов')
    plt.show()

    # Частота слов: топ-20
    word_counts = Counter(filtered_words)
    common_words = word_counts.most_common(20)
    words, counts = zip(*common_words)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.title('Топ 20 самых частых слов')
    plt.ylabel('Частота')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Частота слов: 21–40
    common_words1 = word_counts.most_common()[20:40]
    words1, counts1 = zip(*common_words1)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words1), y=list(counts1))
    plt.title('Следующие 20 самых частых слов')
    plt.ylabel('Частота')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Временная активность
    data['weekday'] = data['date'].dt.day_name()
    data['hour'] = data['date'].dt.hour

    pivot_table = data.pivot_table(
        index='weekday',
        columns='hour',
        values='text',
        aggfunc='count',
        fill_value=0
    )

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(days_order)

    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_table, cmap='magma', annot=True, fmt='.0f')
    plt.title('Активность по дням недели и часам')
    plt.xlabel('Час суток')
    plt.ylabel('День недели')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())
