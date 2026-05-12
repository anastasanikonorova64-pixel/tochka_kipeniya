import telebot
from telebot import types
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import re

# Токен вашего бота
TOKEN = '8437800508:AAEiKX0T2YX9pWhJbJexcgxbg0MOdtVXixE'  # Замените на ваш токен

bot = telebot.TeleBot(TOKEN)

# Загрузка модели и токенизатора для получения векторных представлений
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Пути к файлам с эталонными текстами
levels_files = {
    'A1': 'a1.txt',
    'A2': 'a2.txt',
    'B1': 'b1.txt',
}

# Загрузка эталонных текстов для каждого уровня
level_embeddings = {}

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

for level, filename in levels_files.items():
    text = ''
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    if text:
        level_embeddings[level] = get_text_embedding(text)
    else:
        level_embeddings[level] = None

# Диапазоны для определения уровня по процентам
def get_level_by_percentage_value(percent, level_ranges):
    for level, (min_perc, max_perc) in level_ranges:
        if min_perc <= percent <= max_perc:
            return level
    return None

length_word_levels = [
    ("A1", (40, 50)),
    ("A2", (20, 30)),
    ("B1", (10, 15)),
    ("B2", (5, 10)),
    ("C1", (2, 5)),
    ("C2", (1, 3))
]

sentence_levels = [
    ("A1", (2, 8)),
    ("A2", (9, 15)),
    ("B1", (16, 25)),
    ("B2", (26, 35)),
    ("C1", (36, 45)),
    ("C2", (46, 10000))
]

# Функции анализа текста
def calculate_word_length_percentages(words):
    total = len(words)
    count_1_2 = sum(1 for w in words if len(w) < 3)
    count_3_6 = sum(1 for w in words if 3 <= len(w) <= 6)
    count_7_and_more_7 = sum(1 for w in words if len(w) >= 7)
    return {
        '1-3': (count_1_2 / total) * 100 if total else 0,
        '4-6': (count_3_6 / total) * 100 if total else 0,
        '7 and more_7': (count_7_and_more_7 / total) * 100 if total else 0
    }

def get_level_by_percentage_value(percent, level_ranges):
    for level, (min_perc, max_perc) in level_ranges:
        if min_perc <= percent <= max_perc:
            return level
    return None

def get_sentence_length_category(text):
    words = re.findall(r'\b\w+\b', text)
    count = len(words)
    if count <= 5:
        return 1
    elif count <= 10:
        return 2
    elif count <= 15:
        return 3
    elif count <= 20:
        return 4
    else:
        return 5

def analyze_text_semantic(text):
    text_emb = get_text_embedding(text)
    similarities = {}
    for level, emb in level_embeddings.items():
        if emb is not None:
            cosine_sim = np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb))
            similarities[level] = cosine_sim
    if not similarities:
        return "Не удалось определить уровень."
    best_level = max(similarities, key=similarities.get)
    return best_level

def определить_уровень(total_words, level_counts):
    levels_order = {"A1": 1, "A2": 2, "B1": 3}
    levels_sorted = sorted(level_counts.items(), key=lambda x: x[1], reverse=True)
    top_level, top_count = levels_sorted[0]
    second_level, second_count = levels_sorted[1]
    difference = abs(levels_order[top_level] - levels_order[second_level])

    if total_words > 70:
        if difference <= 15:
            if levels_order[top_level] > levels_order[second_level]:
                return top_level
            else:
                return second_level
        else:
            return top_level
    else:
        if difference <= 5:
            if levels_order[top_level] > levels_order[second_level]:
                return top_level
            else:
                return second_level
        else:
            return top_level

def calculate_overall_level(semantic_level, word_pct_category, sentence_length_category):
    levels_order = {"A1": 1, "A2": 2, "B1": 3}
    try:
        semantic_num = levels_order[semantic_level]
    except KeyError:
        semantic_num = levels_order["A1"]
    try:
        length_word_num = levels_order[word_pct_category]
    except KeyError:
        length_word_num = levels_order["A1"]
    try:
        sentence_num = levels_order[sentence_length_category]
    except KeyError:
        sentence_num = levels_order["A1"]
    overall_score = (
        semantic_num * 0.5 +
        length_word_num * 0.3 +
        sentence_num * 0.2
    )
    # Находим ближайший уровень
    closest_level = min(levels_order.items(), key=lambda x: abs(x[1] - overall_score))[0]
    return closest_level

# Загрузка словарей слов для каждого уровня
def load_level_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f)

levels_files_words = {
    "A1": "a1.txt",
    "A2": "a2.txt",
    "B1": "b1.txt",
}

level_word_sets = {level: load_level_words(path) for level, path in levels_files_words.items()}

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(
        types.KeyboardButton("Проверить свой уровень английского"),
        types.KeyboardButton("Поиск книг"),
        types.KeyboardButton('Анализ текста')
    )
    bot.send_message(message.chat.id, f'Рада вас приветствовать, {message.from_user.first_name}. Я помогаю ориентироваться в литературных произведениях на английском языке, попробуем?', reply_markup=markup)
points=0
# Обработчик текстовых сообщений
@bot.message_handler(content_types=['text'])
def handle_text(message):
    global points
    global point_read
    global point_use_English
    if message.text == "Проверить свой уровень английского":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Listening"),
            types.KeyboardButton("Reading"),
            types.KeyboardButton("Use of English"),
            types.KeyboardButton("Вернуться в главное меню"),
            types.KeyboardButton("Памятка и подсчет итогов")

        )
        bot.send_message(message.chat.id, text="Для проверки уровня английского вам необходимо пройти три раздела по пять вопросов в каждом. Выберите удобный для вас раздел и начните выполнение заданий", reply_markup=markup)
    elif message.text == "Вернуться в главное меню":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Проверить свой уровень английского"),
            types.KeyboardButton("Поиск книг"),
            types.KeyboardButton('Анализ текста')
        )
        bot.send_message(message.chat.id, text="Вы вернулись в главное меню", reply_markup=markup)
    elif message.text == "Listening":
        with open('./dt_listening_2.mp3', 'rb') as file:
            bot.send_message(message.chat.id, text="Пожалуйста, дождитесь загрузки аудио, это может занять время")
            bot.send_audio(message.chat.id, audio=file)
        bot.send_message(message.chat.id, text="Now get ready to answer the questions. You may use your notes to help you answer.")
        # Отправляем кнопку "Готовы начать?"
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(types.KeyboardButton("Начать"))
        markup.add(types.KeyboardButton("Вернуться в главное меню"))
        bot.send_message(message.chat.id, "Готовы начать?", reply_markup=markup)

    elif message.text == "Начать":
        points=0
        # Отправляем первый вопрос
        first_question = "Why does the student go to see the professor?"
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("To apologize for arriving late"),
            types.KeyboardButton("To turn in her proposal to the professor"),
            types.KeyboardButton('To get help in analyzing her statistics'),
            types.KeyboardButton('To discuss improvements for her proposal')
        )
        bot.send_message(message.chat.id, first_question, reply_markup=markup)

        #отслеживаю вопросы. номер один
    if message.chat.type == 'private':
        if message.text=='To discuss improvements for her proposal':
            points+=1
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("To talk to Miriam at the information desk"),
                types.KeyboardButton("To get help setting up the statistical analysis"),
                types.KeyboardButton('To make changes to her statistical results'),
                types.KeyboardButton('To define her subjects linguistic abilities')
            )
            bot.send_message(message.chat.id, 'Why does the professor suggest that the student go to the Computer Center?', reply_markup=markup)

        elif message.text=='To get help in analyzing her statistics':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("To talk to Miriam at the information desk"),
                types.KeyboardButton("To get help setting up the statistical analysis"),
                types.KeyboardButton('To make changes to her statistical results'),
                types.KeyboardButton('To define her subjects linguistic abilities')
             )
            bot.send_message(message.chat.id,'Why does the professor suggest that the student go to the Computer Center?', reply_markup=markup)

        elif message.text=='To turn in her proposal to the professor':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("To talk to Miriam at the information desk"),
                types.KeyboardButton("To get help setting up the statistical analysis"),
                 types.KeyboardButton('To make changes to her statistical results'),
                types.KeyboardButton('To define her subjects linguistic abilities')
            )
            bot.send_message(message.chat.id,'Why does the professor suggest that the student go to the Computer Center?', reply_markup=markup)

        elif message.text=='To apologize for arriving late':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("To talk to Miriam at the information desk"),
                types.KeyboardButton("To get help setting up the statistical analysis"),
                types.KeyboardButton('To make changes to her statistical results'),
                types.KeyboardButton('To define her subjects linguistic abilities')
            )
            bot.send_message(message.chat.id,'Why does the professor suggest that the student go to the Computer Center?', reply_markup=markup)


        # два
        if message.text=='To get help setting up the statistical analysis':
            points+=1
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("How the analysis will be done"),
                types.KeyboardButton("How she got the statistics into a meaningful form"),
                types.KeyboardButton('How she will contact the international students'),
                types.KeyboardButton('How the items will not be selected')
            )
            bot.send_message(message.chat.id, 'According to the professor, what information should the student add in her proposal?', reply_markup=markup)

        elif message.text=='To talk to Miriam at the information desk':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("How the analysis will be done"),
                types.KeyboardButton("How she got the statistics into a meaningful form"),
                types.KeyboardButton('How she will contact the international students'),
                types.KeyboardButton('How the items will not be selected')
             )
            bot.send_message(message.chat.id,' According to the professor, what information should the student add in her proposal?', reply_markup=markup)

        elif message.text=='To make changes to her statistical results':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("How the analysis will be done"),
                types.KeyboardButton("How she got the statistics into a meaningful form"),
                types.KeyboardButton('How she will contact the international students'),
                types.KeyboardButton('How the items will not be selected')
            )
            bot.send_message(message.chat.id,' According to the professor, what information should the student add in her proposal?', reply_markup=markup)

        elif message.text=='To define her subjects linguistic abilities':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("How the analysis will be done"),
                types.KeyboardButton("How she got the statistics into a meaningful form"),
                types.KeyboardButton('How she will contact the international students'),
                types.KeyboardButton('How the items will not be selected')
            )
            bot.send_message(message.chat.id,' According to the professor, what information should the student add in her proposal?', reply_markup=markup)

        # три
        if message.text=='How the analysis will be done':
            points+=1
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("To suggest the kind of test the student should give her subjects"),
                types.KeyboardButton("To demonstrate international students differing linguistic levels"),
                types.KeyboardButton('To illustrate a flaw in the students proposal'),
                types.KeyboardButton('To compare two language groups')
            )
            bot.send_message(message.chat.id, 'Why does the professor say this?', reply_markup=markup)

        elif message.text=='How she got the statistics into a meaningful form':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("To suggest the kind of test the student should give her subjects"),
                types.KeyboardButton("To demonstrate international students differing linguistic levels"),
                types.KeyboardButton('To illustrate a flaw in the students proposal'),
                types.KeyboardButton('To compare two language groups')
            )
            bot.send_message(message.chat.id, 'Why does the professor say this?', reply_markup=markup)

        elif message.text=='How she will contact the international students':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("To suggest the kind of test the student should give her subjects"),
                types.KeyboardButton("To demonstrate international students differing linguistic levels"),
                types.KeyboardButton('To illustrate a flaw in the students proposal'),
                types.KeyboardButton('To compare two language groups')
            )
            bot.send_message(message.chat.id, 'Why does the professor say this?', reply_markup=markup)

        elif message.text=='How the items will not be selected':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("To suggest the kind of test the student should give her subjects"),
                types.KeyboardButton("To demonstrate international students differing linguistic levels"),
                types.KeyboardButton('To illustrate a flaw in the students proposal'),
                types.KeyboardButton('To compare two language groups')
            )
            bot.send_message(message.chat.id, 'Why does the professor say this?', reply_markup=markup)


        if message.text == 'To illustrate a flaw in the students proposal':
            points += 1
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("They will not approve of her getting help in analyzing her statistics"),
                types.KeyboardButton("They will question what stress patterns she will study"),
                types.KeyboardButton('They will be influenced by her definition of her subjects'),
                types.KeyboardButton('They will not understand who her subjects will be')
            )
            bot.send_message(message.chat.id, 'What does the professor imply about the people who will decide on the grant money?',reply_markup=markup)
        elif message.text == 'To compare two language groups':
            points += 0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("They will not approve of her getting help in analyzing her statistics"),
                types.KeyboardButton("They will question what stress patterns she will study"),
                types.KeyboardButton('They will be influenced by her definition of her subjects'),
                types.KeyboardButton('They will not understand who her subjects will be')
            )
            bot.send_message(message.chat.id,'What does the professor imply about the people who will decide on the grant money?', reply_markup=markup)
        elif message.text == 'To demonstrate international students differing linguistic levels':
            points += 0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("They will not approve of her getting help in analyzing her statistics"),
                types.KeyboardButton("They will question what stress patterns she will study"),
                types.KeyboardButton('They will be influenced by her definition of her subjects'),
                types.KeyboardButton('They will not understand who her subjects will be')
            )
            bot.send_message(message.chat.id,'What does the professor imply about the people who will decide on the grant money?', reply_markup=markup)
        elif message.text == 'To suggest the kind of test the student should give her subjects':
            points += 0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("They will not approve of her getting help in analyzing her statistics"),
                types.KeyboardButton("They will question what stress patterns she will study"),
                types.KeyboardButton('They will be influenced by her definition of her subjects'),
                types.KeyboardButton('They will not understand who her subjects will be')
            )
            bot.send_message(message.chat.id,'What does the professor imply about the people who will decide on the grant money?',reply_markup=markup)

        if message.text=='They will not understand who her subjects will be':
            points+=1
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("Reading"),
                types.KeyboardButton("Use of English"),
                types.KeyboardButton("Вернуться в главное меню"),
                types.KeyboardButton("Памятка и подсчет итогов")

            )
            bot.send_message(message.chat.id, f'Вы набрали следующее количество очков аудирования: {points}. Запомните их!',reply_markup=markup)

        elif message.text=='They will be influenced by her definition of her subjects':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("Reading"),
                types.KeyboardButton("Use of English"),
                types.KeyboardButton("Вернуться в главное меню"),
                types.KeyboardButton("Памятка и подсчет итогов")

            )
            bot.send_message(message.chat.id, f'Вы набрали следующее количество очков аудирования: {points}. Запомните их!',reply_markup=markup)

        elif message.text=='They will question what stress patterns she will study':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("Reading"),
                types.KeyboardButton("Памятка и подсчет итогов"),
                types.KeyboardButton("Use of English"),
                types.KeyboardButton("Вернуться в главное меню")
            )
            bot.send_message(message.chat.id, f'Вы набрали следующее количество очков аудирования: {points}. Запомните их!',reply_markup=markup)

        elif message.text=='They will not approve of her getting help in analyzing her statistics':
            points+=0
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(
                types.KeyboardButton("Reading"),
                types.KeyboardButton("Use of English"),
                types.KeyboardButton("Памятка и подсчет итогов"),
                types.KeyboardButton("Вернуться в главное меню")
            )
            bot.send_message(message.chat.id, f'Вы набрали следующее количество очков аудирования: {points}. Запомните их!',reply_markup=markup)
    # ОКОНЧАНИЕ БЛОКА АУДИО
    #начало чтения


    if message.text == "Reading":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(types.KeyboardButton("Начнём"))
        markup.add(types.KeyboardButton("Вернуться в главное меню"))
        t=(' Geologists have shown that for about 80 percent of the past 2.5 million years, ice-age conditions have prevailed on the Earth^s surface. '
           'During the past one million years, increased glacial conditions have run in cycles of approximately 100,000 years.'
)
        g=('Different factors may contribute to these increases in glaciation at regular intervals throughout Earth^s more geologically recent history. '
        'The three most prominent factors probably relate to the amount of sunlight that reaches the Earth.'
        ' This varies over time for three main reasons. First, the bles* as it spins, due to the pull of the sun and moon. '
        'Furthermore, the Earth tilts* on its axis and the degree of tilt changes over time. '
        'Finally, the orbit of the Earth around the sun is elliptical and the length of the major axis of the ellipse changes over a period of about 100,000 years.'
        ' A mathematician named Milutin Milankovitch discovered in the 1930s that the pattern of insolation, or sunlight, predicted by these eccentricities in the Earth^s movement matched the period of the last several eras of intense glaciation.')

        n=('These Milankovitch insolation cycles were the dominant theory in ice-age research for much of the twentieth century despite the fact that the match between periods of peak insolation and most intense glaciation were not exact.'
           ' For example, a cycle of 400,000 years predicted by the Milankovitch theory has never shown up in the climate records obtained through the study of microfossils deposited on the sea floor.'
           ' Also, recent analysis has shown that the insolation theory predicts peaks of sunlight at intervals of 95,000 and 125,000 years. Climatological data does not support this predicted sunlight peaking. '
           'Other damaging evidence was the indication of a precisely measured sudden rise in temperature at a water-filled cave in Nevada, which preceded the increase in solar radiation that was supposed to cause it.')
        l=('These and other problems with the Milankovitch cycles led some researchers to seek alternative explanations for the cyclic arrival of extended ice ages. '
           'In the 1990s, it was discovered that the orbital inclination of the Earth to the sun and planets could also be responsible for climate changes. '
           'If we imagine a flat plane with the sun in the center and the planets revolving around it, the Earth slowly moves in and out of the flat plane by a few degrees, repeating the cycle every 100,000 years. '
           'Two scientists, Muller and MacDonald, have proposed that it is this orbital inclination which is ultimately responsible for the periods of glaciation and warming. '
           'They argue that because of the oscillation, the Earth periodically travels through clouds of debris, in the form of dust and meteoroids.'
           ' Such debris could reduce the amount of solar energy reaching the surface of our planet, thus plunging it into regular cold periods.')
        j=('The advantage of this theory is that it is not confronted with several of the problems associated with the Milankovitch theory. '
           'In particular, the new theory fits well with the analysis of ocean sediments taken from eight locations around the world. '
           'This analysis yielded data clearly showing the peak of the last several ice ages with a period of 100,000 years and corresponding to the periods when the Earth^s oscillating inclination takes it through clouds of extraterrestrial debris. '
           'However, many researchers in this field are not yet persuaded by the inclination hypothesis. The main problem is that the amount of dust that falls to the ground when the Earth travels through space debris is relatively small - not enough to produce radical climate changes.'
           'Volca Supporters have countered that the by-products created by the dust as it vaporizes on entering the atmosphere cause subtle changes to the energy levels. Nevertheless, the necessary physical proof has yet to be found to convince the skeptics.')
        i=('*wobble: to shake or move from side to side. '                                                                                                   
           '*tilt: to be in a sloping position')
        bot.send_message(message.chat.id, 'Carefully read the suggested text. For your convenience, it is divided into paragraphs. Prepare to answer the questions. <em>Heading: causes of Ice Ages</em>', parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id, f'{t}', reply_markup=markup)
        bot.send_message(message.chat.id, f'{g}', reply_markup=markup)
        bot.send_message(message.chat.id, f'{n}', reply_markup=markup)
        bot.send_message(message.chat.id, f'{l}', reply_markup=markup)
        bot.send_message(message.chat.id, f'{j}', reply_markup=markup)
        bot.send_message(message.chat.id, f'{i}', reply_markup=markup)
        bot.send_message(message.chat.id, "Начнём?", reply_markup=markup)

    elif message.text == "Начнём":
        point_read = 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("ruled"),
            types.KeyboardButton("existed"),
            types.KeyboardButton("survived"),
            types.KeyboardButton("triumphed"),
        )
        bot.send_message(message.chat.id, 'Geologists have shown that for about 80 percent of the past 2.5 million years, ice-age conditions have <em>prevailed</em> on the Earth^s surface. '
                                          'During the past one million years, increased glacial conditions have run in cycles of approximately 100,000 years', parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id, 'The word "prevaied" in the passage is closest in meaning to', reply_markup=markup)

    if message.text == 'existed':
        point_read += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("They affect the Earth's spin."),
            types.KeyboardButton("They are geologically recent."),
            types.KeyboardButton('Оnly three factors relate to levels of sunlight.'),
            types.KeyboardButton('Other factors than those relating to the sunlight affect ice buildup.')
        )
        bot.send_message(message.chat.id,'What can be inferred from paragraph 2 about the factors that contribute to glaciation?',reply_markup=markup)

    elif message.text == 'ruled':
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("They affect the Earth's spin."),
            types.KeyboardButton("They are geologically recent."),
            types.KeyboardButton('Оnly three factors relate to levels of sunlight.'),
            types.KeyboardButton('Other factors than those relating to the sunlight affect ice buildup.')
        )
        bot.send_message(message.chat.id,'What can be inferred from paragraph 2 about the factors that contribute to glaciation?',reply_markup=markup)

    elif message.text == 'survived':
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("They affect the Earth's spin."),
            types.KeyboardButton("They are geologically recent."),
            types.KeyboardButton('Оnly three factors relate to levels of sunlight.'),
            types.KeyboardButton('Other factors than those relating to the sunlight affect ice buildup.')
        )
        bot.send_message(message.chat.id,'What can be inferred from paragraph 2 about the factors that contribute to glaciation?',reply_markup=markup)

    elif message.text == 'triumphed':
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("They affect the Earth's spin."),
            types.KeyboardButton("They are geologically recent."),
            types.KeyboardButton('Оnly three factors relate to levels of sunlight.'),
            types.KeyboardButton('Other factors than those relating to the sunlight affect ice buildup.')
        )
        bot.send_message(message.chat.id,'What can be inferred from paragraph 2 about the factors that contribute to glaciation?',reply_markup=markup)

    if message.text == 'Other factors than those relating to the sunlight affect ice buildup.':
        point_read += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("various movements of the Earth as it spins"),
            types.KeyboardButton("degree of change in the Earth's tilt over time"),
            types.KeyboardButton("pattern of insolation matching the Earth's movement"),
            types.KeyboardButton("changing distance to the sun during the Earth's elliptical orbit")
        )
        bot.send_message(message.chat.id,
                         "Many different factors may contribute to these increases in glaciation at regular intervals throughout Earth's more geologically recent history."
                         " The three most prominent factors probably relate to the amount of sunlight that reaches the Earth. "
                         "This varies over time for three main reasons. First, the planet wobbles as it spins, due to the pull of the sun and moon. Furthermore, the Earth tilts on its axis and the degree of tilt changes over time."
                         " Finally, the orbit of the Earth around the sun is elliptical and the length of the major axis of the ellipse changes over a period of about 100,000 years."
                         " A mathematician named Milutin Milankovitch discovered in the 1930s that the pattern of insolation, or sunlight, predicted by <em>these eccentricities </em> in the Earth's movement matched the period of the last several eras of intense glaciation.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,'The phrase "these eccentricities" in the passage refers to all of the following EXCEPT',reply_markup=markup)

    elif message.text == "They affect the Earth's spin.":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("various movements of the Earth as it spins"),
            types.KeyboardButton("degree of change in the Earth's tilt over time"),
            types.KeyboardButton("pattern of insolation matching the Earth's movement"),
            types.KeyboardButton("changing distance to the sun during the Earth's elliptical orbit")
        )
        bot.send_message(message.chat.id,
                         "Many different factors may contribute to these increases in glaciation at regular intervals throughout Earth's more geologically recent history."
                         " The three most prominent factors probably relate to the amount of sunlight that reaches the Earth. "
                         "This varies over time for three main reasons. First, the planet wobbles as it spins, due to the pull of the sun and moon. Furthermore, the Earth tilts on its axis and the degree of tilt changes over time."
                         " Finally, the orbit of the Earth around the sun is elliptical and the length of the major axis of the ellipse changes over a period of about 100,000 years."
                         " A mathematician named Milutin Milankovitch discovered in the 1930s that the pattern of insolation, or sunlight, predicted by <em>these eccentricities </em> in the Earth's movement matched the period of the last several eras of intense glaciation.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,'The phrase "these eccentricities" in the passage refers to all of the following EXCEPT',reply_markup=markup)

    elif message.text == "Оnly three factors relate to levels of sunlight.":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("various movements of the Earth as it spins"),
            types.KeyboardButton("degree of change in the Earth's tilt over time"),
            types.KeyboardButton("pattern of insolation matching the Earth's movement"),
            types.KeyboardButton("changing distance to the sun during the Earth's elliptical orbit")
        )
        bot.send_message(message.chat.id,
                         "Many different factors may contribute to these increases in glaciation at regular intervals throughout Earth's more geologically recent history."
                         " The three most prominent factors probably relate to the amount of sunlight that reaches the Earth. "
                         "This varies over time for three main reasons. First, the planet wobbles as it spins, due to the pull of the sun and moon. Furthermore, the Earth tilts on its axis and the degree of tilt changes over time."
                         " Finally, the orbit of the Earth around the sun is elliptical and the length of the major axis of the ellipse changes over a period of about 100,000 years."
                         " A mathematician named Milutin Milankovitch discovered in the 1930s that the pattern of insolation, or sunlight, predicted by <em>these eccentricities </em> in the Earth's movement matched the period of the last several eras of intense glaciation.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,'The phrase "these eccentricities" in the passage refers to all of the following EXCEPT', reply_markup=markup)

    elif message.text == "They are geologically recent.":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("various movements of the Earth as it spins"),
            types.KeyboardButton("degree of change in the Earth's tilt over time"),
            types.KeyboardButton("pattern of insolation matching the Earth's movement"),
            types.KeyboardButton("changing distance to the sun during the Earth's elliptical orbit")
        )
        bot.send_message(message.chat.id,
                         "Many different factors may contribute to these increases in glaciation at regular intervals throughout Earth's more geologically recent history."
                         " The three most prominent factors probably relate to the amount of sunlight that reaches the Earth. "
                         "This varies over time for three main reasons. First, the planet wobbles as it spins, due to the pull of the sun and moon. Furthermore, the Earth tilts on its axis and the degree of tilt changes over time."
                         " Finally, the orbit of the Earth around the sun is elliptical and the length of the major axis of the ellipse changes over a period of about 100,000 years."
                         " A mathematician named Milutin Milankovitch discovered in the 1930s that the pattern of insolation, or sunlight, predicted by <em>these eccentricities </em> in the Earth's movement matched the period of the last several eras of intense glaciation.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,'The phrase "these eccentricities" in the passage refers to all of the following EXCEPT',reply_markup=markup)


    if message.text == "pattern of insolation matching the Earth's movement":
        point_read += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("the peaks of sunlight occurred at intervals of 95.000 and 125.000 years"),
            types.KeyboardButton("peaks of insolation,intense glaciation did not match"),
            types.KeyboardButton("there were climate records of a 400,000-year cycle"),
            types.KeyboardButton("there were microfossil deposits on the sea floor")
        )
        bot.send_message(message.chat.id,'Scientists accepted the Milankovitch theory even though',reply_markup=markup)

    elif message.text == "changing distance to the sun during the Earth's elliptical orbit":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("the peaks of sunlight occurred at intervals of 95.000 and 125.000 years"),
            types.KeyboardButton("peaks of insolation,intense glaciation did not match"),
            types.KeyboardButton("there were climate records of a 400,000-year cycle"),
            types.KeyboardButton("there were microfossil deposits on the sea floor")
        )
        bot.send_message(message.chat.id, 'Scientists accepted the Milankovitch theory even though', reply_markup=markup)

    elif message.text == "various movements of the Earth as it spins":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("the peaks of sunlight occurred at intervals of 95.000 and 125.000 years"),
            types.KeyboardButton("peaks of insolation,intense glaciation did not match"),
            types.KeyboardButton("there were climate records of a 400,000-year cycle"),
            types.KeyboardButton("there were microfossil deposits on the sea floor")
        )
        bot.send_message(message.chat.id, 'Scientists accepted the Milankovitch theory even though', reply_markup=markup)

    elif message.text == "degree of change in the Earth's tilt over time":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("the peaks of sunlight occurred at intervals of 95.000 and 125.000 years"),
            types.KeyboardButton("peaks of insolation,intense glaciation did not match"),
            types.KeyboardButton("there were climate records of a 400,000-year cycle"),
            types.KeyboardButton("there were microfossil deposits on the sea floor")
        )
        bot.send_message(message.chat.id, 'Scientists accepted the Milankovitch theory even though', reply_markup=markup)

    if message.text == "peaks of insolation,intense glaciation did not match":
        point_read += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("To help the reader imagine the motion of celestial bodies"),
            types.KeyboardButton("To demonstrate to the reader how the Earth orbits the sun"),
            types.KeyboardButton("To support the argument that the orbital inclination increases tilting"),
            types.KeyboardButton("To show how the Milankovitch theory doesn't explain the cyclical changes in climate")
        )
        bot.send_message(message.chat.id,
                         "These and other problems with the Milankovitch cycles led some researchers to seek alternative explanations for the cyclic arrival of extended ice ages. "
                         "In the 1990s, it was discovered that the orbital inclination of the Earth to the sun and planets could also be responsible for climate changes. If we imagine a flat plane with the sun in the center and the planets revolving around it, the Earth slowly moves in and out of the flat plane by a few degrees, repeating the cycle every 100,000 years. "
                         "Two scientists. Muller and MacDonald, have proposed that it is this orbital inclination which is ultimately responsible for the periods of glaciation and warming. They argue that because of the oscillation. "
                         "The Earth periodically travels through clouds of debris, in the form of dust and meteoroids. "
                         "Such debris could reduce the amount of solar energy reaching the surface of our planet, thus plunging it into regular cold periods.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id, 'In paragraph 4, why does the author suggest the image of a flat plane?',
                         reply_markup=markup)

    elif message.text == "the peaks of sunlight occurred at intervals of 95.000 and 125.000 years":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("To help the reader imagine the motion of celestial bodies"),
            types.KeyboardButton("To demonstrate to the reader how the Earth orbits the sun"),
            types.KeyboardButton("To support the argument that the orbital inclination increases tilting"),
            types.KeyboardButton("To show how the Milankovitch theory doesn't explain the cyclical changes in climate")
        )
        bot.send_message(message.chat.id,
                         "These and other problems with the Milankovitch cycles led some researchers to seek alternative explanations for the cyclic arrival of extended ice ages. "
                         "In the 1990s, it was discovered that the orbital inclination of the Earth to the sun and planets could also be responsible for climate changes. If we imagine a flat plane with the sun in the center and the planets revolving around it, the Earth slowly moves in and out of the flat plane by a few degrees, repeating the cycle every 100,000 years. "
                         "Two scientists. Muller and MacDonald, have proposed that it is this orbital inclination which is ultimately responsible for the periods of glaciation and warming. They argue that because of the oscillation. "
                         "The Earth periodically travels through clouds of debris, in the form of dust and meteoroids. "
                         "Such debris could reduce the amount of solar energy reaching the surface of our planet, thus plunging it into regular cold periods.", parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id, 'In paragraph 4, why does the author suggest the image of a flat plane?', reply_markup=markup)

    elif message.text == "there were microfossil deposits on the sea floor":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("To help the reader imagine the motion of celestial bodies"),
            types.KeyboardButton("To demonstrate to the reader how the Earth orbits the sun"),
            types.KeyboardButton("To support the argument that the orbital inclination increases tilting"),
            types.KeyboardButton("To show how the Milankovitch theory doesn't explain the cyclical changes in climate")
        )
        bot.send_message(message.chat.id,
                         "These and other problems with the Milankovitch cycles led some researchers to seek alternative explanations for the cyclic arrival of extended ice ages. "
                         "In the 1990s, it was discovered that the orbital inclination of the Earth to the sun and planets could also be responsible for climate changes. If we imagine a flat plane with the sun in the center and the planets revolving around it, the Earth slowly moves in and out of the flat plane by a few degrees, repeating the cycle every 100,000 years. "
                         "Two scientists. Muller and MacDonald, have proposed that it is this orbital inclination which is ultimately responsible for the periods of glaciation and warming. They argue that because of the oscillation. "
                         "The Earth periodically travels through clouds of debris, in the form of dust and meteoroids. "
                         "Such debris could reduce the amount of solar energy reaching the surface of our planet, thus plunging it into regular cold periods.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id, 'In paragraph 4, why does the author suggest the image of a flat plane?',
                         reply_markup=markup)

    elif message.text == "there were climate records of a 400,000-year cycle":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("To help the reader imagine the motion of celestial bodies"),
            types.KeyboardButton("To demonstrate to the reader how the Earth orbits the sun"),
            types.KeyboardButton("To support the argument that the orbital inclination increases tilting"),
            types.KeyboardButton("To show how the Milankovitch theory doesn't explain the cyclical changes in climate")
        )
        bot.send_message(message.chat.id,
                         "These and other problems with the Milankovitch cycles led some researchers to seek alternative explanations for the cyclic arrival of extended ice ages. "
                         "In the 1990s, it was discovered that the orbital inclination of the Earth to the sun and planets could also be responsible for climate changes. If we imagine a flat plane with the sun in the center and the planets revolving around it, the Earth slowly moves in and out of the flat plane by a few degrees, repeating the cycle every 100,000 years. "
                         "Two scientists. Muller and MacDonald, have proposed that it is this orbital inclination which is ultimately responsible for the periods of glaciation and warming. They argue that because of the oscillation. "
                         "The Earth periodically travels through clouds of debris, in the form of dust and meteoroids. "
                         "Such debris could reduce the amount of solar energy reaching the surface of our planet, thus plunging it into regular cold periods.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id, 'In paragraph 4, why does the author suggest the image of a flat plane?',
                         reply_markup=markup)

    if message.text == "To help the reader imagine the motion of celestial bodies":
        point_read += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("such debris"),
            types.KeyboardButton("solar energy"),
            types.KeyboardButton("the surface"),
            types.KeyboardButton("our planet")
        )
        bot.send_message(message.chat.id,
                         "These and other problems with the Milankovitch cycles led some researchers to seek alternative explanations for the cyclic arrival of extended ice ages."
                         " In the 1990s, it was discovered that the orbital inclination of the Earth to the sun and planets could also be responsible for climate changes. "
                         "If we imagine a flat plane with the sun in the center and the planets revolving around it, the Earth slowly moves in and out of the flat plane by a few degrees, repeating the cycle every 100,000 years."
                         " Two scientists. Muller and MacDonald, have proposed that it is this orbital inclination which is ultimately responsible for the periods of glaciation and warming. "
                         "They argue that because of the oscillation, the Earth periodically travels through clouds of debris, in the form of dust and meteoroids. "
                         "Such debris could reduce the amount of solar energy reaching the surface of our planet. Thus plunging <em> it </em> into regular cold periods.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id, 'The word "it" in the passage refers to', reply_markup=markup)

    elif message.text == "To demonstrate to the reader how the Earth orbits the sun":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("such debris"),
            types.KeyboardButton("solar energy"),
            types.KeyboardButton("the surface"),
            types.KeyboardButton("our planet")
        )
        bot.send_message(message.chat.id,
                         "These and other problems with the Milankovitch cycles led some researchers to seek alternative explanations for the cyclic arrival of extended ice ages."
                         " In the 1990s, it was discovered that the orbital inclination of the Earth to the sun and planets could also be responsible for climate changes. "
                         "If we imagine a flat plane with the sun in the center and the planets revolving around it, the Earth slowly moves in and out of the flat plane by a few degrees, repeating the cycle every 100,000 years."
                         " Two scientists. Muller and MacDonald, have proposed that it is this orbital inclination which is ultimately responsible for the periods of glaciation and warming. "
                         "They argue that because of the oscillation, the Earth periodically travels through clouds of debris, in the form of dust and meteoroids. "
                         "Such debris could reduce the amount of solar energy reaching the surface of our planet. Thus plunging <em> it </em> into regular cold periods.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id, 'The word "it" in the passage refers to', reply_markup=markup)

    elif message.text == "To show how the Milankovitch theory doesn't explain the cyclical changes in climate":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("such debris"),
            types.KeyboardButton("solar energy"),
            types.KeyboardButton("the surface"),
            types.KeyboardButton("our planet")
        )
        bot.send_message(message.chat.id,
                         "These and other problems with the Milankovitch cycles led some researchers to seek alternative explanations for the cyclic arrival of extended ice ages."
                         " In the 1990s, it was discovered that the orbital inclination of the Earth to the sun and planets could also be responsible for climate changes. "
                         "If we imagine a flat plane with the sun in the center and the planets revolving around it, the Earth slowly moves in and out of the flat plane by a few degrees, repeating the cycle every 100,000 years."
                         " Two scientists. Muller and MacDonald, have proposed that it is this orbital inclination which is ultimately responsible for the periods of glaciation and warming. "
                         "They argue that because of the oscillation, the Earth periodically travels through clouds of debris, in the form of dust and meteoroids. "
                         "Such debris could reduce the amount of solar energy reaching the surface of our planet. Thus plunging <em> it </em> into regular cold periods.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id, 'The word "it" in the passage refers to', reply_markup=markup)
    elif message.text == "To support the argument that the orbital inclination increases tilting":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("such debris"),
            types.KeyboardButton("solar energy"),
            types.KeyboardButton("the surface"),
            types.KeyboardButton("our planet")
        )
        bot.send_message(message.chat.id,
                         "These and other problems with the Milankovitch cycles led some researchers to seek alternative explanations for the cyclic arrival of extended ice ages."
                         " In the 1990s, it was discovered that the orbital inclination of the Earth to the sun and planets could also be responsible for climate changes. "
                         "If we imagine a flat plane with the sun in the center and the planets revolving around it, the Earth slowly moves in and out of the flat plane by a few degrees, repeating the cycle every 100,000 years."
                         " Two scientists. Muller and MacDonald, have proposed that it is this orbital inclination which is ultimately responsible for the periods of glaciation and warming. "
                         "They argue that because of the oscillation, the Earth periodically travels through clouds of debris, in the form of dust and meteoroids. "
                         "Such debris could reduce the amount of solar energy reaching the surface of our planet. Thus plunging <em> it </em> into regular cold periods.",
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id, 'The word "it" in the passage refers to', reply_markup=markup)

    if message.text == "our planet":
        point_read += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("The climate records obtained by studying microfossil deposits not matching his predicted cycle"),
            types.KeyboardButton("The irregularities of the Earth's movements through orbital inclinations not following any pattern"),
            types.KeyboardButton("The Earth's spin wobbling in relation to the Earth's oscillating inclination"),
            types.KeyboardButton("The peak in the ice ages occurring at intervals between 95,000 and 125,000 years instead of 400,000")
        )

        bot.send_message(message.chat.id, 'What problem in the Milankovitch theory was mentioned as being explained by the Muller and MacDonald theory?', reply_markup=markup)
        bot.send_message(message.chat.id,"Response options: "
                                         " "
                                         "1. The climate records obtained by studying microfossil deposits not matching his predicted cycle"
                                         " "
                                         "2. The irregularities of the Earth's movements through orbital inclinations not following any pattern"
                                         " "
                                         "3. The Earth's spin wobbling in relation to the Earth's oscillating inclination"
                                         " "
                                         "4. The peak in the ice ages occurring at intervals between 95,000 and 125,000 years instead of 400,000",reply_markup=markup)

    elif message.text == "such debris":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton(
                "The climate records obtained by studying microfossil deposits not matching his predicted cycle"),
            types.KeyboardButton(
                "The irregularities of the Earth's movements through orbital inclinations not following any pattern"),
            types.KeyboardButton("The Earth's spin wobbling in relation to the Earth's oscillating inclination"),
            types.KeyboardButton(
                "The peak in the ice ages occurring at intervals between 95,000 and 125,000 years instead of 400,000")
        )

        bot.send_message(message.chat.id,
                         'What problem in the Milankovitch theory was mentioned as being explained by the Muller and MacDonald theory?',
                         reply_markup=markup)
        bot.send_message(message.chat.id, "Response options: "
                                          " "
                                          "1. The climate records obtained by studying microfossil deposits not matching his predicted cycle"
                                          " "
                                          "2. The irregularities of the Earth's movements through orbital inclinations not following any pattern"
                                          " "
                                          "3. The Earth's spin wobbling in relation to the Earth's oscillating inclination"
                                          " "
                                          "4. The peak in the ice ages occurring at intervals between 95,000 and 125,000 years instead of 400,000",
                         reply_markup=markup)

    elif message.text == "the surface":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton(
                "The climate records obtained by studying microfossil deposits not matching his predicted cycle"),
            types.KeyboardButton(
                "The irregularities of the Earth's movements through orbital inclinations not following any pattern"),
            types.KeyboardButton("The Earth's spin wobbling in relation to the Earth's oscillating inclination"),
            types.KeyboardButton(
                "The peak in the ice ages occurring at intervals between 95,000 and 125,000 years instead of 400,000")
        )

        bot.send_message(message.chat.id,
                         'What problem in the Milankovitch theory was mentioned as being explained by the Muller and MacDonald theory?',
                         reply_markup=markup)
        bot.send_message(message.chat.id, "Response options: "
                                          " "
                                          "1. The climate records obtained by studying microfossil deposits not matching his predicted cycle"
                                          " "
                                          "2. The irregularities of the Earth's movements through orbital inclinations not following any pattern"
                                          " "
                                          "3. The Earth's spin wobbling in relation to the Earth's oscillating inclination"
                                          " "
                                          "4. The peak in the ice ages occurring at intervals between 95,000 and 125,000 years instead of 400,000",
                         reply_markup=markup)

    elif message.text == "solar energy":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton(
                "The climate records obtained by studying microfossil deposits not matching his predicted cycle"),
            types.KeyboardButton(
                "The irregularities of the Earth's movements through orbital inclinations not following any pattern"),
            types.KeyboardButton("The Earth's spin wobbling in relation to the Earth's oscillating inclination"),
            types.KeyboardButton(
                "The peak in the ice ages occurring at intervals between 95,000 and 125,000 years instead of 400,000")
        )

        bot.send_message(message.chat.id,
                         'What problem in the Milankovitch theory was mentioned as being explained by the Muller and MacDonald theory?',
                         reply_markup=markup)
        bot.send_message(message.chat.id, "Response options: "
                                          " "
                                          "1. The climate records obtained by studying microfossil deposits not matching his predicted cycle"
                                          " "
                                          "2. The irregularities of the Earth's movements through orbital inclinations not following any pattern"
                                          " "
                                          "3. The Earth's spin wobbling in relation to the Earth's oscillating inclination"
                                          " "
                                          "4. The peak in the ice ages occurring at intervals between 95,000 and 125,000 years instead of 400,000",
                         reply_markup=markup)

    if message.text == 'The climate records obtained by studying microfossil deposits not matching his predicted cycle':
        point_read += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Listening"),
            types.KeyboardButton("Use of English"),
            types.KeyboardButton("Вернуться в главное меню")
        )
        bot.send_message(message.chat.id, f'Вы набрали следующее количество очков чтения: {point_read}. Запомните их!',reply_markup=markup)
    elif message.text == "The irregularities of the Earth's movements through orbital inclinations not following any pattern":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Listening"),
            types.KeyboardButton("Use of English"),
            types.KeyboardButton("Вернуться в главное меню")
        )
        bot.send_message(message.chat.id, f'Вы набрали следующее количество очков чтения: {point_read}. Запомните их!',
                         reply_markup=markup)
    elif message.text == "The peak in the ice ages occurring at intervals between 95,000 and 125,000 years instead of 400,000":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Listening"),
            types.KeyboardButton("Use of English"),
            types.KeyboardButton("Вернуться в главное меню")
        )
        bot.send_message(message.chat.id, f'Вы набрали следующее количество очков чтения: {point_read}. Запомните их!',
                         reply_markup=markup)
    elif message.text == "The Earth's spin wobbling in relation to the Earth's oscillating inclination":
        point_read += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Listening"),
            types.KeyboardButton("Use of English"),
            types.KeyboardButton("Вернуться в главное меню")
        )
        bot.send_message(message.chat.id, f'Вы набрали следующее количество очков чтения: {point_read}. Запомните их!',
                         reply_markup=markup)

#НОВЫЙ РАЗДЕЛ НАВЫК АНГЛА
    elif message.text == "Use of English":
        bot.send_message(message.chat.id, text="You will be offered a text, read it and decide which answer best fits each space")
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(types.KeyboardButton("Commencement"))
        markup.add(types.KeyboardButton("Вернуться в главное меню"))
        bot.send_message(message.chat.id, "Click Commencement if you are ready to start", reply_markup=markup)


    if message.text == 'Commencement':
        point_use_English=0
        bot.send_message(message.chat.id,'Whose wave power is it?')
        bot.send_message(message.chat.id,"Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
        "More than twenty-one million pounds' <em> <u> (1) ...... </u></em> of funding has been agreed for what is (2) ...... as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
        " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                                             "Once it is in (3) ...... operation, Wave Hub is (4) ...... to support the largest concentration of wave-energy machines anywhere in the world. "
                                             "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                                             , parse_mode='html')

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("amount"),
            types.KeyboardButton("value"),
            types.KeyboardButton("worth"),
            types.KeyboardButton("quantity")
        )
        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>first</u> </em> space.', parse_mode='html', reply_markup=markup)


    if message.text == "worth":
        point_use_English += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("referred"),
            types.KeyboardButton("called"),
            types.KeyboardButton("entitled"),
            types.KeyboardButton("known")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>second</u> </em> space.', parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em>worth</em> of funding has been agreed for what is <em> <u>(2) ......</u></em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in (3) ...... operation, Wave Hub is (4) ...... to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "quantity":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("referred"),
            types.KeyboardButton("called"),
            types.KeyboardButton("entitled"),
            types.KeyboardButton("known")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>second</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em>quantity</em> of funding has been agreed for what is <em> <u>(2) ......</u></em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in (3) ...... operation, Wave Hub is (4) ...... to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "amount":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("referred"),
            types.KeyboardButton("called"),
            types.KeyboardButton("entitled"),
            types.KeyboardButton("known")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>second</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em>amount</em> of funding has been agreed for what is <em> <u>(2) ......</u></em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in (3) ...... operation, Wave Hub is (4) ...... to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "value":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("referred"),
            types.KeyboardButton("called"),
            types.KeyboardButton("entitled"),
            types.KeyboardButton("known")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>second</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em>value</em> of funding has been agreed for what is <em> <u>(2) ......</u></em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in (3) ...... operation, Wave Hub is (4) ...... to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')


    if message.text == "known":
        point_use_English += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("whole"),
            types.KeyboardButton("full"),
            types.KeyboardButton("entire"),
            types.KeyboardButton("thorough")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fird</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)known</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em> <u>(3) ......</u></em> operation, Wave Hub is (4) ...... to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "entitled":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("whole"),
            types.KeyboardButton("full"),
            types.KeyboardButton("entire"),
            types.KeyboardButton("thorough")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fird</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)entitled</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em> <u>(3) ......</u></em> operation, Wave Hub is (4) ...... to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "called":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("whole"),
            types.KeyboardButton("full"),
            types.KeyboardButton("entire"),
            types.KeyboardButton("thorough")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fird</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)called</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em> <u>(3) ......</u></em> operation, Wave Hub is (4) ...... to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "referred":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("whole"),
            types.KeyboardButton("full"),
            types.KeyboardButton("entire"),
            types.KeyboardButton("thorough")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fird</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)referred</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em> <u>(3) ......</u></em> operation, Wave Hub is (4) ...... to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    if message.text == "full":
        point_use_English += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("likely"),
            types.KeyboardButton("probable"),
            types.KeyboardButton("plausible"),
            types.KeyboardButton("surely")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fourth</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)full</em> operation, Wave Hub is <em><u>(4) ......</u></em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "whole":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("likely"),
            types.KeyboardButton("probable"),
            types.KeyboardButton("plausible"),
            types.KeyboardButton("surely")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fourth</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)whole</em> operation, Wave Hub is <em><u>(4) ......</u></em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                        ,parse_mode='html')

    elif message.text == "entire":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("likely"),
            types.KeyboardButton("probable"),
            types.KeyboardButton("plausible"),
            types.KeyboardButton("surely")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fourth</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)entire</em> operation, Wave Hub is <em><u>(4) ......</u></em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "thorough":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("likely"),
            types.KeyboardButton("probable"),
            types.KeyboardButton("plausible"),
            types.KeyboardButton("surely")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fourth</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)thorough</em> operation, Wave Hub is <em><u>(4) ......</u></em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous (5) ...... forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    if message.text == "likely":
        point_use_English += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("tread"),
            types.KeyboardButton("strike"),
            types.KeyboardButton("pace"),
            types.KeyboardButton("step")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fifth</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)likely</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em><u>(5) ......</u></em> forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "probable":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("tread"),
            types.KeyboardButton("strike"),
            types.KeyboardButton("pace"),
            types.KeyboardButton("step")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fifth</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)probable</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em><u>(5) ......</u></em> forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "plausible":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("tread"),
            types.KeyboardButton("strike"),
            types.KeyboardButton("pace"),
            types.KeyboardButton("step")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fifth</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)plausible</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em><u>(5) ......</u></em> forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "surely":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("tread"),
            types.KeyboardButton("strike"),
            types.KeyboardButton("pace"),
            types.KeyboardButton("step")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em> <u>fifth</u> </em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)surely</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em><u>(5) ......</u></em> forward in the development of wave power, which has tended to (6) ...... behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    if message.text == "step":
        point_use_English += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("lag"),
            types.KeyboardButton("leave"),
            types.KeyboardButton("drag"),
            types.KeyboardButton("delay")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em><u>sixth</u></em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em>(5)step</em> forward in the development of wave power, which has tended to <em><u>(6) ......</u></em> behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "pace":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("lag"),
            types.KeyboardButton("leave"),
            types.KeyboardButton("drag"),
            types.KeyboardButton("delay")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em><u>sixth</u></em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em>(5)pace</em> forward in the development of wave power, which has tended to <em><u>(6) ......</u></em> behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "strike":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("lag"),
            types.KeyboardButton("leave"),
            types.KeyboardButton("drag"),
            types.KeyboardButton("delay")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em><u>sixth</u></em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em>(5)strike</em> forward in the development of wave power, which has tended to <em><u>(6) ......</u></em> behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "tread":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("lag"),
            types.KeyboardButton("leave"),
            types.KeyboardButton("drag"),
            types.KeyboardButton("delay")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em><u>sixth</u></em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em>(5)tread</em> forward in the development of wave power, which has tended to <em><u>(6) ......</u></em> behind its cousins in the other main (7) ...... of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')


    if message.text == "lag":
        point_use_English += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("limbs"),
            types.KeyboardButton("branches"),
            types.KeyboardButton("wings"),
            types.KeyboardButton("prongs")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em><u>seventh</u></em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em>(5)</em> forward in the development of wave power, which has tended to <em>(6)lag</em> behind its cousins in the other main <em><u>(7) ......</u></em> of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "leave":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("limbs"),
            types.KeyboardButton("branches"),
            types.KeyboardButton("wings"),
            types.KeyboardButton("prongs")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em><u>seventh</u></em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em>(5)</em> forward in the development of wave power, which has tended to <em>(6)leave</em> behind its cousins in the other main <em><u>(7) ......</u></em> of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "drag":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("limbs"),
            types.KeyboardButton("branches"),
            types.KeyboardButton("wings"),
            types.KeyboardButton("prongs")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em><u>seventh</u></em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em>(5)</em> forward in the development of wave power, which has tended to <em>(6)drag</em> behind its cousins in the other main <em><u>(7) ......</u></em> of renewable energy technology: wind power and solar power."
                         ,parse_mode='html')

    elif message.text == "delay":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("limbs"),
            types.KeyboardButton("branches"),
            types.KeyboardButton("wings"),
            types.KeyboardButton("prongs")
        )

        bot.send_message(message.chat.id, 'Choose an answer option that fits the <em><u>seventh</u></em> space.',
                         parse_mode='html', reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Construction of the world's biggest wave-energy installation is going ahead off the coast of Cornwall in southwest England."
                         "More than twenty-one million pounds' (1) <em></em> of funding has been agreed for what is <em>(2)</em> as Wave Hub, a giant electrical terminal on the seabed ten miles off the coast."
                         " Wave Hub will allow a number of different wave-energy devices operating in the area to transmit the energy they generate along a high-voltage undersea cable, back to the shore."
                         "Once it is in <em>3)</em>, Wave Hub is <em>(4)</em> to support the largest concentration of wave-energy machines anywhere in the world. "
                         "It will also mark an enormous <em>(5)</em> forward in the development of wave power, which has tended to <em>(6)delay</em> behind its cousins in the other main <em><u>(7) ......</u></em> of renewable energy technology: wind power and solar power.",
                         parse_mode='html')


    if message.text == "branches":
        point_use_English += 1
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Listening"),
            types.KeyboardButton("Reading"),
            types.KeyboardButton("Вернуться в главное меню"),
            types.KeyboardButton("Памятка и подсчет итогов")
        )
        bot.send_message(message.chat.id,
                         f'Вы набрали следующее количество очков использования английского: {point_use_English}. Запомните их!',
                         reply_markup=markup)

    elif message.text == "limbs":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Listening"),
            types.KeyboardButton("Reading"),
            types.KeyboardButton("Вернуться в главное меню"),
            types.KeyboardButton("Памятка и подсчет итогов")

        )
        bot.send_message(message.chat.id,
                         f'Вы набрали следующее количество очков использования английского: {point_use_English}. Запомните их!',
                         reply_markup=markup)

    elif message.text == "wings":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Listening"),
            types.KeyboardButton("Reading"),
            types.KeyboardButton("Вернуться в главное меню"),
            types.KeyboardButton("Памятка и подсчет итогов")

        )
        bot.send_message(message.chat.id,
                         f'Вы набрали следующее количество очков использования английского: {point_use_English}. Запомните их!',
                         reply_markup=markup)

    elif message.text == "prongs":
        point_use_English += 0
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Listening"),
            types.KeyboardButton("Reading"),
            types.KeyboardButton("Вернуться в главное меню"),
            types.KeyboardButton("Памятка и подсчет итогов")

        )
        bot.send_message(message.chat.id, f'Вы набрали следующее количество очков использования английского: {point_use_English}. Запомните их!',
                         reply_markup=markup)

#################
    elif message.text == "Памятка и подсчет итогов":
        bot.send_message(message.chat.id, text="Важно помнить: этот тест создан для ориентировки и не может гарантировать абсолютную точность. Он призван помочь вам определить свой примерный уровень и наметить дальнейшие шаги")
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(types.KeyboardButton("Вернуться в главное меню"))
        filee=open('./urovni.jpg','rb')
        bot.send_message(message.chat.id, " Теперь пришло время вспомнить все набранные вами очки, просуммировать их и свериться с приведенной ниже таблицей, чтобы определить ваш приблизительный уровень")
        bot.send_photo(message.chat.id, filee)
        bot.send_message(message.chat.id,  "Поздравляем с завершением! Помните, что каждый результат — это шаг вперед, и всегда есть к чему стремиться. Удачи в обучении!", reply_markup=markup)

#################
    elif message.text == "Поиск книг":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("A2"),
            types.KeyboardButton("B1"),
            types.KeyboardButton("B2"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите свой уровень владения английским языком", reply_markup=markup)





#####уровни пошли
    elif message.text == "B2":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Классика и современная проза"),
            types.KeyboardButton("Научно-популярная литература"),
            types.KeyboardButton("Детектив и триллер"),
            types.KeyboardButton("Фэнтези с элементами романтики"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите привлекающий вас жанр произведений", reply_markup=markup)

    elif message.text == "Классика и современная проза":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("12+"),
            types.KeyboardButton("16+"),
            types.KeyboardButton("18+"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите ваш возраст", reply_markup=markup)

    elif message.text == "12+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=("Могу предложить вам следующие книги:"
                              "\n"
                              "\n"
                             "[The Adventures of Tom Sawyer](https://clc.li/jgUnm)\n\n"
                             "[The Jungle Book](https://clc.li/InWFX)\n\n"
                              "[Alice’s Adventures in Wonderland](https://clc.li/fzrnk)\n\n"
                               "[Little Women](https://clc.li/kPKap)\n\n"
                               "[The Call of the Wild](https://clc.li/oZrvC)\n\n"
                               "[The Secret Garden](https://clc.li/HrRXC)\n\n"
                               "[Anne of Green Gables](https://clc.li/sEPlT)"
                        ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True

                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.", reply_markup=markup)

    elif message.text == "16+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Pride and Prejudice](https://clc.li/gXwWj)\n\n"
                             "[Jane Eyre](https://clc.li/XkMEV)\n\n"
                             "[Wuthering Heights](https://clc.li/MwoEw)\n\n"
                             "[Great Expectations](https://clc.li/dCjeJ)\n\n"
                             "[The Scarlet Letter](https://clc.li/lcGjS)\n\n"
                             "[Frankenstein](https://clc.li/qgCmv)\n\n"
                             "[The Picture of Dorian Gray](https://clc.li/PoLLC)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True

                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)

    elif message.text == "18+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Great Gatsby](https://clc.li/FDAuX)\n\n"
                             "[One Hundred Years of Solitude](https://clc.li/AQkdm)\n\n"
                             "[Madame Bovary](https://clc.li/Adwya)\n\n"
                             "[Tender Is the Night](https://clc.li/aXylj)\n\n"
                             "[Heart of Darkness](https://clc.li/kwJAL)\n\n"
                             "[The Sound and the Fury](https://clc.li/KpfkN)\n\n"
                             "[The Age of Innocence](https://clc.li/uwThv)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True

                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)

    elif message.text == "Научно-популярная литература":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("𝟏𝟐+"),
            types.KeyboardButton("𝟏𝟲+"),
            types.KeyboardButton("𝟏𝟖+"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите ваш возраст", reply_markup=markup)

    elif message.text == "𝟏𝟐+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Bomb: The Race to Build—and Steal—the World’s Most Dangerous Weapon](https://clc.li/suttc)\n\n"
                             "[The Boy Who Harnessed the Wind](https://clc.li/diEoD)\n\n"
                             "[Astrophysics for People in a Hurry](https://clc.li/hjaxd)\n\n"
                             "[The Emperor of All Maladies: A Biography of Cancer](https://clc.li/wDjqe)\n\n"
                             "[The Immortal Life of Henrietta Lacks](https://clc.li/evcwE)\n\n"
                             "[Hidden Figures: The American Dream and the Untold Story of the Black Women Mathematicians Who Helped Win the Space Race](https://clc.li/sIWXL)\n\n"
                             "[The Book Thief](https://clc.li/JbmgA)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True

                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "𝟏𝟲+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Sapiens: A Brief History of Humankind](https://clc.li/PwTrt)\n\n"
                             "[Thinking, Fast and Slow](https://clc.li/LywNe)\n\n"
                             "[Cosmos](https://clc.li/FKmvy)\n\n"
                             "[The Selfish Gene](https://clc.li/JDkFm)\n\n"
                             "[Guns, Germs, and Steel: The Fates of Human Societies](https://clc.li/FImuN)\n\n"
                             "[Bad Blood: Secrets and Lies in a Silicon Valley Startup](https://clc.li/OvnNK)\n\n"
                             "[The Sixth Extinction: An Unnatural History](https://clc.li/VqACv)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True

                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "𝟏𝟖+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[A Brief History of Time](https://clc.li/qdRNx)\n\n"
                             "[The Black Swan: The Impact of the Highly Improbable](https://clc.li/mxEth)\n\n"
                             "[Origin: Fourteen Billion Years of Cosmic Evolution](https://clc.li/oPgQJ)\n\n"
                             "[The Gene: An Intimate History](https://clc.li/OEKLI)\n\n"
                             "[Why Nations Fail: The Origins of Power, Prosperity, and Poverty](https://clc.li/zEgHG)\n\n"
                             "[Chaos: Making a New Science](https://clc.li/vAseF)\n\n"
                             "[Entangled Life: How Fungi Make Our Worlds, Change Our Minds & Shape Our Futures](https://clc.li/OTaUJ)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "Детектив и триллер":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("͎𝟙𝟚+"),
            types.KeyboardButton("͎𝟙𝟞+"),
            types.KeyboardButton("͎𝟙𝟠+"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите ваш возраст", reply_markup=markup)

    elif message.text == "͎𝟙𝟚+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Mysterious Affair at Styles](https://clc.li/NUazr)\n\n"
                             "[The Hound of the Baskervilles](https://clc.li/EDyFL)\n\n"
                             "[And Then There Were None](https://clc.li/qrhFh)\n\n"
                             "[The Da Vinci Code](https://clc.li/wrULT)\n\n"
                             "[The Girl with the Dragon Tattoo](https://clc.li/dcsIk)\n\n"
                             "[The No. 1 Ladies’ Detective Agency](https://clc.li/tExcW)\n\n"
                             "[The Secret Adversary](https://clc.li/JLIkZ)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "͎𝟙𝟞+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Maze Runner](https://clc.li/SUJvb)\n\n"
                             "[The Hunger Games](https://clc.li/GdPZp)\n\n"
                             "[Miss Peregrine’s Home for Peculiar Children](https://clc.li/DKVoC)\n\n"
                             "[The Dry](https://clc.li/BXNNl)\n\n"
                             "[The Woman in Cabin 10](https://clc.li/XsnqR)\n\n"
                             "[Big Little Lies](https://clc.li/qUWie)\n\n"
                             "[The Silent Patient](https://clc.li/Csplf)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "͎𝟙𝟠+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Silence of the Lambs](https://clc.li/LhnUr)\n\n"
                             "[The Secret History](https://clc.li/jiyfb)\n\n"
                             "[Sharp Objects](https://clc.li/RiFKT)\n\n"
                             "[The Girl Who Kicked the Hornets’ Nest](https://clc.li/RzNjM)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "Фэнтези с элементами романтики":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("12+ㅤ"),
            types.KeyboardButton("16+ㅤ"),
            types.KeyboardButton("18+ㅤ"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id, text="Пожалуйста, выберите ваш возраст", reply_markup=markup)



    elif message.text == "12+ㅤ":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Woodwalkers](https://clc.li/Ygmoi)\n\n"
                             "[A Darker Shade of Magic](https://clc.li/PbbkR)\n\n"
                             "[The Bear and the Nightingale](https://clc.li/idHRn)\n\n"
                             "[The School For Good And Evil](https://clc.li/iqSZf)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)



    elif message.text == "16+ㅤ":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Cruel Prince](https://clc.li/zEJch)\n\n"
                             "[A Darker Shade of Magic](https://clc.li/CVYyS)\n\n"
                             "[The Bear and the Nightingale](https://clc.li/idHRn)\n\n"
                             "[Shadow and Bone](https://clc.li/UsSQJ)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "18+ㅤ":

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

        markup.add(

            types.KeyboardButton("Вернуться в главное меню"),

        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[From Blood and Ash](https://clc.li/Wyjvf)\n\n"
                             "[Ninth House](https://clc.li/gHMdz)\n\n"
                             "[The Serpent and the Wings of Night](https://clc.li/OjUBI)\n\n"
                             "[Tender Is the Night](https://clc.li/OTlWe)\n\n"
                             "[A Court of Mist and Fury](https://clc.li/SdLAg)\n\n"
                             "[A Ruin of Roses](https://clc.li/DJPyo)\n\n"
                             "[Neon Gods](https://clc.li/ghiBT)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "B1":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Классика и современная проза."),
            types.KeyboardButton("Научно-популярная литература."),
            types.KeyboardButton("Детектив и триллер."),
            types.KeyboardButton("Фэнтези с элементами романтики."),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите привлекающий вас жанр произведений", reply_markup=markup)

    elif message.text == "Классика и современная проза.":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("ㅤ12+"),
            types.KeyboardButton("ㅤ16+"),
            types.KeyboardButton("ㅤ18+"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите ваш возраст", reply_markup=markup)

    elif message.text == "ㅤ12+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Oliver Twist](https://clc.li/IPzwo)\n\n"
                             "[White Fang](https://clc.li/kzCIY)\n\n"
                             "[The Secret Garden](https://clc.li/SGQHg)\n\n"
                             "[Number the Stars](https://clc.li/anKbV)\n\n"
                             "[The Boy in the Striped Pyjamas](https://clc.li/jjxQb)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "ㅤ16+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Catcher in the Rye](https://clc.li/hkpyM)\n\n"
                             "[Fahrenheit 451](https://clc.li/WYMlc)\n\n"
                             "[Lord of the Flies](https://clc.li/USsfc)\n\n"
                             "[The Curious Incident of the Dog in the Night-Time](https://clc.li/Hqlvy)\n\n"
                             "[Life of Pi](https://clc.li/lpOVl)\n\n"
                             "[The Fault in Our Stars](https://clc.li/qLzzT)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "ㅤ18+":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Animal Farm](https://clc.li/EjLtu)\n\n"
                             "[The Perks of Being a Wallflower](https://clc.li/ydyUV)\n\n"
                             "[The Handmaid’s Tale](https://clc.li/aqZfy)\n\n"
                             "[Fight Club](https://clc.li/zHfBv)\n\n"
                             "[Convenience Store Woman](https://clc.li/NZlZD)\n\n"
                             "[High Fidelity](https://clc.li/TcYew)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)

    elif message.text == "Научно-популярная литература.":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("1̃2̃+̃"),
            types.KeyboardButton("1̃6̃+̃"),
            types.KeyboardButton("1̃8̃+̃"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите ваш возраст", reply_markup=markup)

    elif message.text == "1̃2̃+̃":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Royal Children of English History](https://clc.li/nYFyg)\n\n"
                             "[A Briefer History of Time](https://clc.li/prruG)\n\n"
                             "[Who Was Albert Einstein?](https://clc.li/pjUWW)\n\n"
                             "[What If?](https://clc.li/qkgUO)\n\n"
                             "[A Really Short History of Nearly Everything: Bill Bryson](https://clc.li/UKSlM)\n\n"
                             "[The Boy Who Harnessed the Wind (William Kamkwamba)](https://clc.li/ASxmo)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "1̃6̃+̃":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[How Not to Be Wrong: The Power of Mathematical Thinking](https://clc.li/kFiAB)\n\n"
                             "[Stiff: The Curious Lives of Human Cadavers](https://clc.li/wNzKf)\n\n"
                             "[Hidden Figures](https://clc.li/KwAjt)\n\n"
                             "[The Last Lecture](https://clc.li/AIjMf)\n\n"
                             "[Man’s Search for Meaning](https://clc.li/gZubG)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)



    elif message.text == "1̃8̃+̃":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Sapiens: A Brief History of Humankind](https://clc.li/jrESn)\n\n"
                             "[Why do we sleep? The new science of healthy sleep and dreams](https://clc.li/eLYjf)\n\n"
                             "[Steve Jobs: Walter Isaacson](https://clc.li/opZCw)\n\n"
                             "[Quiet: The Power of Introverts in a World That Can’t Stop Talking](https://clc.li/ofSru)\n\n"
                             "[Into the Wild](https://clc.li/hBwEh)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "Детектив и триллер.":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("~1̃2̃+̃"),
            types.KeyboardButton("~1̃6̃+̃"),
            types.KeyboardButton("~1̃8̃+̃"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите ваш возраст", reply_markup=markup)

    elif message.text == "~1̃2̃+̃":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Escape from Mr. Lemoncello’s Library](https://clc.li/JjofS)\n\n"
                             "[Truly Devious](https://clc.li/WaEDw)\n\n"
                             "[The Mysterious Benedict Society](https://clc.li/YpCMB)\n\n"
                             "[Rebecca](https://clc.li/STofw)\n\n"
                             "[The Phantom of the Opera](https://clc.li/XGssv)\n\n"
                             "[The Canterville Ghost](https://clc.li/onEyU)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "~1̃6̃+̃":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Woman in Black](https://clc.li/ZLsYK)\n\n"
                             "[Truly Devious](https://clc.li/KQnFS)\n\n"
                             "[A Good Girl’s Guide to Murder](https://clc.li/mrruU)\n\n"
                             "[Shiver](https://clc.li/vABSw)\n\n"
                             "[Code Name Verity](https://clc.li/dBTlz)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)

    elif message.text == "~1̃8̃+̃":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Ten](https://clc.li/wPDZw)\n\n"
                             "[The Inheritance Games](https://clc.li/qbOxd)\n\n"
                             "[Gone Girl](https://clc.li/wRuQO)\n\n"
                             "[The Girl on the Train](https://clc.li/scbmf)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "Фэнтези с элементами романтики.":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("1̃2̃🞡"),
            types.KeyboardButton("1̃6̃🞡"),
            types.KeyboardButton("1̃8̃🞡"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id, text="Пожалуйста, выберите ваш возраст", reply_markup=markup)

    elif message.text == "1̃2̃🞡":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Beauty and the Beast](https://clc.li/yhyGW)\n\n"
                             "[Ella Enchanted](https://clc.li/RlSXL)\n\n"
                             "[An Ember in the Ashes](https://clc.li/BOIkl)\n\n"
                             "[A Wrinkle in Time](https://clc.li/puCPt)\n\n"
                             "[Inkheart](https://clc.li/jDWJF)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)



    elif message.text == "1̃6̃🞡":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Cinder](https://clc.li/pCtsZ)\n\n"
                             "[Graceling](https://clc.li/GeIxS)\n\n"
                             "[Throne of Glass](https://clc.li/zSxgY)\n\n"
                             "[Red Queen](https://clc.li/ZkTXS)\n\n"
                             "[The Wrath and the Dawn](https://clc.li/QfEFI)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)
    elif message.text == "1̃8̃🞡":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Stardust](https://clc.li/jgiPP)\n\n"
                             "[Serpent & Dove](https://clc.li/aGwXh)\n\n"
                             "[Six of Crows](https://clc.li/LJyzB)\n\n"
                             "[Carmilla](https://clc.li/DTudA)\n\n"
                             "[Neverwhere](https://clc.li/XwpwZ)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)





















# PRODOLCHIM
    elif message.text == "A2":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("`Классика и современная проза`"),
            types.KeyboardButton("`Детектив и триллер`"),
            types.KeyboardButton("`Фэнтези с элементами романтики`"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите привлекающий вас жанр произведений", reply_markup=markup)

    elif message.text == "`Классика и современная проза`":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("1̂2̂+̂"),
            types.KeyboardButton("1̂6̂+̂"),
            types.KeyboardButton("1̂8̂+̂"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите ваш возраст", reply_markup=markup)

    elif message.text == "1̂2̂+̂":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Pussy and Doggy Tales](https://clc.li/hZWYq)\n\n"
                             "[The Green Fairy Book](https://clc.li/HLyLl)\n\n"
                             "[More Beasts (For Wors Children)](https://clc.li/jmvdv)\n\n"
                             "[The Velveteen Rabbit](https://clc.li/Cygpl)\n\n"
                             "[The Jungle Book (адаптированная версия)](https://clc.li/lNtAN)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "1̂6̂+̂":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Eight Cousins](https://clc.li/iiOFr)\n\n"
                             "[Three Men in a Boat](https://clc.li/xzXfD)\n\n"
                             "[The Giver](https://clc.li/gnznE)\n\n"
                             "[Wonder](https://clc.li/XWgyV)\n\n"
                             "[Stargirl](https://clc.li/lYGSg)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "1̂8̂+̂":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Holes](https://clc.li/MulQE)\n\n"
                             "[13 Reasons Why](https://clc.li/wsogS)\n\n"
                             "[The Perks of Being a Wallflower](https://clc.li/PZZUz)\n\n"
                             "[The Stranger](https://clc.li/dizcm)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)

    elif message.text == "`Детектив и триллер`":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("1̇2̇✛"),
            types.KeyboardButton("1̇6̇✛"),
            types.KeyboardButton("1̇8̇✛"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text="Пожалуйста, выберите ваш возраст", reply_markup=markup)

    elif message.text == "1̇2̇✛":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Mysterious Benedict Society](https://clc.li/HBjbA)\n\n"
                             "[The Westing Game](https://clc.li/XStLW)\n\n"
                             "[Gone with the Wind (адаптированная версия)](https://clc.li/EANcg)\n\n"
                             "[The Lord of the Rings](https://clc.li/KALtW)\n\n"
                             "[Howl’s Moving Castle](https://clc.li/zGliT)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)

    elif message.text == "1̇6̇✛":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Secret Garden (адаптированная)](https://clc.li/JWPut)\n\n"
                             "[Mystery of the Roman Villa](https://clc.li/afINf)\n\n"
                             "[The Lottery](https://clc.li/KCryP)\n\n"
                             "[The Case of the Missing Cat](https://clc.li/clmNj)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "1̇8̇✛":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Turn of the Screw (адаптированная)](https://clc.li/DYRzT)\n\n"
                             "[Rosemary's Baby](https://clc.li/gfSMe)\n\n"
                             "[The Stepford Wives](https://clc.li/Oweqb)\n\n"
                             "[The Mystery of the Blue Diamond (адаптированная)](https://clc.li/TJbrI)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)


    elif message.text == "`Фэнтези с элементами романтики`":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("1͛2͛+͛"),
            types.KeyboardButton("1͛6͛+͛"),
            types.KeyboardButton("1͛8͛+͛"),
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id, text="Пожалуйста, выберите ваш возраст", reply_markup=markup)

    elif message.text == "1͛2͛+͛":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Tale of Despereaux](https://clc.li/mlZVQ)\n\n"
                             "[The Golden Compass](https://clc.li/PoSBW)\n\n"
                             "[Fablehaven by Brandon Mull](https://clc.li/cMfzi)\n\n"
                             "[The Princess Bride by William Goldman](https://clc.li/oHGXe)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)

    elif message.text == "1͛6͛+͛":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[The Moonstone (адаптированная)](https://clc.li/UfqES)\n\n"
                             "[The Goblin Emperor](https://clc.li/WmkgB)\n\n"
                             "[A Darker Shade of Magic](https://clc.li/octWZ)\n\n"
                             "[Graceling (адаптированная)](https://clc.li/GeIxS)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)
    elif message.text == "1͛8͛+͛":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("Вернуться в главное меню"),
        )
        bot.send_message(message.chat.id,
                         text=(
                             "Могу предложить вам следующие книги:\n\n"
                             "[Ghost Ship Story](https://clc.li/QIyXg)\n\n"
                             "[Graceling (адаптированная)](https://clc.li/GeIxS)\n\n"
                            "[Harry Potter and the Prisoner of Azkaban](https://clc.li/lZnAn)"
                         ),
                         parse_mode='MarkdownV2',
                         disable_web_page_preview=True
                         )
        bot.send_message(message.chat.id,
                         text="Узнайте больше: нажмите на название любой книги, чтобы увидеть ее описание.",
                         reply_markup=markup)



# КОНЕЦ


    if message.text == "Анализ текста":
        bot.send_message(message.chat.id, 'Пришлите текст, уровень которого вы хотите узнать')
        bot.register_next_step_handler(message, analyze_text_handler)
    # else:
    #     bot.send_message(message.chat.id, "Пожалуйста, выберите действие из меню.")

#HJKFGDHSJFHJDSHJFDSJHLFJHDSJKHFHJKSDJHF
def analyze_text_with_dictionary(words, level_word_sets, bot, chat_id):
    """
    Анализирует текст, учитывая только слова, найденные в словарях.
    Выводит сообщение о количестве слов, которых не было в словарях.
    """
    unknown_words_count = 0
    known_words = []

    for word in words:
        found = False
        for lvl in level_word_sets:
            if word in level_word_sets[lvl]:
                found = True
                known_words.append(word)
                break
        if not found:
            unknown_words_count += 1

    bot.send_message(
        chat_id,
        f"В словарях не было найдено: {unknown_words_count} слов"
    )

    return known_words





# Обработка анализа текста
def analyze_text_handler(message):
    text = message.text

    # Анализ по смыслу
    semantic_level = analyze_text_semantic(text)

    # Анализ по словарям
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    level_counts = {"A1": 0, "A2": 0, "B1": 0}
    unknown_words_count = 0

    for word in words:
        found = False
        for lvl in ["A1", "A2", "B1"]:
            if word in level_word_sets[lvl]:
                level_counts[lvl] += 1
                found = True
                break
        if not found:
            # слово не в словаре — игнорируем при определении уровня
            unknown_words_count += 1

    # Определение уровня по найденным словам
    def определить_уровень(level_counts):
        levels_order = {"A1": 1, "A2": 2, "B1": 3}
        total_found_words = sum(level_counts.values())
        if total_found_words == 0:
            return "A1"  # или другой уровень по умолчанию, если ничего не найдено
        sorted_counts = sorted(level_counts.items(), key=lambda x: x[1], reverse=True)
        top_level, top_count = sorted_counts[0]
        second_level, second_count = sorted_counts[1]
        difference = abs(levels_order[top_level] - levels_order[second_level])
        if total_found_words > 70:
            if difference <= 15:
                if levels_order[top_level] > levels_order[second_level]:
                    return top_level
                else:
                    return second_level
            else:
                return top_level
        else:
            if difference <= 5:
                if levels_order[top_level] > levels_order[second_level]:
                    return top_level
                else:
                    return second_level
            else:
                return top_level

    average_level = определить_уровень(level_counts)

    # Анализ по длине слов
    # percents = calculate_word_length_percentages(words)
    # length_word_level = get_level_by_percentage_value(percents['1-3'], length_word_levels)
    def analyze_word_length_category(words):
        total = len(words)
        if total == 0:
            return "A1"  # если слов нет, возвращаем уровень по умолчанию

        count_short = sum(1 for w in words if len(w) < 3)
        count_medium = sum(1 for w in words if 3 <= len(w) <= 6)
        count_long = sum(1 for w in words if len(w) > 6)

        perc_short = (count_short / total) * 100
        perc_medium = (count_medium / total) * 100
        perc_long = (count_long / total) * 100

        # Определение преобладающей категории
        if perc_short >= 50:
            return "A1"
        elif perc_medium >= 50:
            return "A2"
        elif perc_long >= 50:
            return "B1"
        else:
            max_perc = max(perc_short, perc_medium, perc_long)
            if max_perc == perc_short:
                return "A1"
            elif max_perc == perc_medium:
                return "A2"
            else:
                return "B1"
    length_word_level = analyze_word_length_category(words)
    if length_word_level is None:
        length_word_level = "A1"  # дефолтное значение

    # Категория длины предложения
    total_sentences = len(re.findall(r'[.!?]', text)) + 1
    perc_words_in_sentence = (total_words / total_sentences) if total_sentences else total_words
    sentence_length_category = get_level_by_percentage_value(perc_words_in_sentence, sentence_levels)
    if sentence_length_category is None:
        sentence_length_category = "A1"

    # Итоговые уровни
    levels_order = {"A1": 1, "A2": 2, "B1": 3}

    # Итоговый уровень по смыслу
    semantic_level_final = analyze_text_semantic(text)

    # Преобразуем уровни в числа
    try:
        semantic_num = levels_order[semantic_level_final]
    except KeyError:
        semantic_num = levels_order["A1"]
    try:
        length_word_num = levels_order[length_word_level]
    except KeyError:
        length_word_num = levels_order["A1"]
    try:
        sentence_num = levels_order[sentence_length_category]
    except KeyError:
        sentence_num = levels_order["A1"]

    avg_level_num = (semantic_num + length_word_num + sentence_num) / 3

    # Находим ближайший уровень
    closest_level = min(levels_order.items(), key=lambda x: abs(x[1] - avg_level_num))[0]

    # Формируем сообщение
    result_message = (
            f"В тексте {total_words} слов.\n"
            f"Слов не найдено в словарях: {unknown_words_count}\n"
            f"Распределение по уровням:\n"
            # + "\n".join([f"{lvl}: {count}" for lvl, count in level_counts.items()])
            + f"\nСредний уровень (по словарям): {average_level}"
            + f"\nКатегория длины слова: {length_word_level}"
            + f"\nКатегория длины предложения: {sentence_length_category}"
            + f"\nИтоговый уровень: {closest_level}"
    )

    bot.send_message(message.chat.id, result_message)

# Запуск бота
bot.polling(none_stop=True, interval=0)