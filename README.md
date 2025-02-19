# 🌳 Tree Health Classification

## 📌 Описание проекта

Этот проект решает задачу классификации состояния деревьев (Good, Fair, Poor) по данным из NY 2015 Street Tree Census с помощью глубокой нейросети (MLP).

## 📊 Данные

Источник: NYC Open Data – 2015 Street Tree Census

Формат: Табличные данные (CSV)

Целевая переменная: health (Good / Fair / Poor)

Количество признаков: 19 (погодные, географические, категориальные и числовые характеристики)

## 📌 Обоснование выбора архитектуры MLP
Для решения задачи классификации состояния деревьев (3 класса: Good, Fair, Poor) была выбрана многослойная перцептронная (MLP) модель.
Ниже приведены основные аргументы в пользу этого выбора:

🔹 Почему MLP?
✅ Подходит для табличных данных: В отличие от сверточных (CNN) или рекуррентных (RNN) сетей, MLP хорошо работает с табличными признаками.
✅ Обрабатывает числовые и категориальные признаки: Полносвязные слои позволяют эффективно обучать зависимости в данных.

🔹 Архитектура модели
4 полносвязных слоя – достаточная глубина для улавливания сложных зависимостей, но без избыточной сложности.
Dropout – предотвращает переобучение за счет случайного отключения нейронов во время обучения.
Активация ReLU – ускоряет сходимость модели, устраняя проблему исчезающего градиента.
Функция потерь CrossEntropyLoss – стандартный выбор для многоклассовой классификации.
Оптимизатор Adam – адаптивно настраивает скорости обучения, улучшая сходимость.
🔹 Итог
MLP – это простой, но эффективный выбор для классификации деревьев по табличным данным. Архитектура модели сбалансирована для точности и скорости обучения.

## 🚀 Установка и запуск

🔹 Скопируйте репозиторий
 ```bash
   git clone https://github.com/SergeyGermanovichML/NY_2015_Street_tree_census.git
   cd NY_2015_Street_tree_census
   ```
🔹 Установка зависимостей и запуск app
```bash
   python -m venv venv
   source venv/bin/activate  # для Windows используйте: venv\Scripts\activate
   pip install -r requirements.txt
   python app.py
   ```
🔹 Скачайте и переместите датасет в папку с ноунбуком EDA.ipynb 
https://www.kaggle.com/datasets/new-york-city/ny-2015-street-tree-census-tree-data

## 📡 Использование API

🔹 Запрос (GET)

{
    "tree_dbh": 27,
    "curb_loc_OnCurb": 1,
    "steward_3or4": 0,
    "steward_4orMore": 0,
    "steward_No": 1,
    "guards_Helpful": 1,
    "guards_No": 0,
    "guards_Unsure": 0,
    "sidewalk_NoDamage": 0,
    "root_stone_Yes": 1,
    "root_grate_Yes": 1,
    "trunk_wire_Yes": 0,
    "trnk_light_Yes": 0,
    "brch_light_Yes": 0,
    "brch_other_Yes": 0,
    "borough_Brooklyn": 1,
    "borough_Manhattan": 0,
    "borough_Queens": 0,
    "borough_Staten_Island": 0 
    }

🔹 Ответ

{

  "prediction": "Good"
  
}
