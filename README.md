# FERMATRICA Flexible Econometrics Framework. Reporting

Репортинг и постмоделлинг для эконометрических моделей, построенных при помощи FERMATRICA.

### 1. Ключевые идеи

FERMATRICA_REP - это весь функционал отчётности и бизнес-анализа по уже построенной модели. (Модель строится при помощи FERMATRICA.) Выделим основные блоки:

1. Исследование модели - факт и прогноз, декомпозиция на факторы, статистические выкладки и кривые эффективности. Для большей части функционала этого типа достаточно исторических данных, только некоторые части требуют представления о будущих периодах (`fermatrica_rep.curves_full`).
2. Прогнозирование будущего - прогноз KPI на будущие периоды при заданных условиях. Трансляция годовых бюджетов в маркетинговые переменные, расчёт резюме (summary) по сценарию инвестирования, оптимизация бюджетов под заданный KPI.
3. Визуализация и экспорт результатов - таблицы, графики `plotly`, экспорт в XLSX и PPTX.

> Функционал FERMATRICA_REP может использоваться самостоятельно или через дашборд FERMATRICA_DASH (логика репортинга вынесена в FERMATRICA_REP).

### 2. Состав

Репозиторий включает модули, отвечающие за репортинг и постмоделлинг для эконометрических моделей, построенных при помощи FERMATRICA.

- Стандартные компоненты репортинга
  - Метрики и статистический вывод (`fermatrica_rep.stats`)
  - Фит (факт и прогноз) (`fermatrica_rep.fit`, `fermatrica_rep.category`)
  - Декомпозиция в динамике (`fermatrica_rep.decomposition`)
  - Декомпозиция за период (водопад) (`fermatrica_rep.waterfall`)
  - Кривые эффективности (`fermatrica_rep.curves`, `fermatrica_rep.curves_full`)
  - Трансформации (`fermatrica_rep.transformation`)
- Просчёт опций / сценариев инвестирования (`fermatrica_rep.options`)
- Оптимизатор бюджетов (`fermatrica_rep.options.optim`)
- Экспорт модели в XLSX (`fermatrica_rep.model_exp`)
- Генератор стандартных слайдов в PPTX (`fermatrica_rep.reporting`)

### 3. Установка

Для корректной работы рекомендуется установить все составляющие фреймворка FERMATRICA. Предполагается, что работа будет вестись в PyCharm

1. Создайте виртуальную среду Python удобного для вас типа (Anaconda, Poetry и т.п.) или воспользуйтесь ранее созданной. Имеет смысл завести отдельную виртуальную среду для эконометрических задач и для каждой новой версии FERMATRICA
   - Мини-гайд по виртуальным средам (внешний): https://blog.sedicomm.com/2021/06/29/chto-takoe-venv-i-virtualenv-v-python-i-kak-ih-ispolzovat/
   - Для версии фреймворка v010 пусть виртуальная среда называется FERMATRICA_v010
2. Клонируйте в удобное для вас место репозитории FERMATRICA
    ```commandline
    cd [my_fermatrica_folder]
    git clone https://github.com/FERMATRICA/fermatrica_utils.git 
    git clone https://github.com/FERMATRICA/fermatrica.git
    git clone https://github.com/FERMATRICA/fermatrica_rep.git 
    ```
    - Для работы с интерактивным дашбордом FERMATRICA_DASH: _coming soon_
    - Для предварительной работы с данными FERMATRICA_DATA: _coming soon_
3. В каждом из репозиториев выберите среду FERMATRICA_v010 (FERMATRICA_v020, FERMATRICA_v030 и т.д.) через `Add interpreter` (в интерфейсе PyCharm) и переключитесь в соответствующую ветку гита
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    git checkout v010 [v020, v030...]
    ```
4. Установите все склонированные пакеты, кроме FERMATRICA_DASH, используя `pip install`
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    pip install .
    ```
   - Вместо перехода в папку каждого проекта можно указать путь к нему в `pip install`
   ```commandline
   pip install [path_to_fermatrica_part]
   ```
5. При необходимости, поставьте сторонние пакеты / библиотеки Python, которые требуются для функционирования FERMATRICA, используя `conda install` или `pip install`. Для обновления версий сторонних пакетов используйте `conda update` или `pip install -U`

> FERMATRICA_REP для корректной работы может потребоваться установленный MS Office. Это связано со сложным экспортом в формат PPTX. Работа на системах, отличных от Windows, не гарантируется.

