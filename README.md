# FERMATRICA Flexible Econometrics Framework. Reporting

[_Russian version below_](#RU)

Reporting and post-modeling for econometric models built using FERMATRICA.

### 1. Key Ideas

FERMATRICA_REP encompasses all reporting and business analysis functionality for an already built model. (The model is constructed using FERMATRICA.) The main blocks are:

1. Model exploration - actual data and forecasts, decomposition into factors, statistical outputs, and efficiency curves. For most of this type of functionality, historical data is sufficient; only some components require an understanding of future periods (`fermatrica_rep.curves_full)`.
2. Forecasting the future - KPI forecasts for future periods under specified conditions. Translating annual budgets into marketing variables, calculating summaries for investment scenarios, optimizing budgets against specified KPIs.
3. Visualization and export of results - tables, `plotly` graphs, export to XLSX and PPTX. 

The functionality of FERMATRICA_REP can be used independently or through the FERMATRICA_DASH dashboard (reporting logic is encapsulated within FERMATRICA_REP).

### 2. Components

The repository includes modules responsible for reporting and post-modeling for econometric models built using FERMATRICA.

- Standard reporting components
  - Metrics and statistical outputs (`fermatrica_rep.stats`)
  - Fit (actual data and forecasts) (`fermatrica_rep.fit`, `fermatrica_rep.category`)
  - Decomposition over time (`fermatrica_rep.decomposition`)
  - Decomposition over a period (waterfall) (`fermatrica_rep.waterfall`)
  - Efficiency curves (`fermatrica_rep.curves`, `fermatrica_rep.curves_full`)
  - Transformations (`fermatrica_rep.transformation`)
- Calculation of investment options/scenarios (`fermatrica_rep.options`)
- Budget optimizer (`fermatrica_rep.options.optim`)
- Export model to XLSX (`fermatrica_rep.model_exp`)
- Standard slide generator in PPTX (`fermatrica_rep.reporting`)

### 3. Installation

To facilitate work, it is recommended to install all components of the FERMATRICA framework. It is assumed that work will be conducted in PyCharm (VScode is OK also for sure).

1. Create a Python virtual environment of your choice (Anaconda, Poetry, etc.) or use a previously created one. It makes sense to establish a separate virtual environment for econometric tasks and for every new version of FERMATRICA.
    1. Mini-guide on virtual environments (external): https://blog.sedicomm.com/2021/06/29/chto-takoe-venv-i-virtualenv-v-python-i-kak-ih-ispolzovat/
    2. For framework version v010, let the virtual environment be named FERMATRICA_v010.
2. Clone the FERMATRICA repositories to a location of your choice.
    ```commandline
    cd [my_fermatrica_folder]
    git clone https://github.com/FERMATRICA/fermatrica_utils.git 
    git clone https://github.com/FERMATRICA/fermatrica.git
    git clone https://github.com/FERMATRICA/fermatrica_rep.git 
    ```
   1. To work with the interactive dashboard: _coming soon_
    2. For preliminary data work: _coming soon_
3. In each of the repositories, select the FERMATRICA_v010 environment (FERMATRICA_v020, FERMATRICA_v030, etc.) through `Add interpreter` in the PyCharm interface.
4. Install all cloned packages except FERMATRICA_DASH using pip install.
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    pip install .
    ```
   1. Instead of navigating to each project's folder, you can specify the path to it in pip install:
       ```commandline
       pip install [path_to_fermatrica_part]
       ```
5. If necessary, install third-party packages/libraries required for the functioning of FERMATRICA using `conda install` or `pip install`. To update versions of third-party packages, use `conda update` or `pip install -U`.

>FERMATRICA_REP may require installed MS Office for proper functionality. This is due to the complexity of exporting to the PPTX format. Operation on systems other than Windows is not guaranteed.

-------------------------------------

<a name="RU"></a>
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
3. В каждом из репозиториев выберите среду FERMATRICA_v010 (FERMATRICA_v020, FERMATRICA_v030 и т.д.) через `Add interpreter` (в интерфейсе PyCharm)
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

