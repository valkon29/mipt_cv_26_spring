# Проект “Интеллектуальная система улучшения изображений”
Все отчёты можно найти в папке `reports`

Итоговый отчёт приведён в файлах `final_report.md` и `final_report.pdf`

Промежуточный отчёт соответственно в `old_report.md` и `old_report.pdf`


Структура проекта:
```
.
├── datasets.py                       # Деление на train/val/test
├── 01_baseline_implementation.ipynb  # Классические методы
├── 02_static_lut.ipynb               # Статический LUT
├── 03_adaptive_lut.ipynb             # Custom CNN + LUT
├── 05_mobilenet_lut.ipynb            # MobileNetV2 + LUT
├── 06_metrics_comparison.ipynb       # Сводное сравнение всех методов
├── data_analysis.ipynb               # анализ датасета
├── mit_adobe_5k_dataset/             # Датасет
│   ├── raw/                          # 5000 RAW-изображений
│   └── c/                            # 5000 изображений эксперта C
├── baseline_results/                 # Результаты классических методов
├── static_lut_results/               # Результаты статического LUT
├── adaptive_lut_results/             # Результаты адаптивного LUT
├── mobilenet_lut_results/            # Результаты MobileNetV2+LUT
├── comparison_results/               # Сводные графики сравнения
├── README.md
└── reports/                          # Отчёты
```
