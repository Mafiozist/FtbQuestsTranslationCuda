import os
import re
import concurrent.futures
from pathlib import Path
from functools import lru_cache

import torch
from transformers import MarianMTModel, MarianTokenizer
import tqdm

# Определяем, доступен ли GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@lru_cache(maxsize=4)
def get_translator(lang_to: str):
    """
    Загружает модель и токенизатор для перевода с английского на целевой язык.
    Модель перемещается на GPU (если доступен).
    """
    model_name = f'Helsinki-NLP/opus-mt-en-{lang_to}'
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_to(text: str, lang_to: str) -> str:
    """
    Переводит переданный текст целиком с помощью модели MarianMT.
    Выводит исходный и переведённый варианты для отладки.
    """
    model, tokenizer = get_translator(lang_to)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Перевод:\n  Исходное: {text}\n  Переведённое: {translated_text}\n")
    return translated_text

def safe_translate(text: str, lang_to: str) -> str:
    """
    Безопасно переводит текст для полей title, subtitle, description.
    Алгоритм:
      1. Извлекаем все форматирующие коды (например, &4, &e, &r) и заменяем их на плейсхолдеры (<<<1>>>, <<<2>>> и т.д.).
      2. Переводим очищенный текст целиком.
      3. Восстанавливаем исходные форматирующие коды на места плейсхолдеров.
    При этом выводятся исходный и переведённый тексты для отладки.
    """
    # Найдем все форматирующие коды в порядке появления
    codes = re.findall(r'(&[0-9a-zA-Z]+)', text)
    
    # Функция замены для создания уникальных плейсхолдеров
    count = 1
    def repl(match):
        nonlocal count
        placeholder = f"<<<{count}>>>"
        count += 1
        return placeholder

    # Заменяем все найденные коды на плейсхолдеры
    cleaned_text = re.sub(r'(&[0-9a-zA-Z]+)', repl, text)
    print(f"\nИсходный текст (без форматирования): {cleaned_text}")
    
    # Переводим очищенный текст целиком
    translated = translate_to(cleaned_text, lang_to)
    
    # Восстанавливаем форматирование: заменяем плейсхолдеры на исходные коды по порядку
    for i, code in enumerate(codes, start=1):
        placeholder = f"<<<{i}>>>"
        translated = translated.replace(placeholder, code, 1)
    
    print(f"Результат перевода с восстановлением форматирования: {translated}\n")
    return translated

def process_file(file_path: Path, lang_to: str):
    """
    Обрабатывает один файл:
      - Читает содержимое.
      - Переводит поля title, subtitle и description с сохранением форматирования.
      - Записывает переведённый результат в новую папку (заменяя 'chapters' на 'chapters-translate').
    """
    print(f"\nОбработка файла: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Применяем безопасный перевод для полей title, subtitle и description
    content = re.sub(
        r'(title:\s*\")(.*?)(\")',
        lambda m: f'{m.group(1)}{safe_translate(m.group(2), lang_to)}{m.group(3)}',
        content
    )
    content = re.sub(
        r'(subtitle:\s*\")(.*?)(\")',
        lambda m: f'{m.group(1)}{safe_translate(m.group(2), lang_to)}{m.group(3)}',
        content
    )
    content = re.sub(
        r'(description:\s*\[)(.*?)(\])',
        lambda m: f'{m.group(1)}{safe_translate(m.group(2), lang_to)}{m.group(3)}',
        content,
        flags=re.DOTALL
    )
    
    output_path = str(file_path).replace('chapters', 'chapters-translate')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content)

def process_all_files(directory: str, lang_to: str):
    """
    Обрабатывает все файлы с расширением .snbt в указанной директории,
    используя параллельную обработку с отображением прогресса.
    """
    files = list(Path(directory).rglob("*.snbt"))
    total_files = len(files)
    
    with tqdm.tqdm(total=total_files, desc="Обработка файлов") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for idx, file_path in enumerate(files):
                print(f"\nТекущий файл ({idx + 1}/{total_files}): {file_path}")
                futures.append(executor.submit(process_file, file_path, lang_to))
                pbar.update(1)
            for future in concurrent.futures.as_completed(futures):
                future.result()

def main():
    directory = input("Введите путь к папке с файлами (например, /home/username/quests/chapters): ").strip()
    lang_to = input("Введите целевой язык (например, ru для русского): ").strip()
    process_all_files(directory, lang_to)

if __name__ == '__main__':
    main()
