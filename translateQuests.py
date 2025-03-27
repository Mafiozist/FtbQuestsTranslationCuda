import os
import re
import concurrent.futures
from transformers import MarianMTModel, MarianTokenizer
import torch
import tqdm

# Определяем, доступен ли GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Функция для загрузки модели
def get_translator(lang_to):
    model_name = f'Helsinki-NLP/opus-mt-en-{lang_to}'
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Функция для перевода текста
def translate_to(text, lang_to):
    # Выводим исходный текст
    print(f"Исходный текст: {text}")
    
    model, tokenizer = get_translator(lang_to)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # Выводим переведенный текст
    print(f"Переведенный текст: {translated_text}\n")
    
    return translated_text

# Функция для обработки файла
def process_file(file_path, lang_to):
    print(f"Обработка файла: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Применяем регулярное выражение для поиска и перевода текста
    content = re.sub(
        r'(title:\s*\")(.*?)(\")',
        lambda m: f'{m.group(1)}{translate_to(m.group(2), lang_to)}{m.group(3)}',
        content
    )

    # Записываем переведенный контент обратно в файл
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# Функция для обработки всех файлов в директории
def process_all_files(directory, lang_to):
    # Получаем список всех файлов в директории
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    total_files = len(files)

    # Прогресс-бар для всех файлов
    with tqdm.tqdm(total=total_files, desc="Обработка файлов") as pbar:
        # Используем ThreadPoolExecutor для параллельной обработки файлов
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for idx, file_path in enumerate(files):
                # Выводим текущий прогресс и файл
                print(f"Текущий файл ({idx + 1}/{total_files}): {file_path}")
                futures.append(executor.submit(process_file, file_path, lang_to))
                pbar.update(1)

            # Ожидаем завершения всех задач
            for future in concurrent.futures.as_completed(futures):
                future.result()

def main():
    # Путь к директории с файлами
    directory = input("Введите путь к папке с файлами (например, /home/username/quests/chapters): ").strip()
    lang_to = input("Введите целевой язык (например, ru для русского): ").strip()
    
    process_all_files(directory, lang_to)

if __name__ == '__main__':
    main()
