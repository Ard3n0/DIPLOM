import json

DATA_PATH = "dataset.json"

def find_json_break():
    print(f"Сканирование {DATA_PATH} на предмет структурных повреждений")
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            line_idx = 0
            for line in f:
                line_idx += 1
                if line_idx % 100000 == 0:
                    print(f"Проверено {line_idx} строк")
            
            f.seek(0)
            json.load(f)
            print("Файл целый.")
    except json.JSONDecodeError as e:
        print(f"Ошибка в строке {e.lineno}, колонка {e.colno}")
        print(f"Сообщение: {e.msg}")
        

        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start_view = max(0, e.lineno - 3)
            end_view = min(len(lines), e.lineno + 2)
            print("фрагмент кода с ошибкой")
            for i in range(start_view, end_view):
                marker = ">>> " if i + 1 == e.lineno else "    "
                print(f"{i+1:6}: {marker}{lines[i].strip()}")

if __name__ == "__main__":
    find_json_break()