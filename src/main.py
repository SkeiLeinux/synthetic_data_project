from data_manager import DataManager
import pandas as pd


def main():
    dm = DataManager()

    # Проверка чтения данных
    query = "SELECT * FROM synthetic_data_schema.raw_data;"
    df = dm.load_data(query)
    print("Исходные данные из БД:")
    print(df)

    # Проверка записи данных (сделаем новую таблицу для теста)
    test_data = pd.DataFrame({
        'column_1': ['example_1', 'example_2'],
        'column_2': [123, 456],
        'column_3': [1.23, 4.56]
    })

    dm.save_data(test_data, 'test_table')

    print("✅ Данные успешно записаны в таблицу test_table.")


if __name__ == "__main__":
    main()
