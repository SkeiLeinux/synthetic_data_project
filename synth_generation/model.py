# synth_generation/model.py

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_dim, hidden_dim=256):
        """
        Инициализация генератора.
        :param noise_dim: Размерность входного шума.
        :param condition_dim: Размерность условного вектора (например, one-hot представление категориальных признаков).
        :param output_dim: Размерность выходного вектора (количество признаков в сгенерированном образце).
        :param hidden_dim: Размерность скрытых слоев.
        """
        super(Generator, self).__init__()
        # Совмещаем шум и условный вектор (concatenation)
        input_dim = noise_dim + condition_dim
        # Определяем последовательную модель (MLP) генератора
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Первое линейное преобразование
            nn.ReLU(True),  # Функция активации ReLU
            nn.Linear(hidden_dim, hidden_dim),  # Еще один скрытый слой
            nn.ReLU(True),  # Функция активации ReLU
            nn.Linear(hidden_dim, output_dim),  # Выходной слой, выходное число соответствует числу признаков
            nn.Tanh()  # Tanh, чтобы привести значения в диапазон [-1, 1]
        )

    def forward(self, noise, condition):
        """
        Прямой проход: объединяем шум и условный вектор и генерируем синтетический образец.
        :param noise: Партия случайного шума размерностью (batch_size, noise_dim)
        :param condition: Условный вектор размерностью (batch_size, condition_dim)
        :return: Синтетические данные размерностью (batch_size, output_dim)
        """
        # Объединяем шум и условный вектор по размерности признаков (axis=1)
        x = torch.cat([noise, condition], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=256):
        """
        Инициализация дискриминатора.
        :param input_dim: Размерность входного образца (количество признаков).
        :param condition_dim: Размерность условного вектора.
        :param hidden_dim: Размерность скрытых слоев.
        """
        super(Discriminator, self).__init__()
        # Определяем модель дискриминатора, которая получает на вход данные + условие
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),  # Объединение данных и условия
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU для предотвращения затухания градиента
            nn.Linear(hidden_dim, hidden_dim),  # Скрытый слой
            nn.LeakyReLU(0.2, inplace=True),  # Активация LeakyReLU
            nn.Linear(hidden_dim, 1),  # Выходной слой: однозначная вероятность
            nn.Sigmoid()  # Сигмоида для выдачи значения в диапазоне [0, 1]
        )

    def forward(self, data, condition):
        """
        Прямой проход: объединяем реальные/синтетические данные с условным вектором и вычисляем вероятность.
        :param data: Образец данных размерностью (batch_size, input_dim)
        :param condition: Условный вектор размерностью (batch_size, condition_dim)
        :return: Вероятность того, что данные являются реальными (batch_size, 1)
        """
        # Объединяем данные и условие по признаковому измерению
        x = torch.cat([data, condition], dim=1)
        return self.model(x)


class CTGAN(nn.Module):
    def __init__(self, noise_dim, condition_dim, data_dim, hidden_dim=256):
        """
        Объединяющий класс CTGAN, включающий генератор и дискриминатор.
        :param noise_dim: Размерность шума для генератора.
        :param condition_dim: Размерность условного вектора.
        :param data_dim: Количество признаков в данных.
        :param hidden_dim: Размерность скрытых слоев для обеих сетей.
        """
        super(CTGAN, self).__init__()
        # Инициализируем генератор и дискриминатор с нужными размерами входов и выходов
        self.generator = Generator(noise_dim, condition_dim, data_dim, hidden_dim)
        self.discriminator = Discriminator(data_dim, condition_dim, hidden_dim)

    def generate(self, noise, condition):
        """
        Генерирует синтетические данные на основе заданного шума и условного вектора.
        :param noise: Входной шум для генератора.
        :param condition: Условный вектор для генерации.
        :return: Сгенерированные синтетические данные.
        """
        return self.generator(noise, condition)

    def discriminate(self, data, condition):
        """
        Оценивает, насколько входные данные соответствуют реальным данным с учётом условия.
        :param data: Образцы данных (реальные или синтетические).
        :param condition: Условный вектор, соответствующий данным.
        :return: Вероятности, что данные являются реальными.
        """
        return self.discriminator(data, condition)


# Пример создания экземпляра модели
if __name__ == "__main__":
    # Задаем размеры векторов для демонстрации (примерные значения)
    noise_dim = 100  # Размерность шума
    condition_dim = 10  # Размерность условного вектора (например, число категориальных признаков)
    data_dim = 20  # Число признаков в исходных данных
    batch_size = 16  # Размер партии данных

    # Инициализируем модель CTGAN
    ctgan = CTGAN(noise_dim, condition_dim, data_dim, hidden_dim=256)

    # Генерируем случайный шум и условные векторы
    noise = torch.randn(batch_size, noise_dim)  # Случайный шум (из нормального распределения)
    condition = torch.randn(batch_size,
                            condition_dim)  # Случайные условные векторы (в реальном применении – one-hot векторы)

    # Генерируем синтетические данные
    synthetic_data = ctgan.generate(noise, condition)
    print("Синтетические данные:", synthetic_data)

    # Передаём сгенерированные данные в дискриминатор для оценки
    output = ctgan.discriminate(synthetic_data, condition)
    print("Оценка дискриминатора:", output)

