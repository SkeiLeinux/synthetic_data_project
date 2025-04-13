# synth_generation/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Импортируем гиперпараметры из настроек и модель CTGAN
from config.settings import MODEL_PARAMS, ANONYMIZATION_PARAMS
from synth_generation.model import CTGAN
from anonymization.anonymity_metrics import evaluate  # Функция, рассчитывающая метрики анонимности


def compute_anonymity_penalty(synthetic_data_np):
    """
    Вычисляет штраф за анонимность на основе метрик k-анонимности, l-разнообразия и t-близости.
    :param synthetic_data_np: Сгенерированные данные в формате numpy.ndarray.
    :return: Штраф (скаляр), который равен сумме недоборных показателей.
    """
    # Рассчитываем текущие метрики по синтетическим данным
    metrics = evaluate(synthetic_data_np)
    target_k = ANONYMIZATION_PARAMS['k']  # Желаемое значение k
    target_l = ANONYMIZATION_PARAMS['l']  # Желаемое значение l
    target_t = ANONYMIZATION_PARAMS['t']  # Желаемое значение t

    penalty = 0.0
    k_metric = metrics.get('k_anonymity', target_k)
    l_metric = metrics.get('l_diversity', target_l)
    t_metric = metrics.get('t_closeness', target_t)

    # Если текущая k-анонимность меньше желаемой, штрафуем разницу
    if k_metric < target_k:
        penalty += (target_k - k_metric)
    # Если текущая l-разнообразность меньше желаемой, штрафуем разницу
    if l_metric < target_l:
        penalty += (target_l - l_metric)
    # Если текущая t-близость превышает допустимый порог, штрафуем разницу
    if t_metric > target_t:
        penalty += (t_metric - target_t)

    return penalty


def train_model(real_data, noise_dim, condition_dim, data_dim,
                num_epochs=MODEL_PARAMS['epochs'],
                batch_size=MODEL_PARAMS['batch_size'],
                lr=MODEL_PARAMS['learning_rate'],
                lambda_penalty=1.0):
    """
    Функция обучения CTGAN с дополнительным штрафом за несоблюдение метрик анонимности.
    :param real_data: Исходный (реальный) датасет в формате numpy.ndarray.
    :param noise_dim: Размерность входного шума.
    :param condition_dim: Размерность условного вектора.
    :param data_dim: Число признаков в данных.
    :param num_epochs: Количество эпох обучения.
    :param batch_size: Размер обучающего батча.
    :param lr: Скорость обучения оптимизатора.
    :param lambda_penalty: Весовой коэффициент для штрафа по метрикам анонимности.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Инициализируем CTGAN модель и переводим на выбранное устройство (CPU или GPU)
    ctgan = CTGAN(noise_dim, condition_dim, data_dim, hidden_dim=256).to(device)

    # Определяем оптимизаторы для генератора и дискриминатора
    optimizer_G = optim.Adam(ctgan.generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(ctgan.discriminator.parameters(), lr=lr)

    # Функция потерь для GAN (Binary Cross Entropy)
    adversarial_loss = nn.BCELoss()

    # Преобразуем реальные данные в тензор и перемещаем на устройство
    real_data_tensor = torch.tensor(real_data, dtype=torch.float).to(device)

    # Вычисляем число батчей
    num_batches = real_data_tensor.size(0) // batch_size

    # Обучающий цикл по эпохам
    for epoch in range(num_epochs):
        for i in range(num_batches):
            # Формируем батч реальных данных
            real_batch = real_data_tensor[i * batch_size:(i + 1) * batch_size]

            # Создаем случайный условный вектор (можно заменить на реальные условия, если они есть)
            condition = torch.randn(batch_size, condition_dim).to(device)

            # Генератор: создаем случайный шум и генерируем синтетические данные
            noise = torch.randn(batch_size, noise_dim).to(device)
            synthetic_data = ctgan.generator(noise, condition)

            # Формируем метки: для реальных данных = 1, для синтетических = 0
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            ## Обучение дискриминатора
            optimizer_D.zero_grad()
            # Дискриминатор оценивает реальные данные
            real_pred = ctgan.discriminate(real_batch, condition)
            loss_real = adversarial_loss(real_pred, valid)
            # Дискриминатор оценивает синтетические данные (без обратного прохода для генератора)
            fake_pred = ctgan.discriminate(synthetic_data.detach(), condition)
            loss_fake = adversarial_loss(fake_pred, fake)
            # Итоговая потеря дискриминатора
            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            ## Обучение генератора
            optimizer_G.zero_grad()
            # Генератор пытается обмануть дискриминатор
            fake_pred = ctgan.discriminate(synthetic_data, condition)
            g_loss = adversarial_loss(fake_pred, valid)

            # Вычисляем штраф по метрикам анонимности из сгенерированных данных
            # Преобразуем синтетические данные в формат numpy для вычисления метрик
            synthetic_data_np = synthetic_data.detach().cpu().numpy()
            penalty = compute_anonymity_penalty(synthetic_data_np)
            # Преобразуем штраф в torch.tensor (не дифференцируемую константу)
            penalty_tensor = torch.tensor(penalty, dtype=torch.float).to(device)

            # Итоговая потеря генератора: базовая adversarial_loss плюс штраф
            total_g_loss = g_loss + lambda_penalty * penalty_tensor

            total_g_loss.backward()
            optimizer_G.step()

            # Выводим информацию каждые 10 батчей
            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{i}/{num_batches}] "
                      f"D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}  Penalty: {penalty:.4f}")

    print("Обучение завершено")


# Пример вызова функции обучения (при условии, что real_data передан в виде numpy-массива)
if __name__ == "__main__":
    # Допустим, real_data – случайная матрица для демонстрации (например, 1000 образцов с data_dim признаками)
    np.random.seed(42)
    data_dim = 20
    real_data = np.random.rand(1000, data_dim)

    noise_dim = 100  # Размерность шума
    condition_dim = 10  # Размерность условного вектора

    train_model(real_data, noise_dim, condition_dim, data_dim)
