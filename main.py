from src.data_preparation import load_data, preprocess_data
from src.model import train_model, evaluate_model
import pandas as pd

def main():
    # Загрузка данных
    train, test = load_data('E:\\titanic-survival-prediction\\src\\train.csv', 'E:\\titanic-survival-prediction\\src\\test.csv')
    
    # Предобработка данных
    X, y, X_test = preprocess_data(train, test)
    
    # Обучение модели
    model = train_model(X, y)
    
    # Проверка наличия 'PassengerId' в test
    if 'PassengerId' not in test.columns:
        print("Ошибка: 'PassengerId' отсутствует в тестовом наборе данных.")
        return
    
    # Предсказание на тестовых данных
    predictions = model.predict(X_test)
    
    # Проверка содержимого predictions
    print("Первые 10 предсказаний:", predictions[:10])
    
    # Создание DataFrame для submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })
    
    # Проверка структуры DataFrame
    print("Структура submission:")
    print(submission.head())
    
    # Сохранение результатов
    submission.to_csv('submission.csv', index=False)
    print("Файл submission.csv успешно сохранен.")

if __name__ == "__main__":
    main()