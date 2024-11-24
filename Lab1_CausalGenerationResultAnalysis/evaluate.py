import pandas as pd
from pgmpy.models import BayesianNetwork
from typing import List, Dict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def predict(model_in: BayesianNetwork, data: pd.DataFrame, aim_variables_lst: list):
    """
    使用贝叶斯模型进行预测。

    该函数首先从数据集中移除目标变量，以防止这些变量影响模型的预测结果。
    然后，使用处理后的数据集对模型进行训练。
    最后，返回模型对处理后数据集的预测结果。

    参数:
    - model_in: BayesianNetwork，一个贝叶斯统计模型实例，用于进行数据预测。
    - data: DataFrame，输入的数据集，包含需要预测的特征和可能的目标变量。
    - aim_variables_lst: List[str]，目标变量列表，这些变量将从数据集中移除。

    返回:
    - prediction: Series or DataFrame，模型对处理后数据集的预测结果。
    """
    # 从原始数据中移除目标变量，防止这些变量影响预测结果
    drop_data = data.drop(aim_variables_lst, axis=1)

    # 使用处理后的数据集对模型进行训练
    model_in.fit(data)

    # 返回模型对处理后数据集的预测结果
    return model_in.predict(drop_data)


def calculate_metrics(prediction: pd.DataFrame, data_in: pd.DataFrame, aim_variables_lst: List[str]) -> Dict[
    str, Dict[str, float]]:
    """
    计算预测结果的准确率、召回率、F1分数等指标。
    :param prediction: 预测结果
    :param data_in: 输入数据
    :param aim_variables_lst: 目标变量列表
    :return: 准确率、召回率、F1分数等指标
    """
    if not isinstance(prediction, pd.DataFrame) or not isinstance(data_in, pd.DataFrame):
        raise TypeError("Both prediction and data_in must be pandas DataFrames.")

    if not isinstance(aim_variables_lst, list) or not all(isinstance(var, str) for var in aim_variables_lst):
        raise TypeError("aim_variables_lst must be a list of strings.")

    if not set(aim_variables_lst).issubset(set(data_in.columns)):
        raise ValueError("Some variables in aim_variables_lst are not present in data_in columns.")

    try:
        # data取aim_variables_lst列
        data = data_in[aim_variables_lst]

        # prediction和data需要是结构相同大小相同的DataFrame
        if prediction.shape != data.shape:
            raise ValueError(
                f"The prediction and data must have the same shape, got {prediction.shape} and {data.shape} respectively.")

        result = {}
        print('Evaluation Results:')
        for aim_variable in aim_variables_lst:
            print("\naim variable:", aim_variable)

            # 提前缓存数据
            y_true = data[aim_variable]
            y_pred = prediction[aim_variable]

            # 计算指标
            metrics = compute_metrics(y_true, y_pred)

            result[aim_variable] = metrics
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print()

        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    计算准确率、召回率、精确率和F1分数。
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 指标字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1_score': f1}


if __name__ == '__main__':
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10],
        'C': [3, 6, 9, 12, 15]
    })
    model = BayesianNetwork([('A', 'B'), ('B', 'C')])
    predict_result = predict(model, df, ['C', 'B'])

    print(predict_result)
    print(calculate_metrics(predict_result, df, ['C', 'B']))
