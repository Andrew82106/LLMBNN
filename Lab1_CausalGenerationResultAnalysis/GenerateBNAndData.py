from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models.BayesianNetwork import BayesianNetwork
import pandas as pd


def define_bayesian_network_from_bif(file_path) -> BayesianNetwork:
    """
    从BIF文件中读取贝叶斯网络的结构。

    参数:
    file_path (str): BIF文件路径。

    返回:
    BayesianNetwork: 定义好的贝叶斯网络对象。
    """
    reader = BIFReader(file_path)
    model = reader.get_model()
    return model


def generate_samples(model_in, num_of_samples=100) -> pd.DataFrame:
    """
    使用贝叶斯网络生成样本数据。

    参数:
    model_in (BayesianNetwork): 贝叶斯网络对象。
    num_of_samples (int): 生成的样本数量，默认为100。

    返回:
    DataFrame: 包含生成样本数据的DataFrame。
    """
    sampler = BayesianModelSampling(model_in)
    data = sampler.forward_sample(size=num_of_samples)
    return data


def save_samples_to_csv(data, file_path):
    """
    将生成的样本数据保存到CSV文件中。

    参数:
    data (DataFrame): 包含生成样本数据的DataFrame。
    file_path (str): CSV文件路径。
    """
    data.to_csv(file_path, index=False)


# 示例使用
if __name__ == "__main__":
    # 生成100个样本数据
    num_samples = 1000

    # 从BIF文件中读取贝叶斯网络
    bif_file_path = '/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1:CausalGenerationResultAnalysis/Insurance/insurance.bif'  # 替换为你的BIF文件路径
    model = define_bayesian_network_from_bif(bif_file_path)

    samples_data = generate_samples(model, num_samples)

    # 将样本数据保存到CSV文件中
    csv_file_path = '/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1:CausalGenerationResultAnalysis/Insurance/Insurance_generated_samples.csv'  # 替换为你想要保存的CSV文件路径
    save_samples_to_csv(samples_data, csv_file_path)

    print(f"生成的样本数据已保存到 {csv_file_path}")

    bif_file_path = "/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1:CausalGenerationResultAnalysis/Child/child.bif"
    model = define_bayesian_network_from_bif(bif_file_path)
    samples_data = generate_samples(model, num_samples)
    csv_file_path = "/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1:CausalGenerationResultAnalysis/Child/Child_generated_samples.csv"
    save_samples_to_csv(samples_data, csv_file_path)
    print(f"生成的样本数据已保存到 {csv_file_path}")