import sys

sys.path.append("/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1_CausalGenerationResultAnalysis/LLM")
sys.path.append("/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1_CausalGenerationResultAnalysis")

from pgmpy.models import BayesianNetwork
from LLM import Qwen
from LLM.Meta_LLM import LargeLanguageModel
import pandas as pd
import pprint
import networkx as nx


def is_connected(bn_structure):
    """
    判断贝叶斯网络是否连通
    :param bn_structure: 贝叶斯网络结构，格式为列表的元组，例如 [('A', 'B'), ('B', 'C')]
    :return: 是否连通
    """
    G = nx.DiGraph(bn_structure)
    return nx.is_weakly_connected(G)


def has_missing_nodes(bn_structure_, node_list):
    """
    判断贝叶斯网络是否存在节点缺失
    :param bn_structure_: 贝叶斯网络结构，格式为列表的元组，例如 [('A', 'B'), ('B', 'C')]
    :param node_list: 贝叶斯网络的节点列表
    :return: 是否存在节点缺失
    """
    G = nx.DiGraph(bn_structure_)
    existing_nodes = set(G.nodes)
    expected_nodes = set(node_list)
    missing_nodes_ = expected_nodes - existing_nodes
    return bool(missing_nodes_), missing_nodes_


def has_cycles(bn_structure_):
    """
    判断贝叶斯网络是否存在环
    :param bn_structure_: 贝叶斯网络结构，格式为列表的元组，例如 [('A', 'B'), ('B', 'C')]
    :return: 是否存在环
    """
    G = nx.DiGraph(bn_structure_)
    try:
        nx.find_cycle(G, orientation='original')
        return True
    except nx.NetworkXNoCycle:
        return False


def check_BN(bn_structure_, node_list):
    """
    检查网络结构是否合法
    :param bn_structure_: 网络结构，格式为列表的元组，例如 [('A', 'B'), ('B', 'C')]
    :param node_list: 网络节点列表
    :return: 合法性结果，
    """
    connected_ = is_connected(bn_structure_)

    # 判断是否存在节点缺失
    missing_, missing_nodes_ = has_missing_nodes(bn_structure_, node_list)

    # 判断是否存在环
    cyclic_ = has_cycles(bn_structure_)

    return connected_ & (~missing_) & (~cyclic_), {'connected': connected_, 'missing': missing_, 'cyclic': cyclic_,
                                             'missing_nodes': missing_nodes_}


def Refine_BN(raw_BNN, nodes_name_lst, web_llm):
    check_res_flag, check_res = check_BN(raw_BNN, nodes_name_lst)
    if check_res_flag:
        print('legal network structure')
    else:
        while 1:
            print('illegal network structure')
            Error_prompt = f"你给的网络拓扑格式有误，请重新生成。错误的原因是："
            Reason_prompt = ""
            if check_res['missing']:
                Reason_prompt += f"存在节点缺失，缺失节点为：{check_res['missing_nodes']};"
            if check_res['cyclic']:
                Reason_prompt += f"存在环;"
            if check_res['connected']:
                Reason_prompt += f"网络不连通;"
            Error_prompt += Reason_prompt
            response1 = web_llm.response_only_text(web_llm.generate_msg(Error_prompt))
            raw_BNN = eval(response1)
            check_res_flag, check_res = check_BN(raw_BNN, nodes_name_lst)
            if check_res_flag:
                print('legal network structure')
                break
    return raw_BNN

def init_LLM() -> LargeLanguageModel:
    LLM_cur = Qwen.Qwen()
    return LLM_cur


def generate_BNN_MStar_Naive(nodes_name_lst, nodes_info_list) -> BayesianNetwork:
    web_llm = init_LLM()
    web_llm.open_history_log()

    Zone = '保险领域'

    merge_info = [nodes_name_lst[i] + ":" + nodes_info_list[i] for i in range(len(nodes_info_list))]

    identity_prompt = f"你是一个{Zone}的专家，你需要用你的专业知识解决如下的问题："
    problem_prompt = "你需要根据提供的信息，给出一个最合理的网络结构，这些信息包括节点名称和节点含义的详细解释"
    info_prompt = f"节点的名称和含义的详细解释为：{merge_info}"
    CoT_prompt = "你需要一步一步的进行思考并且将你的思考过程展示出来。具体来说，你可以先对节点进行分层，然后再构建拓扑"

    prompt = identity_prompt + problem_prompt + info_prompt + CoT_prompt
    print('waiting for LLM response of M*(stage 1/2):')
    response = web_llm.response_only_text(web_llm.generate_msg(prompt))

    mission_prompt = "现在结合你的思考过程，将网络的结构总结汇总。"
    structure_prompt = "最后返回的网络拓扑格式有特殊要求。举个例子，对于一个三个节点的网络，拓扑为A到B，B到C，则返回：[('A', 'B'),('B','C')]。你本次的返回值只应该包含网络拓扑列表，不要返回其他任何的多余字符。"

    prompt1 = mission_prompt + structure_prompt
    print('waiting for LLM response of M*(stage 2/2)')
    response1 = web_llm.response_only_text(web_llm.generate_msg(prompt1))

    while 1:
        try:
            raw_BNN = eval(response1)
            BN_Structure = BayesianNetwork(raw_BNN)
            break
        except Exception as e:
            print(e)
            Error_prompt = f"你给的网络拓扑格式有误，解析时报错如下：{e}，请重新生成。"
            response1 = web_llm.response_only_text(web_llm.generate_msg(Error_prompt))
    raw_BNN_new = Refine_BN(raw_BNN, nodes_name_lst, web_llm)  # TODO 这里不合理，生成新的Raw_BNN_new后可能还是无法使用eval进行解析
    return BN_Structure


def generate_BNN_MStar(nodes_name_lst, nodes_info_list) -> BayesianNetwork:
    """
    输入网络节点信息和解释，输出网络结构
    :param nodes_name_lst: 网络节点名称
    :param nodes_info_list: 网络节点信息
    :return: 网络结构
    """
    return generate_BNN_MStar_Naive(nodes_name_lst, nodes_info_list)


if __name__ == '__main__':
    """
    Insurance_define_df = pd.read_csv(
        "/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1_CausalGenerationResultAnalysis/Insurance/InsuranceDefineChinese.csv")
    node_names = list(Insurance_define_df['var_name'])
    node_detail = list(Insurance_define_df['var_description'])
    response = generate_BNN_MStar(node_names, node_detail)
    pprint.pprint(response.edges())
    """
    # 示例网络结构
    bn_structure = [
        ('Age', 'RiskAversion'), ('Age', 'GoodStudent'), ('RiskAversion', 'DrivingSkill'),
        ('SocioEcon', 'RiskAversion'), ('SocioEcon', 'SeniorTrain'), ('SocioEcon', 'MakeModel'),
        ('SeniorTrain', 'DrivingSkill'), ('MakeModel', 'CarValue'), ('MakeModel', 'Airbag'),
        ('MakeModel', 'Antilock'), ('GoodStudent', 'DrivingSkill'), ('DrivingSkill', 'DrivHist'),
        ('DrivHist', 'DrivQuality'), ('DrivQuality', 'Accident'), ('CarValue', 'RuggedAuto'),
        ('VehicleYear', 'CarValue'), ('VehicleYear', 'Mileage'), ('HomeBase', 'AntiTheft'),
        ('Airbag', 'Cushioning'), ('Antilock', 'Cushioning'), ('RuggedAuto', 'ThisCarDam'),
        ('Accident', 'ThisCarDam'), ('Accident', 'Theft'), ('Accident', 'MedCost'), ('Accident', 'ILiCost'),
        ('ThisCarDam', 'Cushioning'), ('ThisCarDam', 'ThisCarCost'), ('Theft', 'ThisCarCost'),
        ('ThisCarCost', 'PropCost'), ('OtherCar', 'OtherCarCost'), ('OtherCarCost', 'PropCost')
    ]

    # 示例节点列表
    node_lists = [
        'Age', 'RiskAversion', 'GoodStudent', 'DrivingSkill', 'DrivHist', 'DrivQuality', 'Accident',
        'CarValue', 'RuggedAuto', 'ThisCarDam', 'Theft', 'MedCost', 'ILiCost', 'ThisCarCost', 'PropCost',
        'SocioEcon', 'SeniorTrain', 'MakeModel', 'Airbag', 'Antilock', 'Cushioning', 'VehicleYear', 'Mileage',
        'HomeBase', 'OtherCar', 'OtherCarCost'
    ]

    # 判断是否连通
    connected = is_connected(bn_structure)
    print(f"贝叶斯网络是否连通: {connected}")

    # 判断是否存在节点缺失
    missing, missing_nodes = has_missing_nodes(bn_structure, node_lists)
    print(f"贝叶斯网络是否存在节点缺失: {missing}")
    if missing:
        print(f"缺失的节点: {missing_nodes}")

    # 判断是否存在环
    cyclic = has_cycles(bn_structure)
    print(f"贝叶斯网络是否存在环: {cyclic}")
