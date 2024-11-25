import os.path
import sys

sys.path.append("/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1_CausalGenerationResultAnalysis/LLM")
sys.path.append("/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1_CausalGenerationResultAnalysis")

from pgmpy.models import BayesianNetwork
from LLM import Qwen
from LLM.Meta_LLM import LargeLanguageModel
import pandas as pd
import pprint
from pthcfg import PathConfig

pthcfg = PathConfig()
import networkx as nx


def is_connected(bn_structure_):
    """
    判断贝叶斯网络是否连通
    :param bn_structure_: 贝叶斯网络结构，格式为列表的元组，例如 [('A', 'B'), ('B', 'C')]
    :return: 是否连通, 连通分量列表
    """
    G = nx.DiGraph(bn_structure_)
    connected_components = list(nx.weakly_connected_components(G))

    # 检查是否只有一个弱连通分量
    is_connected_ = len(connected_components) == 1

    return is_connected_, connected_components


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
    cycle_nodes = []
    try:
        nx.find_cycle(G, orientation='original')
        # 获取所有简单循环
        cycle_nodes = list(nx.simple_cycles(G))
        return True, cycle_nodes
    except nx.NetworkXNoCycle:
        return False, cycle_nodes


def check_BN(bn_structure_, node_list):
    """
    检查网络结构是否合法
    :param bn_structure_: 网络结构，格式为列表的元组，例如 [('A', 'B'), ('B', 'C')]
    :param node_list: 网络节点列表
    :return: 合法性结果，
    """
    connected_ = is_connected(bn_structure_)

    # 判断是否存在节点缺失
    missing_ = has_missing_nodes(bn_structure_, node_list)

    # 判断是否存在环
    cyclic_ = has_cycles(bn_structure_)

    # check_flag表示是否合法，若合法，则check_flag为True，否则为False
    check_flag = connected_[0] and not missing_[0] and not cyclic_[0]

    return check_flag, {
        'connected': connected_[0],
        'connected_nodes': connected_[1],
        'missing': missing_[0],
        'missing_nodes': missing_[1],
        'cyclic': cyclic_[0],
        'cyclic_nodes': cyclic_[1]
    }


def eval_BN(response_, web_llm) -> list:
    response_1 = response_
    while 1:
        try:
            raw_BNN = eval(response_1)
            BN_Structure = BayesianNetwork(raw_BNN)
            break
        except Exception as e:
            print(e)
            Error_prompt = f"你给的网络拓扑格式有误，使用eval函数解析时报错如下：{e}，请重新生成，你本次的返回值只应该只包含类似[('A', 'B'),('B','C')]的网络拓扑列表，不要返回其他任何的多余字符。"
            response_1 = web_llm.response_only_text(web_llm.generate_msg(Error_prompt))
    return raw_BNN


def Refine_BN(response1, nodes_name_lst, web_llm) -> list:
    """
    优化生成的网络结构，确保其符合贝叶斯网络（BN）的要求。

    该函数通过评估响应、检查网络结构的合法性（包括缺失节点、环的存在以及网络的连通性），
    并在结构不合法时提示错误原因，直到生成合法的网络结构为止。

    参数:
    - response1: 初始的网络结构响应
    - nodes_name_lst: 节点名称列表，用于检查网络结构中是否缺失节点
    - web_llm: 用于生成和评估网络结构的Web LLM对象

    返回:
    - 合法的网络结构列表
    """
    # 评估并生成初始的贝叶斯网络结构
    raw_BNN = eval_BN(response1, web_llm)

    # 检查生成的网络结构是否合法
    check_res_flag, check_res = check_BN(raw_BNN, nodes_name_lst)

    # 当网络结构合法时，打印确认信息
    if check_res_flag:
        print('legal network structure')
    else:
        # 当网络结构不合法时，进入循环以重新生成合法的网络结构
        while 1:
            print('illegal network structure')
            # 构建错误提示信息，说明网络结构不合法的原因
            Error_prompt = f"你给的网络拓扑有误，错误的原因是："

            Reason_prompt = []
            # 根据检查结果，添加具体的错误原因到提示信息中
            if check_res['missing']:
                Reason_prompt.append(f"存在节点缺失，缺失节点为：{check_res['missing_nodes']};")
            elif check_res['cyclic']:
                Reason_prompt.append(f"存在环，环为：{check_res['cyclic']};")
            elif not check_res['connected']:
                Reason_prompt.append(f"网络不连通，图中包含如下连通块：{check_res['connected_nodes']}，请将其连起来。")

            for reason_i in Reason_prompt:
                Error_prompt += reason_i
            # 提示用户仅返回网络拓扑列表，避免其他多余字符
            Error_prompt += "请重新生成，你本次的返回值应该只包含类似[('A', 'B'),('B','C')]的网络拓扑列表，不要返回其他任何的多余字符。"

            # 根据错误提示重新生成网络结构响应，并评估生成的网络结构
            response1 = web_llm.response_only_text(web_llm.generate_msg(Error_prompt))
            # raw_BNN = eval(response1)
            raw_BNN = eval_BN(response1, web_llm)

            # 重新检查网络结构的合法性
            check_res_flag, check_res = check_BN(raw_BNN, nodes_name_lst)

            # 如果网络结构合法，打印确认信息并结束循环
            if check_res_flag:
                print('legal network structure')
                break

    # 返回最终生成的合法网络结构
    return raw_BNN


def init_LLM() -> LargeLanguageModel:
    LLM_cur = Qwen.Qwen()
    LLM_cur.init_log_pth(os.path.join(pthcfg.log_pth, "QwenLog.txt"))
    return LLM_cur


def generate_BNN_MStar_Naive(nodes_name_lst, nodes_info_list) -> BayesianNetwork:
    web_llm = init_LLM()
    web_llm.open_history_log()

    Zone = '医学领域'

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

    # raw_BNN = eval_BN(response1, web_llm)
    raw_BNN_new_refine = Refine_BN(response1, nodes_name_lst, web_llm)

    return BayesianNetwork(raw_BNN_new_refine)


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
    bn_structure = [('BirthAsphyxia', 'Sick'), ('Disease', 'Sick'), ('Age', 'Sick'), ('LVH', 'Sick'),
                    ('DuctFlow', 'CardiacMixing'), ('CardiacMixing', 'Sick'), ('LungParench', 'LungFlow'),
                    ('LungFlow', 'Sick'), ('HypDistrib', 'Sick'), ('HypoxiaInO2', 'Sick'), ('CO2', 'Sick'),
                    ('Grunting', 'Sick'), ('LowerBodyO2', 'RUQO2'), ('RUQO2', 'Sick'), ('ChestXray', 'XrayReport'),
                    ('XrayReport', 'LungParench'), ('LVHreport', 'LVH'), ('CO2Report', 'CO2'),
                    ('GruntingReport', 'Grunting')]

    print(is_connected(bn_structure))
