import sys
sys.path.append("/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1_CausalGenerationResultAnalysis/LLM")
sys.path.append("/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1_CausalGenerationResultAnalysis")


from pgmpy.models import BayesianNetwork
from LLM import Qwen
from LLM.Meta_LLM import LargeLanguageModel
import pandas as pd
import pprint


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
    Insurance_define_df = pd.read_csv(
         "/Users/andrewlee/Desktop/Projects/实验室/114项目/Lab1_CausalGenerationResultAnalysis/Insurance/InsuranceDefineChinese.csv")
    node_names = list(Insurance_define_df['var_name'])
    node_detail = list(Insurance_define_df['var_description'])
    response = generate_BNN_MStar(node_names, node_detail)
    pprint.pprint(response.edges())