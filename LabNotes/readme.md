# 总体思路

第一步：首先，探索构建$M^*$算法的方式

第二步：然后，对$M^*$算法进行测试。具体而言，使用现有的贝叶斯网络G进行数据采样，反演出原样本数据，然后使用$M^*$算法构建新网络G1，对G和G1进行模型推理测试，比较评测指标。

第三步：最后，使用构建好的$M^*$算法在电诈数据集上使用即可

# 项目记录

## 2024.11.23

第一步和第二步已经完成，但是$M^*$模型效果不及预期：

![output1.png](./output1.png)

当前的M*算法很简单，核心建图就是用最直接的Prompt：

```py
def generate_BNN_MStar(nodes_name_lst, nodes_info_list) -> BayesianNetwork:
    """
    输入网络节点信息和解释，输出网络结构
    :param nodes_name_lst: 网络节点名称
    :param nodes_info_list: 网络节点信息
    :return: 网络结构
    """
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
```

## 2024.11.24

用小的贝叶斯网络测试，发现$M^*$能直接复原出原图，这应该是因为有知识背景，所以能直接复原出原图。但是对于大的网络，$M^*$算法效果不好，就和上面的一样。

用$M^*$算法搞出来的Asia图长这样：

![testAsia.png](testAsia.png)

和官方的图一模一样。

现阶段，小网络的数据污染问题先不管，先管大网络的问题。

大网络用$M^*$算法，现阶段还存在一些问题，最突出的一个就是构建出来的网络结构是非法的，这意味着网络要么不联通要么有环或者要么有些点丢掉了。

针对这种情况，应该设置第三个Prompt，对网络结构进行修改。