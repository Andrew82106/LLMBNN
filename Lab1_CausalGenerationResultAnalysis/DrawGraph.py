import matplotlib.pyplot as plt
import networkx as nx


def visualize_bayesian_network(bn_structure_, savepath, figsize=(12, 12)):
    """
    可视化贝叶斯网络
    :param bn_structure_: 贝叶斯网络结构，格式为列表的元组，例如 [('A', 'B'), ('B', 'C')]
    :param savepath: 图像保存路径
    :param figsize: 图像大小，默认为 (12, 12)
    """
    # 创建有向图对象
    model = nx.DiGraph(bn_structure_)

    # 计算节点和边的数量
    num_nodes = len(model.nodes)

    # 获取节点的层次信息
    def get_levels(G):
        levels_ = {node_: 0 for node_ in G.nodes}
        for node__ in G.nodes:
            if not list(G.predecessors(node__)):
                levels_[node__] = 0
            else:
                levels_[node__] = max([levels_[pred] for pred in G.predecessors(node__)]) + 1
        return levels_

    levels = get_levels(model)

    # 将层级信息设置为节点的属性
    for node, level in levels.items():
        model.nodes[node]['level'] = level

    # 使用 multipartite_layout 布局
    pos = nx.multipartite_layout(model, subset_key="level", align='horizontal')

    # 调整节点大小和颜色
    node_sizes = [300] * num_nodes  # 默认节点大小
    node_colors = ['skyblue'] * num_nodes  # 默认节点颜色

    # 调整字体大小
    font_size = max(8, 15 - int(num_nodes / 10))

    # 设置图像大小
    plt.figure(figsize=figsize)

    # 绘制图形
    nx.draw(model, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=font_size,
            font_weight='bold')
    plt.title('Bayesian Network Structure')

    # 保存图像
    plt.savefig(savepath)
    plt.close()


if __name__ == '__main__':
    # 示例网络结构
    bn_structure = [('BirthAsphyxia', 'LVH'), ('LVH', 'DuctFlow'), ('LVH', 'LungParench'), ('Disease', 'CardiacMixing'), ('CardiacMixing', 'LungParench'), ('CardiacMixing', 'LungFlow'), ('LungParench', 'HypDistrib'), ('LungFlow', 'HypDistrib'), ('LungFlow', 'CO2'), ('LungFlow', 'Grunting'), ('LungFlow', 'HypoxiaInO2'), ('HypDistrib', 'LowerBodyO2'), ('HypDistrib', 'RUQO2'), ('Sick', 'Age'), ('Sick', 'ChestXray'), ('Sick', 'LVHreport'), ('Sick', 'CO2Report'), ('Sick', 'XrayReport'), ('Sick', 'GruntingReport'), ('Sick', 'HypDistrib')]
    visualize_bayesian_network(bn_structure, 'test.png', figsize=(12, 12))
