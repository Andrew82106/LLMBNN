from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
# https://pgmpy.org/examples/Monty%20Hall%20Problem.html


# Defining the network structure
model = BayesianNetwork([("C", "H"), ("P", "H")])

# Defining the CPDs:
cpd_c = TabularCPD("C", 3, [[0.33], [0.33], [0.33]])
cpd_p = TabularCPD("P", 3, [[0.33], [0.33], [0.33]])
cpd_h = TabularCPD(
    "H",
    3,
    [
        [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
        [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
        [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0],
    ],
    evidence=["C", "P"],
    evidence_card=[3, 3],
)

# Associating the CPDs with the network structure.
model.add_cpds(cpd_c, cpd_p, cpd_h)

# Some other methods
model.get_cpds()


infer = VariableElimination(model)
posterior_p = infer.query(["P"], evidence={"C": 0, "H": 2})
print(posterior_p)

"""
好的，以下是这些快捷键的表格展示：

| 操作 | Windows 快捷键 | Macos 快捷键 |
| --- | --- | --- |
| 触发补全 | Alt + P | Option + P |
| 更换生成结果 | Alt + ] | Option + ] |
| 采纳全部生成的代码 | Tab | Tab |
| 逐行采纳生成的代码 | Ctrl + ↓ | Cmd + ↓ |
| 关闭/打开对话面板 | Ctrl + Shift + L | Cmd + Shift + L |

希望这个表格可以帮助您更好地理解和记忆这些快捷键。
"""