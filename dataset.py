from datasets import load_dataset
import json

# 加载 EternaBench-ChemMapping 数据集
dataset = load_dataset("multimolecule/eternabench-cm")


train = "trainset.jsonl"


with open(train, "w") as f_out:
    for entry in dataset["train"]:
        structure = entry.get("secondary_structure")
        sequence = entry.get("sequence")
        if structure and sequence:
            # 将结构和序列保存为 JSONL 格式
            json.dump({"structure": structure, "sequence": sequence}, f_out)
            f_out.write("\n")


test = "testset.jsonl"


with open(test, "w") as f_out:
    for entry in dataset["test"]:
        structure = entry.get("secondary_structure")
        sequence = entry.get("sequence")
        if structure and sequence:
            # 将结构和序列保存为 JSONL 格式
            json.dump({"structure": structure, "sequence": sequence}, f_out)
            f_out.write("\n")
print(f"Processed data saved to {test}")