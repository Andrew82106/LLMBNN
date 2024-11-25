import os


class PathConfig:
    def __init__(self):
        self.root_file_name = "114项目"
        self.cur_path = os.path.dirname(os.path.abspath(__file__))
        # cur_path中包含了root_file_name
        self.root_path = os.path.dirname(self.cur_path)
        self.lab1_path = os.path.join(self.root_path, "Lab1_CausalGenerationResultAnalysis")
        self.log_pth = os.path.join(self.lab1_path, "logs")


if __name__ == '__main__':
    print(PathConfig().root_path)
    print(PathConfig().cur_path)
    print(PathConfig().lab1_path)
    print(PathConfig().log_pth)