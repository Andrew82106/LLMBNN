from datetime import datetime


class LargeLanguageModel:
    def __init__(self):
        self.base_url = ""
        self.api_key = ""
        self.default_llm_identity = "assistant"
        self.default_user_identity = "user"
        self.chat_history = []
        self.log_history = False
        self.log_pth = None

    @staticmethod
    def generate_single_log(info: str):
        time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info = str(info).replace('}, {', '}\n {')
        return 50*">" + f"[{time_stamp}]:\n {info}"

    def init_log_pth(self, log_pth):
        self.log_pth = log_pth
        with open(log_pth, "w") as f:
            f.write("")

    def save_logfile(self, info: str):
        if self.log_pth is not None:
            with open(self.log_pth, "a") as f:
                f.write(self.generate_single_log(info) + "\n")

    def response(self, message):
        raise Exception("请重写本方法")

    def response_only_text(self, message):
        raise Exception("请重写本方法")

    def reset_history(self):
        self.chat_history = []

    def open_history_log(self):
        self.log_history = True

    def generate_msg(self, input_msg):
        if self.log_history:
            self.chat_history.append({"role": self.default_user_identity, "content": input_msg})
            if self.log_pth is not None:
                self.save_logfile(info={"role": self.default_user_identity, "content": input_msg})
                # self.save_logfile(info=self.generate_single_log(self.chat_history))
        return [{"role": self.default_user_identity, "content": input_msg}] if not self.log_history else self.chat_history
