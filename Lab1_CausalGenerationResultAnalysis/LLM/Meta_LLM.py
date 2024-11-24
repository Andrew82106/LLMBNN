class LargeLanguageModel:
    def __init__(self):
        self.base_url = ""
        self.api_key = ""
        self.default_llm_identity = "assistant"
        self.default_user_identity = "user"
        self.chat_history = []
        self.log_history = False

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
        return [{"role": self.default_user_identity, "content": input_msg}] if not self.log_history else self.chat_history
