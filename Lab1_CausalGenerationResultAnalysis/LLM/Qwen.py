from Meta_LLM import LargeLanguageModel
from openai import OpenAI


class Qwen(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        self.api_key = "sk-9b0515f545954ca2bd65adfdc676e828"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = "qwen-max"
        self.default_llm_identity = 'system'

    def response(self, messages):
        """
        根据提供的消息生成聊天机器人的回复。

        此函数调用OpenAI的ChatCompletion API来生成回复消息。它首先使用类中定义的模型名称和提供的消息列表
        创建一个聊天完成对象，然后将该对象序列化为JSON格式的字符串并返回。这个过程涉及到与OpenAI API的网络通信，
        因此需要处理网络请求和响应。

        参数:
        messages (list): 一个消息字典的列表，每个字典包含角色（如"system", "user", "assistant"）和内容。
                         例如: [{"role": "user", "content": "你好"}]

        返回:
        str: 一个包含聊天机器人回复信息的复合格式
        """
        # 创建聊天完成对象
        completion = self.client.chat.completions.create(
            model=self.model_name,  # 使用定义的模型名称
            messages=messages,  # 提供的消息列表
        )

        if self.log_history:
            self.chat_history.append({"role": self.default_llm_identity, "content": completion.choices[0].message.content})
            if self.log_pth is not None:
                self.save_logfile(info=self.generate_single_log(self.chat_history))

        return completion

    def response_only_text(self, messages):
        """
        根据输入的消息生成一个只包含文本的响应。

        此方法主要用于处理接收到的消息，并返回一个由OpenAI模型生成的，
        仅包含文本内容的响应。它会从模型的响应中提取出最相关的文本信息。

        参数:
        messages (list): 包含消息的列表，这些消息将被用来生成响应。

        返回:
        str: 由OpenAI模型生成的，与输入消息相关的文本内容。
        """
        response = self.response(messages)
        text_response = response.choices[0].message.content
        # if self.log_history: # duplicated codes
        #     self.chat_history.append({"role": self.default_llm_identity, "content": text_response})
        return text_response


if __name__ == '__main__':
    LLM = Qwen()
    print(LLM.response_only_text(LLM.generate_msg("你好，帮我查一下今天是农历几月几日")))


