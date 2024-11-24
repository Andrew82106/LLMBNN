from zhipuai import ZhipuAI
import requests
from openai import OpenAI
from Meta_LLM import LargeLanguageModel



class ChatGLM_Origin_Zhipu(LargeLanguageModel):
    def __init__(self, model_name="glm-4-plus"):
        super().__init__()
        self.api_key = "33b333df733a7ba7174034ef5d757c8f.1MlCkHLb22BysIPi"
        self.model_name = model_name
        self.client = ZhipuAI(api_key=self.api_key)  # 请填写您自己的APIKey

    def response(self, messages):
        """
        生成对给定消息的响应。

        此函数调用OpenAI的Chat API来生成对一系列消息的响应。它使用类中定义的模型名称和提供的消息列表作为输入。

        参数:
        - messages (list): 包含对话历史的消息列表，这些消息将被用作模型的输入，以生成合适的响应。

        返回:
        - response.choices[0].message: 模型生成的响应消息。API的响应对象包含多个选择（choices），我们选择第一个选项的消息内容作为最终响应。
        """
        response = self.client.chat.completions.create(
            model=self.model_name,  # 请填写您要调用的模型名称
            messages=messages,
        )
        return response.choices[0].message

    def response_only_text(self, messages):
        """
        回复消息，仅返回文本内容

        该方法调用了`response`方法来处理消息，并仅返回响应的文本内容

        参数:
        - messages: 消息列表，作为回复的输入

        返回:
        - response.content: 响应的文本内容
        # TODO: 加入History记录功能
        """
        # 调用response方法处理消息
        response = self.response(messages)
        # 返回响应的文本内容
        return response.content


class GLM_Silicon_requests(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        self.api_key = "sk-arewqkdkvqngbqxcmetmaihrggvlukewzqkzjcqsvtmojhep"
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.model_name = "THUDM/glm-4-9b-chat"
        self.stream = False
        self.max_tokens = 512
        self.temperature = 0.7
        self.top_p = 0.7
        self.top_k = 50
        self.frequency_penalty = 0.5
        self.n = 1
        self.response_format = {"type": "text"}
        self.headers = {
            "Authorization": "Bearer <token>",
            "Content-Type": "application/json"
        }

    def response(self, messages):
        """
        根据给定的消息生成响应。

        该方法构造了一个payload（负载）对象，包含了为模型生成响应所需的各种参数，
        然后使用requests库向指定的URL发送POST请求，并返回响应结果。

        参数:
        - messages (list): 一个消息列表，包含了对话历史或当前讨论的主题。

        返回:
        - response (Response): 发送POST请求后的响应对象，包含了模型的响应信息。
        """

        # 构造请求负载
        payload = {
            "model_in": self.model_name,  # 模型名称
            "messages": messages,  # 消息列表
            "stream": self.stream,  # 是否开启流式响应
            "max_tokens": self.max_tokens,  # 响应的最大token数
            "temperature": self.temperature,  # 采样温度，决定输出的随机性
            "top_p": self.top_p,  # 核采样的比例
            "top_k": self.top_k,  # 核采样的数量
            "frequency_penalty": self.frequency_penalty,  # 频率惩罚，避免重复的响应
            "n": self.n,  # 生成的响应数量
            "response_format": self.response_format,  # 响应格式
        }

        # 发送POST请求
        response = requests.request("POST", self.url, json=payload, headers=self.headers)

        # 返回响应对象
        return response


class GLM_Silicon_Openai(LargeLanguageModel):
    def __init__(self):
        super().__init__()
        self.api_key = "sk-arewqkdkvqngbqxcmetmaihrggvlukewzqkzjcqsvtmojhep"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        self.model_name = "THUDM/glm-4-9b-chat"

    def response(self, message):
        """
        生成对给定消息的响应。

        此函数调用OpenAI的Chat API来生成与输入消息相关的响应。它指定模型名称、消息内容和响应格式，
        并返回生成的响应对象。

        参数:
        message (list): 包含对话历史的列表，用于生成上下文相关的响应。

        返回:
        response: 由OpenAI API生成的响应对象，包含模型输出的文本响应。
        """
        # 创建聊天完成响应，指定模型名称、消息和响应格式
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            response_format={"type": "text"}
        )

        # 返回生成的响应对象
        return response

    def response_only_text(self, message):
        """
        根据给定的消息返回响应文本。

        本函数调用了`self.response`方法来处理输入的消息，并从响应中提取出第一个选择的内容作为输出。
        这种处理方式适用于只需要文本内容，而不需其他元数据的场景。

        参数:
        - message: 要处理的消息内容。

        返回:
        - 提取自响应的文本内容。
        """
        # 调用response方法获取响应
        response = self.response(message)
        # 提取并返回响应中的文本内容
        return response.choices[0].message.content


if __name__ == '__main__':
    LLM = GLM_Silicon_Openai()
    messages = LLM.generate_msg("你好，帮我查一下今天是农历几月几日")
    print(LLM.response_only_text(messages))