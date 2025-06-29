from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from tools.knowledge_tool import KnowledgeTool
from tools.api_tool import WeatherAPI
from tools.local_function_tool import LocalFunctionTool

# 初始化 LLM
llm = OpenAI(temperature=0)

# 初始化工具
knowledge_tool = KnowledgeTool()
weather_api = WeatherAPI()
local_function_tool = LocalFunctionTool()

tools = [
    Tool(
        name="知识库问答",
        func=knowledge_tool.query,
        description="回答基于知识库的问题"
    ),
    Tool(
        name="天气查询",
        func=weather_api.get_weather,
        description="根据城市名查询实时天气"
    ),
    Tool(
        name="本地排序",
        func=lambda input_str: local_function_tool.sort_list(eval(input_str)),
        description="对数字列表进行排序，输入格式示例：[5,3,1]"
    )
]

agent = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True)

def main():
    print("智能协作Agent已启动，输入 exit 退出")
    chat_history = []
    while True:
        user_input = input("你：")
        if user_input.lower() == "exit":
            break
        response = agent.run(user_input)
        print("Agent：", response)
        chat_history.append((user_input, response))

if __name__ == "__main__":
    main()

