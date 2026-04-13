from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient

import requests
import json

YUANFENJU_API_KEY = "qIJsiugRrSTsag0poJMxyeiIO"

@tool
def test():
    """Text tool"""
    return "test"

@tool
def search(query: str):
    """只有需要了解实时信息或者不知道的事情的时候才会使用搜索工具。"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    print("实时搜索结果：", result)
    return result

@tool
def get_info_from_local_db(query: str):
    """只有回答与2026年运势或者马年运势相关的问题的时候，会使用这个工具。"""
    client = Qdrant(
        client=QdrantClient(path="/Users/tomiezhang/Desktop/shensuan-数学/bot/local_qdrand"),
        collection_name="local_documents",
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = client.as_retriever(search_type="mmr")
    result = retriever.invoke(query)
    return result

@tool
def bazi_cesuan(query: str):
    """只有做八字排盘的时候才会使用这个工具，需要输入用户姓名和出生年月日，如果缺少姓名和出生年月日则不可用"""
    url = "https://api.yuanfenju.com/index.php/v1/Bazi/paipan"
    prompt = ChatPromptTemplate.from_template(
        """
        你是一个参数查询助手，根据用户输入内容找出相关的参数并按照json格式返回。
        JSON字段：
        - "api_key": "qIJsiugRrSTsag0poJMxyeiIO"
        - "name": 姓名
        - "sex": 性别，0男，1女
        - "type": 日历，0农历，1公历
        - "year": 出生年, "month": 月, "day": 日, "hour": 时, "minute": 0
        无参数则提醒用户，只返回JSON，无其他内容
        用户输入：{query}
        """
    )
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | ChatOpenAI(temperature=0) | parser
    data = chain.invoke({"query": query})
    print("八字查询参数：", data)

    res = requests.post(url, data=data)
    if res.status_code == 200:
        try:
            res_json = res.json()
            return_str = f"八字为：{res_json['data']['bazi_info']['bazi']}"
            return return_str
        except Exception as e:
            return "八字查询失败，请先询问用户姓名与出生年月日。"
    else:
        return "技术错误，请告诉用户稍后再试。"

@tool
def yaoyigua():
    """只有用户想要占卜抽签的时候才会使用这个工具。"""
    url = "https://api.yuanfenju.com/index.php/v1/Zhanbu/meiri"
    res = requests.get(url, params={"api_key": YUANFENJU_API_KEY})
    if res.status_code == 200:
        res_json = res.json()
        print(res_json)
        return res_json
    else:
        return "技术错误，请告诉用户稍后再试。"

@tool
def jiemeng(query: str):
    """只有用户想要解梦的时候才会使用这个工具，需要梦境内容。"""
    url = "https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    llm = ChatOpenAI(temperature=0)
    prompt = PromptTemplate.from_template("提取关键词，只返回关键词，内容：{topic}")
    keyword = llm.invoke(prompt.format(topic=query))
    
    if hasattr(keyword, "content"):
        keyword = keyword.content.strip()

    print("提取关键词：", keyword)
    res = requests.post(url, data={"api_key": YUANFENJU_API_KEY, "keyword": keyword})
    
    if res.status_code == 200:
        res_json = res.json()
        print(res_json)
        return res_json
    else:
        return "技术错误，请告诉用户稍后再试。"