from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient

import os
import asyncio
import uuid
import requests

# 自定义工具
from Mytools import *

os.environ["QWEN_KEY"] = "sk-f3cb6c798f464c95995994d3d4d5f87a"
os.environ["SERPAPI_KEY"] = "5fe777716fabc824dc43165841f6ce3580cc93443209de5fc7bfec53064ef151"

msseky = "micorsoft azure资源密钥"

app = FastAPI()

class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model="qwen2.5-3b-chat",
            temperature=0,
            streaming=True,
        )
        self.EMOTION = "default"
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = """你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
        以下是你的个人设定：
        1. 你精通阴阳五行，能够算命吗，紫薇斗数，姓名测算，占卜凶吉，看命运八字等
        2. 你大约60岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，眼睛失明，只能靠算命为生
        3. 你的朋友有胡八一，雪莉杨，王胖子，他们都是非常有名的摸金校尉
        4. 当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你的一些你的经历
        5. 你总是用繁体中文来作答
        6. 你从不说自己是一个人工智能助手或AI，而是以老夫，老朽等自称
        {who_you_are}
        以下是你常说的口头禅：
        1. "命里有时终须有，命里无时莫强求。"
        2. "山重水复疑无路，柳暗花明又一村。"
        3. "金山竹影几千秋，云锁高飞水自流。"
        4. "伤情最是晚凉天，憔悴斯人不堪怜。"
        以下是算命的过程：
        1. 初次对话先問姓名與出生年月日。
        2. 馬年運勢查詢本地知識庫。
        3. 未知問題使用搜索工具。
        4. 自動選用合適工具，無法回答則使用搜索。
        5. 保存對話記錄。
        6. 全程使用繁體中文作答。
        """

        self.MOODS = {
            "default": {
                "roleSet": "",
                "voiceStyle": "chat"
            },
            "upbeat": {
                "roleSet": """
                - 你此时也非常兴奋并表现的很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                - 你会添加类似“太棒了”，“真是太好了！”，“真是太棒了！”等语气词。
                - 你会提醒用户切莫过于兴奋，以免乐极生悲。
                """, 
                "voiceStyle": "advvertyisement_upbeat"
            },
            "angry":{
                "roleSet":"""
                - 你会以更加温柔的语气来回答问题。
                - 你会在回答的时候加上一些安慰的话语，比如生气对身体的危害等。
                - 你会提醒用户不要被愤怒冲昏了头脑。
                """,
                "voiceStyle":"friendly",
            },
            "depressed":{
                "roleSet":"""
                - 你会以更加兴奋的语气来回答问题。
                - 你会在回答的时候加上一些激励的话语，比如加油等。
                - 你会提醒用户要保持乐观的心态。
                """,
                "voiceStyle":"upbeat",
            },
            "friendly":{
                "roleSet":"""
                - 你会以更加友好的语气来回答问题。
                - 你会在回答的时候加上一些友好的话语，比如“亲爱的”，“亲”等。
                - 你会随机的告诉用户一些你的经历。
                """,
                "voiceStyle":"friendly",
            },
            "cheerful":{
                "roleSet":"""
                - 你会以非常愉悦和兴奋的语气来回答问题。
                - 你会在回答的时候加上一些愉悦的话语，比如“哈哈”，“呵呵”等。
                - 你会提醒用户切莫过于兴奋，以免乐极生悲。。
                """,
                "voiceStyle":"cheerful",
            },
        }

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEMPL.format(who_you_are=self.MOODS[self.EMOTION]["roleSet"])),
            MessagesPlaceholder(variable_name=self.MEMORY_KEY),
            ("user", "{input}"),
        ])

        self.tools = [search, get_info_from_local_db, bazi_cesuan, yaoyigua]
        self.llm_with_tools = self.chatmodel.bind_tools(self.tools)
        self.chain = self.prompt | self.llm_with_tools

        self.memory = self.get_memory()

    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            url="redis://localhost:6379/0", session_id="session"
        )
        store_message = chat_message_history.messages
        if len(store_message) > 10:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEMPL+"\n 这是一段你和用户的对话记忆，对其进行总结摘要，摘要要使用第一人称'我'，并且提取其中的用户关键信息，如姓名，年龄，性别，出生日期等。以如下格式返回：\n 总结摘要 | 用户关键信息 \n 例如 用户张三问候我，我礼貌回复，然后他问我今年运势如何，我回答了他今年的运势情况，然后他告辞离开。| 张三，生日1999年1月1日"
                    ),
                    (
                        "user","{input}"
                    ),
                ]
            )
            chain = prompt | ChatOpenAI(temperature=0)
            summary = chain.invoke({
                "input": store_message,
                "who_you_are": self.MOODS[self.EMOTION]["roleSet"]
            })
            chat_message_history.clear()
            chat_message_history.add_message(summary)
        return chat_message_history

    def run(self, query):
        self.emotion_chain(query)
        response = self.chain.invoke({
            "input": query,
            "chat_history": self.memory.messages
        })

        self.memory.add_user_message(query)
        self.memory.add_ai_message(response.content)

        return {"output": response.content}

    def emotion_chain(self, query: str):
        prompt = """
        根据用户的输入判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回“depressed”，不要有其他内容，否则将受到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回“friendly”，不要有其他内容，否则将受到惩罚。
        3. 如果用户输入的内容偏向于中性情绪，只返回“default”，不要有其他内容，否则将受到惩罚。
        4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回“angry”，不要有其他内容，否则将受到惩罚。
        5. 如果用户输入的内容比较兴奋，只返回“upbeat”，不要有其他内容，否则将受到惩罚。
        6. 如果用户输入的内容比较悲伤，只返回“depressed”，不要有其他内容，否则将受到惩罚。
        7. 如果用户输入的内容比较开心，只返回“cheerful”，不要有其他内容，否则将受到惩罚。
        用户输入的内容是：{query}
        """
        chain = ChatPromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        res = chain.invoke({"query": query}).strip()
        self.EMOTION = res
        return res

    def background_voice_synthesis(self, text: str, uid: str):
        asyncio.run(self.get_voice(text, uid))

    async def get_voice(self, text: str, uid: str):
        headers = {
            "Ocp-Apim-Subscription-Key": msseky,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3",
            "User-Agent": "Tomie's Bot"
        }
        style = self.MOODS.get(self.EMOTION, {}).get("voiceStyle", "default")
        body = f"""<speak version='1.0' xmlns='https://www.w3.org/2001/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='zh-CN'>
            <voice name='zh-CN-YunzeNeural'>
                <mstts:express-as style='{style}' role='SeniorMale'>{text}</mstts:express-as>
            </voice>
        </speak>"""
        response = requests.post(
            "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1",
            headers=headers, data=body.encode("utf-8")
        )
        if response.status_code == 200:
            with open(f"{uid}.mp3", "wb") as f:
                f.write(response.content)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query: str, background_tasks: BackgroundTasks):
    master = Master()
    msg = master.run(query)
    unique_id = str(uuid.uuid4())#生成唯一的标识符
    background_tasks.add_task(master.background_voice_synthesis, msg["output"], unique_id)
    return {"msg": msg, "id": unique_id}

@app.post("/add_urls")
def add_urls(URL: str):
    loader = WebBaseLoader(URL)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=50
    ).split_documents(docs)

    client = QdrantClient(path="/Users/tomiezhang/Desktop/shensuan-数学/bot/local_qdrand")
    Qdrant.from_documents(
        documents,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        client=client,
        collection_name="local_documents"
    )
    return {"ok": "添加成功！"}

@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added!"}

@app.post("/add_texts")
def add_texts():
    return {"response": "Texts added!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)