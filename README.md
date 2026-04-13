# 服务器端：接口 -> langchain -> openai\ollama。。
# 客户端： 电报机器人，微信机器人，website
# 接口：http,https,websocket

# 服务器：
1. 接口访问，python选型fastapi
2. /chat的接口，post的请求
3. /add_ursl 从url中学习知识
4. /add_pdfs 从pdf中学习知识
5. /add_texts 从txt文本中学习知识

# 人性化
1. 用户输入 -> AI判读一下当前问题的情绪倾向？ -> 判断 -> 反馈 -> agent判断
2. 工具调用： 用户发起请求 -> agent判断使用哪个工具 -> 带着相关的参数去请求工具 -> 得到观察结果

# 截至目前：
1. api
2. agent框架
3. tools：搜索，查询信息，专业知识库
4. 记忆，长时记忆
5. 学习能力

## 从url来学习，实现增强
1. 输入URL
2. 地址的HTML变成文本
3. 向量化
4. 检索 -> 相关文本块
5. LLM回答

# 算命大师AI机器人
基于FastAPI + LangChain实现的智能算命机器人，支持八字测算、运势查询、占卜、解梦等功能，集成工具调用、情绪识别、语音合成、长时记忆等能力。

## 功能特性
1. 多工具调用：搜索、本地知识库（2026/马年运势）、八字排盘、占卜抽签、解梦
2. 情绪识别：根据用户输入判断情绪，动态调整回复语气
3. 长时记忆：基于Redis存储对话记录，自动摘要精简
4. 语音合成：调用Azure TTS生成带情绪的语音回复
5. 多接口支持：HTTP接口（/chat、/add_urls等）、WebSocket、LangServe

## 快速开始
### 1. 环境准备
```bash
# 克隆仓库
git clone https://github.com/你的用户名/shensuan-bot.git
cd shensuan-bot

# 安装依赖
pip install -r requirements.txt
