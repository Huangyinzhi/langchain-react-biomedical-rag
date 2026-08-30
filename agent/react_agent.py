from dotenv import load_dotenv
load_dotenv()

from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks import BaseCallbackHandler
from model.factory import get_chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from agent.tools.agent_tools import (
    rag_summarize,
    calculate_risk_score,
    fill_context_for_report,
    calc_cvd_risk
)

from utils.logger_handler import logger




# 回调处理器：监控打印工具调用日志
class ToolMonitorCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        tool_name = serialized.get("name")
        logger.info(f"[tool monitor]执行工具：{tool_name}")
        logger.info(f"[tool monitor]传入参数：{input_str}")

    def on_tool_end(self, output, **kwargs):
        logger.info(f"[tool monitor]工具调用成功")

    def on_tool_error(self, error, **kwargs):
        logger.error(f"工具调用失败，原因：{str(error)}")


class ReactAgent:
    def __init__(self):
        self.llm = get_chat_model()
        self.tools = [
            rag_summarize,
            calculate_risk_score,
            fill_context_for_report,
            calc_cvd_risk
        ]
        self.base_system_prompt = load_system_prompts()
        self.report_system_prompt = load_report_prompts()
        self.callbacks = [ToolMonitorCallback()]

    def execute_stream(self, query: str):
        temp_agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            state_modifier=self.base_system_prompt
        )
        input_data = {"messages": [{"role": "user", "content": query}]}
        need_report = False
        buffer_messages = []
        # 第一轮试探执行
        for chunk in temp_agent.stream(input_data, stream_mode="values"):
            messages = chunk.get("messages", [])
            buffer_messages = messages
            logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息。")
            last_msg = messages[-1]
            # 判断工具返回标记，触发报告模式
            if last_msg.type == "tool" and "fill_context_for_report" in last_msg.name:
                need_report = True
        # 如果检测到报告标记，重新创建Agent，加载报告提示词，完整执行
        if need_report:
            logger.info("检测到报告生成标记，切换报告提示词")
            report_agent = create_react_agent(
                model=self.llm,
                tools=self.tools,
                state_modifier=self.report_system_prompt
            )
            stream_iter = report_agent.stream({"messages": buffer_messages}, stream_mode="values")
        else:
            stream_iter = temp_agent.stream({"messages": buffer_messages}, stream_mode="values")
        # 流式输出给前端
        for chunk in stream_iter:
            messages = chunk.get("messages", [])
            if not messages:
                continue
            latest_msg = messages[-1]
            if latest_msg.content:
                yield latest_msg.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("患者ID:P002，请评估慢病风险，生成一份个性化随访报告"):
        print(chunk, end="", flush=True)
