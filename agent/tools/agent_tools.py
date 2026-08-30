import os
import pandas as pd
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

rag = RagSummarizeService()

# 加载患者慢病数据表
PATIENT_CSV_PATH = get_abs_path(agent_conf["external_data_path"])
df_patient = pd.read_csv(PATIENT_CSV_PATH, encoding="utf-8-sig")


@tool
def rag_summarize(query: str) -> str:
    """从向量知识库检索慢病医学指南资料，回答健康科普问题"""
    return rag.rag_summarize(query)


def fetch_patient_data(patient_id: str) -> dict:
    """
    根据患者ID获取完整体检慢病数据
    """
    row = df_patient[df_patient["patient_id"] == patient_id]
    if row.empty:
        return {"error": f"未找到编号为 {patient_id} 的患者数据"}

    patient = row.iloc[0].to_dict()
    # 性别翻译
    gender_map = {"F": "女", "M": "男"}
    patient["gender_cn"] = gender_map.get(patient["gender"], "未知")
    # 运动量翻译
    sport_map = {
        "Low": "低运动量",
        "Moderate": "中等运动量",
        "High": "高运动量"
    }
    patient["sport_cn"] = sport_map.get(patient["sport_freq"], "未知")
    return patient


@tool
def calculate_risk_score(patient_id: str) -> str:
    """
    评估患者心血管慢病风险等级，生成随访报告时使用。入参：patient_id，例如 P002
    """
    patient = fetch_patient_data(patient_id)
    if "error" in patient:
        return patient["error"]

    risk_map = {
        "INTERMEDIARY": "中风险",
        "HIGH": "高风险",
        "LOW": "低风险"
    }
    risk_cn = risk_map.get(patient["cvd_risk_level"], "未知风险")

    output = (
        f"患者ID：{patient_id}\n"
        f"心血管风险等级：{risk_cn}\n"
        f"CVD风险得分：{patient['cvd_risk_score']}"
    )
    return output

@tool
def calc_cvd_risk(age:int, systolic_bp:int, smoker:bool, bmi:float):
    """
    根据年龄、收缩压、吸烟状态、BMI 简易计算心血管患病风险等级
    :param age: 患者年龄
    :param systolic_bp: 收缩压 mmHg
    :param smoker: 是否吸烟，True吸烟 / False不吸烟
    :param bmi: BMI身体质量指数
    :return: 风险等级：低风险 / 中风险 / 高风险
    """
    score = 0
    if age >=55:
        score +=1
    if systolic_bp >=140:
        score +=1
    if smoker:
        score +=1
    if bmi >=28:
        score +=1

    if score <=1:
        level = "低风险"
    elif score == 2:
        level = "中风险"
    else:
        level = "高风险"

    return f"心血管简易风险评估结果：{level}，得分：{score}"



@tool
def fill_context_for_report() -> str:
    """无入参，调用后触发报告模式，切换随访报告专用提示词"""
    return "fill_context_for_report已调用"
