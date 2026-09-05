# 本草问心 · 多Agent协同中医辨证论治系统 (TCM Multi-Agent System)

> 融汇伤寒、温病、脾胃、火神、中西医汇通五大学派大医思维。系统性十问四诊交互，主控脑枢动态追问，多位名医 Agent 各陈方略会诊，输出具有置信度评估的理法方药案札。

---

## 🌟 系统核心架构与特性

1. **东方古籍线装卷轴美学**
   - 沉浸式宣纸质感肌理、朱砂手盖印章、黛青立线排版、仿古名牌签条与仿古方单案札。
2. **古法十问海量结构化题库（10大卷 · 29条四诊条目）**
   - 寒热、汗出、疼痛、饮食口味、二便、睡眠精神、胸腹头身、情志脉络、专科体质与病史。
3. **大模型双核架构 (OpenAI 兼容协议)**
   - **Fast 敏捷核**（推荐 `deepseek-v4-flash` / `gpt-4o-mini`）：主控精准研判、动态追问补充关键病机。
   - **Pro 辨证核**（推荐 `deepseek-v4-pro` / `gpt-4o` / `deepseek-r1`）：伤寒、温病、脾胃、火神、汇通五大学派并行深层推理与主控合参整合。
4. **多学派学术论难与会诊报告**
   - 支持实时 SSE 流式推演。
   - 双选项卡呈现：【合参统宗案】（核心主证/病机阐微/治则方药/共识与分歧）与【五派学术辨】（各派各自主张对比）。
5. **管治枢纽（运行时无感换源）**
   - 界面右上角「枢」印章进入管理配置，无需重启服务即可热加载 API Key、Base URL 与双模型路由。

---

## 🛠️ 技术栈

- **前端**：React 18 + Vite + Tailwind CSS + Framer Motion (东方动效) + Lucide Icons
- **后端**：Python 3.12 + FastAPI + Uvicorn + Pydantic v2 + HTTPX (异步高并发)
- **数据库**：PostgreSQL 16 + pgvector (向量检索与病案存储底座)
- **部署**：Docker + Docker Compose 一键编排容器化运行

---

## 🚀 快速启动

### 方式一：Docker Compose 一键部署（推荐）

1. 克隆代码库：
   ```bash
   git clone <REPO_URL>
   cd "多Agent辩证的中医论治系统"
   ```

2. 复制环境模版：
   ```bash
   cp .env.example .env
   ```

3. 启动全栈服务：
   ```bash
   docker compose up -d --build
   ```

4. 打开浏览器访问：
   - 前端体验界面：[http://localhost:5173](http://localhost:5173)
   - 后端接口文档：[http://localhost:8001/docs](http://localhost:8001/docs)

### 方式二：本地研发模式

**后端启动**：
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**前端启动**：
```bash
cd frontend
npm install
npm run dev
```

---

## ⚙️ 模型接入配置指南

进入前台后，点击右上角「**枢**」印泥印章，输入管理令牌（默认 `tcm-admin`）即可配置任意 OpenAI 兼容的 API：
- **Base URL**：例如 `https://api.openai.com/v1` 或各类兼容中继/官方端点
- **API Key**：你的模型密钥 `sk-...`（本地持久化存储，严格脱敏）
- **Fast 敏捷核**：填入轻量快速模型
- **Pro 辨证核**：填入高智力辨证模型
- 点击「测试连通性」验证无误后即刻生效，全系统自动由规则底座无缝跃迁至大模型推理。

---

## 📜 法律与医学伦理免责声明

> 本系统生成的所有辨证推演、学术论难、病机分析与方药建议，均由多 Agent 人工智能协同推求，仅供中医爱好者学术研讨、思路参考与健康调摄辅助之用。
> **AI 不能替代执业中医师的临证四诊望闻问切。涉及具体处方剂量、急重症或用药，务必遵从正规医疗机构专业医师医嘱。**
