# MW答题系统 Demo（FastAPI + 静态前端）

## 目录
- backend/ FastAPI 接口 + 静态文件托管
- frontend/ 纯 HTML + JS 前端
- questions.json 题库（从 MW设计.docx 抽取了 3 题做演示）

## 本地运行
1) 进入 backend 并安装依赖
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

2) 启动
```bash
uvicorn main:app --reload --port 8000
```

3) 打开浏览器访问
- http://127.0.0.1:8000/

## 接口
- GET /api/lessons
- GET /api/questions?lesson_id=L1
- POST /api/score  {question_id, answer_text}

## 说明
这是“半自动评分”的最小可用 Demo：
- 结构要素（判断/理由/结果）+ 关键词 + 结论命中
- 输出缺失提示，便于学生按模板完善答案
