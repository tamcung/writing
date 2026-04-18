# SQL 注入检测系统

本系统用于对接业务应用请求，完成 SQL 注入攻击检测、告警统计、请求审计和模型配置管理。系统默认面向“业务应用检测”场景：业务请求由网关复制到检测后端，原业务链路不被阻塞，检测结果进入系统总览和流量监测页面。

## 功能边界

当前最终版保留 5 个前端模块：

- 系统总览：展示业务请求量、恶意命中率、高风险告警、接入应用数量、趋势图、风险结构、Top 告警 URI 和最新恶意事件。
- 在线检测：提供单条请求检测工具，可选择主模型并加入对比模型，便于演示不同模型输出差异。
- 流量监测：查看接入检测系统的业务请求记录，支持按应用、风险等级、判定结果和关键词筛选。
- 应用管理：维护业务应用 Host、默认检测模型、判定阈值和启停状态。
- 模型管理：查看 TextCNN、BiLSTM、CodeBERT 等模型的加载状态、指标和 checkpoint。

不再保留独立“检测历史”“鲁棒性评估”“系统状态”模块，相关能力已收敛到总览、流量监测、应用管理和模型管理中。

## 目录结构

```text
system/
├── backend/              FastAPI + SQLAlchemy 后端服务
├── demo_app/             本地业务应用演示服务
├── frontend/             React + TypeScript + Ant Design + ECharts 前端
├── gateway/              Nginx / OpenResty 业务请求复制示例
├── scripts/              启停、灌入演示数据、截图脚本
├── docker-compose.yml    PostgreSQL / Elasticsearch / Kibana / backend / frontend / gateway
└── README.md
```

## 后端接口

- `GET /api/v1/health`：服务健康检查。
- `GET /api/v1/models`：查看可用模型与 checkpoint 状态。
- `GET /api/v1/applications`：查看业务应用配置。
- `POST /api/v1/applications`：新增业务应用。
- `PATCH /api/v1/applications/{id}`：更新业务应用配置。
- `DELETE /api/v1/applications/{id}`：删除业务应用。
- `POST /api/v1/predict`：在线检测单条请求。
- `POST /api/v1/batch-predict`：批量检测。
- `GET /api/v1/traffic-records`：供流量监测页面查询检测记录。
- `POST /api/v1/ingest/traffic`：接收网关复制过来的业务请求。

## 模型加载

后端启动时会检查 `system/backend/storage/checkpoints/`：

- 如果 checkpoint 已存在，直接加载。
- 如果 checkpoint 不存在且 `AUTO_BOOTSTRAP=true`，则基于论文实验数据自动训练默认模型。

默认模型：

- `textcnn:clean_ce`
- `textcnn:pair_ce`
- `textcnn:pair_canonical`
- `bilstm:clean_ce`
- `bilstm:pair_ce`
- `bilstm:pair_canonical`
- `codebert:clean_ce`
- `codebert:pair_ce`

CodeBERT 需要本地已有 `microsoft/codebert-base` 权重。常用环境变量：

```bash
BOOTSTRAP_MODELS=textcnn:clean_ce,textcnn:pair_ce,textcnn:pair_canonical,bilstm:clean_ce,bilstm:pair_ce,bilstm:pair_canonical,codebert:clean_ce,codebert:pair_ce
BOOTSTRAP_CODEBERT=true
CODEBERT_LOCAL_ONLY=true
CODEBERT_FREEZE_ENCODER=true
CODEBERT_EPOCHS=1
CODEBERT_BATCH_SIZE=8
CODEBERT_MAX_LEN=256
```

## 本地开发

后端：

```bash
cd /Users/tansong/Workspaces/writing
source .venv/bin/activate
PYTHONPATH=system/backend uvicorn app.main:app --host 0.0.0.0 --port 8000
```

前端：

```bash
cd /Users/tansong/Workspaces/writing/system/frontend
npm install
npm run dev
```

默认地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000/api/v1`

## Docker Compose

```bash
cd /Users/tansong/Workspaces/writing/system
docker compose up --build
```

启动后可访问：

- 前端控制台：`http://127.0.0.1:3000`
- 后端 API：`http://127.0.0.1:8000/api/v1`
- Kibana：`http://127.0.0.1:5601`

## 演示启动

推荐使用宿主机后端 + 宿主机前端 + Docker 网关的方式，便于截图和调试：

```bash
cd /Users/tansong/Workspaces/writing
bash system/scripts/run_host_demo.sh
bash system/scripts/seed_demo_data.sh
bash system/scripts/capture_demo_screenshots.sh
```

截图输出到：

- `system/demo_screenshots/overview.png`
- `system/demo_screenshots/detect.png`
- `system/demo_screenshots/traffic.png`
- `system/demo_screenshots/applications.png`
- `system/demo_screenshots/models.png`

关闭演示环境：

```bash
bash system/scripts/stop_host_demo.sh
```

## DVWA 靶场演示

系统内置开源 DVWA 靶场演示脚本，用于产生真实 SQL 注入攻击流量。

启动 DVWA 与检测网关：

```bash
cd /Users/tansong/Workspaces/writing
bash system/scripts/run_dvwa_lab.sh
```

默认地址：

- DVWA 直连：`http://127.0.0.1:4280`
- DVWA 检测入口：`http://127.0.0.1:8090`

产生一轮 SQL 注入攻击请求：

```bash
bash system/scripts/exercise_dvwa_lab.sh
```

生成靶场页与流量监测页截图：

```bash
bash system/scripts/capture_dvwa_demo_screenshots.sh
```

一键完成“起靶场 → 打攻击 → 截图”：

```bash
bash system/scripts/run_dvwa_detection_demo.sh
```

停止靶场：

```bash
bash system/scripts/stop_dvwa_lab.sh
```

## 业务应用接入

业务系统无需修改代码。网关把请求复制到检测后端，检测系统根据应用 Host 或 `application_id` 匹配应用配置，并使用对应模型和阈值完成检测。

接收接口：

```text
POST /api/v1/ingest/traffic?application_id=<ID>
```

Nginx 示例位于 `system/gateway/nginx/default.conf.template`，OpenResty 示例位于 `system/gateway/openresty/`。两者都会保留原业务转发，同时把请求方法、URI、Host、源 IP、User-Agent、请求体等上下文复制到检测后端。

## 预处理说明

- URL 查询串和 `application/x-www-form-urlencoded` 表单字段会按系统默认规则解码。
- 原始 payload 不额外解码，避免破坏原始攻击载荷形态。
- 如果请求中没有可提取的 URL/Form 参数，系统会使用完整请求文本作为送检文本。
