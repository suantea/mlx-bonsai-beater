# 接入指南：qwen36-35b-a3b-rq 推理服务 API

> 更新日期：2026-09-19 · 实测验证（非仅文档承诺）

## 1. 链接地址

| 访问方式 | Base URL | 说明 |
|---|---|---|
| **推荐（mDNS 域名）** | `http://lingda-SJdeMacBook-Air.local:8080/v1` | IP 变了无需改配置，域名自动跟随 |
| 固定 IP（当前） | `http://192.168.50.246:8080/v1` | LAN 内直连等价；IP 改变需同步更新 |

> 两台机器需在同一局域网；对端服务常驻（LaunchAgent 自拉起）。

## 2. Key（鉴权）

**当前无鉴权** —— 任何能访问该地址的客户端都可调用。

- `Authorization: Bearer <anything>` 不会被拒绝（服务端不校验），填任意值即可。
- ⚠️ 仅限可信局域网使用。若暴露公网，必须先开启鉴权（说一声即可加 API key 校验）。

接入配置里 Key 字段可填：`mlx-local` （占位，实际不校验）

## 3. 模型 ID

```
/Users/lingda-sj/.cache/hf-models/qwen36-35b-a3b-rq
```

（可用 `GET {base}/models` 实时查询当前全部模型 ID）

## 4. 支持的 API

| 端点 | 方法 | 状态 | 说明 |
|---|---|---|---|
| `/v1/models` | GET | ✅ 实测 200 | 列出模型 |
| `/v1/chat/completions` | POST | ✅ 实测 200（流式/非流式） | **主流聊天**，OpenAI 兼容 |
| `/v1/completions` | POST | ✅ 实测 200 | 纯文本补全（非对话） |
| `/v1/embeddings` | POST | ❌ 404 | 不支持 |
| `/health` | GET | ✅ 200 | 存活探针 |
| `/openapi.json` / `/docs` | GET | ❌ 404 | 无文档端点 |

## 5. chat/completions 请求格式

```bash
curl -X POST http://lingda-SJdeMacBook-Air.local:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/Users/lingda-sj/.cache/hf-models/qwen36-35b-a3b-rq",
    "messages": [
      {"role": "system", "content": "你是一名乐于助人的助手。"},
      {"role": "user", "content": "用一句话解释什么是 KV Cache"}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.9,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "stop": null,
    "stream": false
  }'
```

### 请求参数（均实测兼容）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 用上面列出的完整模型 ID |
| `messages` | array | 必填 | `role` ∈ system / user / assistant；支持多轮历史 |
| `temperature` | float | 0.0 | 采样温度 |
| `max_tokens` | int | — | 生成最大 token 数 |
| `top_p` | float | — | 核采样 |
| `presence_penalty` | float | — | 存在惩罚 |
| `frequency_penalty` | float | — | 频率惩罚 |
| `stop` | string/array | — | 停止生成词 |
| `stream` | bool | false | SSE 流式返回 |

### 流式响应（stream: true）

SSE 格式，`data:` 行逐块推送，结尾 `[DONE]`：

```
data: {"id": "chatcmpl-...", "choices": [{"delta": {"role": "assistant", "content": "KV Cache…"}}]}
data: {...}
data: [DONE]
```

`system_fingerprint` 字段含运行时版本（`0.31.3-0.32.2-macOS`）。

## 6. 响应格式（非流式）

```json
{
  "id": "chatcmpl-...",
  "model": "/Users/lingda-sj/.cache/hf-models/qwen36-35b-a3b-rq",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35}
}
```

## 7. 样例：OpenAI SDK / LangChain 接入

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://lingda-SJdeMacBook-Air.local:8080/v1",
    api_key="mlx-local",          # 未鉴权，占位即可
)
resp = client.chat.completions.create(
    model="/Users/lingda-sj/.cache/hf-models/qwen36-35b-a3b-rq",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)
for chunk in resp:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 8. 运维备注

- 服务常驻；异常退出自动拉起（LaunchAgent `com.lingdasj.mlxserver`，KeepAlive）。
- 日志：对端 `~/Library/Logs/mlx/server.{out,err}.log`
- 滑窗 `FA_WINDOW=2048`：模型只能看到最近 2048 token，超长上下文为近似。
- 本地调试用 GUI：`http://127.0.0.1:7860`（代理指向同一服务）。