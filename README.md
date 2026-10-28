# transcription-dashscope

给 Hermes 加语音识别，收到语音消息时自动转成文字，桌面或 CLI 听写时边说边出字。识别走千问AI平台（原阿里云百炼，DashScope）的 Qwen-Audio-ASR，默认模型 `qwen-audio-3.1-asr-flash-message`。

## 准备

一个千问AI平台的 API Key，写进 `.env`（也认 `DASHSCOPE_API_KEY`）：

```bash
HERMES_CUSTOM_DASHSCOPE_API_KEY=sk-xxx
```

系统里要有 `ffmpeg`，插件用它把各种录音格式统一成 16 kHz 单声道再送去识别。

装好插件并重启网关，让新依赖生效：

```bash
hermes plugins enable transcription-dashscope
hermes gateway restart
```

## 打开

```bash
hermes config set stt.provider dashscope
hermes config set stt.enabled true
hermes config set stt.streaming true
```

`stt.streaming` 控制边说边出字，只想要语音消息转文字可以不开。

以下设置一般不用动，需要时再用 `hermes config set` 改：

```yaml
stt:
  language: zh
  dashscope:
    model: qwen-audio-3.1-asr-flash-message
    base_url: https://maas.qianwenaiapi.com/api/v1
    workspace: ''
```

`base_url` 填 HTTP 地址就行，插件会自己换成实时接口地址；`/compatible-mode/v1` 这种 OpenAI 兼容地址也能用，SDK 只看主机名。Key 和地址都不配也能跑：它们依次读 `TRANSCRIPTION_DASHSCOPE_*`（本插件专用）、`HERMES_CUSTOM_DASHSCOPE_*`（自定义 provider 那套）、`DASHSCOPE_*`（内置 DashScope 那套），任意一套配好就可用。地址还可以在 `config.yaml` 里覆盖，优先级最高；`workspace` 只从配置或 `TRANSCRIPTION_DASHSCOPE_WORKSPACE` 读，填了才走工作空间网关。

`stt.dashscope` 下的键名和接口参数一致，除三处例外：`format`、`sample_rate` 由插件固定（音频先转成 16 kHz 单声道），`api_key` 只从环境变量读，`base_url`、`language`、`parameters` 是插件自己用的。其余原样透传，比如润色、VAD 断句和热词列表：

```bash
hermes config set stt.dashscope.disfluency_removal_enabled true
hermes config set stt.dashscope.max_sentence_silence 400
hermes config set stt.dashscope.vad_model near_meeting_16k
```

还能用 `keep_dialect`（保留方言原文）、`speech_noise_threshold`（语音/噪音阈值）、`vocabulary_id`（预编译热词）、`language_hints`（语种，最多 4 个）；冷门参数可以用 `parameters` 整体塞一个字典。

## 还原

```bash
hermes config set stt.provider local
hermes plugins disable transcription-dashscope
```

## 出问题时

- 提示 `DashScope SDK is not importable in this process`：依赖装好了但当前进程还没切过去，重启网关。
- 提示 `Arrearage: Access denied`：账号欠费或该模型未开通，去千问AI平台控制台处理。
- 已经在出中间结果但不想看到：`hermes config set stt.streaming false`。
- SDK 2.0 的语音接口官方标注为 preview，升级 SDK 后先发一条语音消息确认一次。

## 限制

- 模型面向短音频和会话级识别，超长录音请换厂商的文件转写模型。
- 实时接口只收 16 kHz 采样率，其他格式由插件先转码。
