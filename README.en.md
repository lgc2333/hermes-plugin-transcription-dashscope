# transcription-dashscope

[简体中文](README.md)

Speech recognition for Hermes. Voice messages become text automatically. On the desktop app or the CLI, dictation shows words as you speak. Recognition runs on Qwen-Audio-ASR from Qwen AI Platform (formerly Alibaba Cloud Model Studio, DashScope). The default model is `qwen-audio-3.1-asr-flash-message`.

## Prepare

You need an API key from Qwen AI Platform. Put it in `.env`:

```bash
DASHSCOPE_API_KEY=sk-xxx
```

The plugin also reads `HERMES_CUSTOM_DASHSCOPE_API_KEY` and `TRANSCRIPTION_DASHSCOPE_API_KEY`.

You also need `ffmpeg`. The plugin converts every recording to 16 kHz mono before it sends the audio for recognition.

Enable the plugin, then restart the gateway so the new dependencies load:

```bash
hermes plugins enable transcription-dashscope
hermes gateway restart
```

## Turn on

```bash
hermes config set stt.provider dashscope
hermes config set stt.enabled true
hermes config set stt.streaming true
```

`stt.streaming` controls word-by-word output. Turn it off if you only want voice messages transcribed.

You usually do not need the settings below. Change them with `hermes config set` when you do:

```yaml
stt:
  language: zh
  dashscope:
    model: qwen-audio-3.1-asr-flash-message
    base_url: https://maas.qianwenaiapi.com/api/v1
    workspace: ''
```

`base_url` accepts a plain HTTP address. The plugin converts it to the realtime endpoint itself. OpenAI-compatible addresses such as `/compatible-mode/v1` also work, because the SDK reads only the host name.

The plugin runs without any key or address set. It reads `TRANSCRIPTION_DASHSCOPE_*` (this plugin), then `HERMES_CUSTOM_DASHSCOPE_*` (a custom provider entry), then `DASHSCOPE_*` (the built-in DashScope provider). One working set is enough. The address can also be set in `config.yaml`, which wins over all of them. `workspace` is read only from the config or from `TRANSCRIPTION_DASHSCOPE_WORKSPACE`; set it to use the workspace gateway.

Keys under `stt.dashscope` match the API parameters, with three exceptions. `format` and `sample_rate` are fixed by the plugin, because audio is converted to 16 kHz mono first. `api_key` is read from the environment only. `base_url`, `language` and `parameters` belong to the plugin. Every other key passes through unchanged, for example polish, VAD sentence breaking and hot words:

```bash
hermes config set stt.dashscope.disfluency_removal_enabled true
hermes config set stt.dashscope.max_sentence_silence 400
hermes config set stt.dashscope.vad_model near_meeting_16k
```

Other supported keys are `keep_dialect` (keep the original dialect text), `speech_noise_threshold`, `vocabulary_id` (precompiled hot words) and `language_hints` (up to 4 languages). For an uncommon parameter, pass the whole dict as `parameters`.

## Data

The plugin sends the audio and your API key to the endpoint you configure. It sends nothing else.

## Restore

```bash
hermes config set stt.provider local
hermes plugins disable transcription-dashscope
```

## When something fails

- `DashScope SDK is not importable in this process`: the dependencies are installed, but the running process has not switched to them yet. Restart the gateway.
- `Arrearage: Access denied`: the account is out of credit, or the model is not activated. Fix it in the Qwen AI Platform console.
- You get interim results and do not want them: `hermes config set stt.streaming false`.
- Qwen marks the SDK 2.0 speech API as preview. After you upgrade the SDK, send one voice message to confirm it still works.

## Limits

- The model targets short audio and session-level recognition. For very long recordings, use a vendor's file-transcription model.
- The realtime API accepts 16 kHz only. The plugin transcodes other formats first.
