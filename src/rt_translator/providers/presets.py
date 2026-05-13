"""Built-in catalog of OpenAI-compatible chat API endpoints.

The translator (``providers/llm/openai_compatible.py``) talks the OpenAI
Chat Completions protocol, so swapping providers is purely a config
change: pick a base URL, an API key, and a model name. This module
ships sensible defaults for the most useful providers so the GUI can
populate a dropdown out of the box.

Speech recognition is NOT part of this catalog any more. It now runs
entirely offline through Vosk -- see ``providers/asr/vosk_model.py``
for the model catalog and ``providers/asr/vosk_full.py`` for the
``StreamingASRProvider`` implementation. The chat / translation
provider and the ASR engine are completely decoupled.

If your chat provider is not listed here, choose ``custom`` and fill
in the fields manually. The data model is intentionally tiny -- no
plugin registry, no late-binding magic -- so users can audit and edit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderPreset:
    """A named OpenAI-compatible chat endpoint preset.

    ``id`` is a stable machine identifier used in config files and the
    secret store. ``label`` is human-facing.
    """

    id: str
    label: str
    base_url: str
    # Conventional environment-variable name for this provider's API key.
    # Also used as the storage key inside ``secrets.json``.
    api_key_env: str
    # False for local providers like Ollama / LM Studio that do not
    # require authentication.
    auth_required: bool = True
    # Free-form note shown in the settings dialog (one or two lines).
    notes: str = ""
    # Chat models the user is most likely to want, pre-filled in
    # dropdowns.
    suggested_chat_models: tuple[str, ...] = field(default_factory=tuple)
    # Pre-filled URL for "where do I get a key?" -- opened from the
    # settings dialog. Empty string means "no link / not applicable".
    signup_url: str = ""


# Order matters: the GUI shows providers in this order. Put paid-but-
# proven first, then free / cheap alternatives, then local, then custom.
PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        id="openai",
        label="OpenAI (官方付费 / 精度最高)",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        notes=(
            "付费。Agent 推荐用 gpt-5 / gpt-5.5 等推理型模型以获得"
            "媲美 ChatGPT App 的回答质量；翻译用 gpt-4o-mini 即可。"
        ),
        suggested_chat_models=(
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-5-mini",
            "gpt-5",
            "gpt-5.5",
            "o3-mini",
            "o3",
        ),
        signup_url="https://platform.openai.com/api-keys",
    ),
    ProviderPreset(
        id="deepseek",
        label="DeepSeek (国内便宜)",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        notes="国内可访问，价格远低于 OpenAI。仅 chat，不提供 Realtime ASR。",
        suggested_chat_models=(
            "deepseek-chat",
            "deepseek-reasoner",
        ),
        signup_url="https://platform.deepseek.com/api_keys",
    ),
    ProviderPreset(
        id="groq",
        label="Groq (有免费额度 / 推理极快)",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        notes="免费额度可观，token 输出速度领先。注册即送 key（需海外网络）。",
        suggested_chat_models=(
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ),
        signup_url="https://console.groq.com/keys",
    ),
    ProviderPreset(
        id="zhipu",
        label="智谱 BigModel (国内 / glm-4-flash 完全免费)",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key_env="ZHIPU_API_KEY",
        notes=(
            "国内可直接访问。glm-4-flash 完全免费、无限调用，"
            "翻译质量足够日常使用。注册手机号即可拿 key。"
        ),
        suggested_chat_models=(
            "glm-4-flash",
            "glm-4-air",
            "glm-4-plus",
            "glm-4",
        ),
        signup_url="https://open.bigmodel.cn/usercenter/apikeys",
    ),
    ProviderPreset(
        id="siliconflow",
        label="SiliconFlow 硅基流动 (国内 / 部分模型免费)",
        base_url="https://api.siliconflow.cn/v1",
        api_key_env="SILICONFLOW_API_KEY",
        notes=(
            "国内可直接访问。带 (Free) 字样的模型免费。"
            "Qwen2.5-7B-Instruct 翻译质量与速度都不错。"
        ),
        suggested_chat_models=(
            "Qwen/Qwen2.5-7B-Instruct",
            "THUDM/glm-4-9b-chat",
            "Qwen/Qwen2.5-14B-Instruct",
            "deepseek-ai/DeepSeek-V2.5",
        ),
        signup_url="https://cloud.siliconflow.cn/account/ak",
    ),
    ProviderPreset(
        id="dashscope",
        label="阿里云 DashScope (国内 / Qwen 官方 / 有免费额度)",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        notes=(
            "Qwen 官方端点。每月百万 token 级免费额度，"
            "国内访问稳定。阿里云控制台开通 DashScope 拿 key。"
        ),
        suggested_chat_models=(
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen2.5-7b-instruct",
        ),
        signup_url="https://bailian.console.aliyun.com/?apiKey=1",
    ),
    ProviderPreset(
        id="openrouter",
        label="OpenRouter (模型聚合 / 含免费模型)",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        notes="一个 key 调用上百个模型。模型 id 带 ':free' 后缀的是免费档。",
        suggested_chat_models=(
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "deepseek/deepseek-chat",
            "qwen/qwen-2.5-72b-instruct:free",
        ),
        signup_url="https://openrouter.ai/keys",
    ),
    ProviderPreset(
        id="together",
        label="Together AI (有免费 tier)",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        notes="开源模型托管，新账户送试用额度。",
        suggested_chat_models=(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ),
        signup_url="https://api.together.xyz/settings/api-keys",
    ),
    ProviderPreset(
        id="ollama",
        label="Ollama (本地，免费 / 无需 key)",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        auth_required=False,
        notes=(
            "完全本地、无需 API Key。先到 ollama.com 安装客户端，"
            "运行 `ollama pull qwen2.5:7b` 拉模型，确保 Ollama 后台"
            "服务在 11434 端口可访问后即可使用。"
        ),
        suggested_chat_models=(
            "qwen2.5:7b",
            "qwen2.5:14b",
            "llama3.2:3b",
            "mistral:7b",
        ),
        signup_url="https://ollama.com/download",
    ),
    ProviderPreset(
        id="lmstudio",
        label="LM Studio (本地，免费 / 无需 key)",
        base_url="http://localhost:1234/v1",
        api_key_env="LMSTUDIO_API_KEY",
        auth_required=False,
        notes=(
            "桌面工具，无需 API Key。下载 LM Studio，加载一个 gguf "
            "模型，点 \"Start Server\" 后即可。"
        ),
        suggested_chat_models=("local-model",),
        signup_url="https://lmstudio.ai",
    ),
    ProviderPreset(
        id="custom",
        label="自定义 (任意 OpenAI 兼容端点)",
        base_url="",
        api_key_env="CUSTOM_API_KEY",
        notes="填入符合 OpenAI Chat Completions 协议的 base_url 即可。",
    ),
)


_BY_ID = {p.id: p for p in PRESETS}


def get_preset(preset_id: str) -> ProviderPreset | None:
    """Return the preset with that id, or None if unknown."""
    return _BY_ID.get(preset_id)


def chat_presets() -> list[ProviderPreset]:
    """All chat / translation presets, in display order."""
    return list(PRESETS)
