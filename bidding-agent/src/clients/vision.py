import asyncio
import base64
import binascii

from zai import ZhipuAiClient

from src.config import settings

# 默认提示词按招投标场景定制：上传的图片多是招标公告截图、证照、报价单，
# 泛泛地问"描述这张图"拿不到关键字段，故要求结构化输出。
DEFAULT_PROMPT = (
    "请解析这张图片中的内容，并遵守：\n"
    "1. 先说明图片类型（如招标公告截图、资质证照、报价单、表格、票据等）；\n"
    "2. 用 Markdown 表格或清单提取其中的关键信息，"
    "招投标类图片重点关注：项目名称、项目编号、采购人、代理机构、预算金额、"
    "开标时间、投标截止时间、资质要求、联系方式；\n"
    "3. 金额、日期、编号务必逐字核对，看不清的字段标注「无法辨认」，不要猜测；\n"
    "4. 最后用一句话概括图片主旨。"
)

# 图片魔数 → MIME。智谱接口需要 data URL 前缀，前缀写错会被服务端拒绝
# （早期实现把前缀写死成 image/jpeg，上传 PNG 时会因格式不符而失败）。
_MAGIC_TO_MIME = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def _detect_mime(raw: bytes) -> str:
    """按魔数判断图片类型；认不出来时回退 jpeg。

    为什么不用前端传来的 MIME：那是客户端自报的值，可被伪造且不一定与内容一致。
    后端按字节判断才是可信来源；认不出时回退 jpeg 是因为智谱对 jpeg 的兼容性最好。
    """
    for magic, mime in _MAGIC_TO_MIME:
        if raw.startswith(magic):
            return mime
    # WEBP：RIFF....WEBP
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _decode_image(image_base64: str) -> tuple[str, str]:
    """把入参归一为 `(纯 base64 串, mime)`。

    前端可能传两种形式，两种都要收：
      1. 纯 base64（既有 /api/vision 契约）
      2. 完整 data URL（`data:image/png;base64,xxx`）——前端用 canvas 转码后
         顺手带上前缀更自然，不该强迫它再剥一层

    顺手剥掉 base64 里的空白字符：某些复制/拼接路径会带入换行，
    而 `base64.b64decode` 默认对非法字符是抛异常而不是忽略。
    """
    text = (image_base64 or "").strip()
    if text.startswith("data:"):
        head, _, body = text.partition(",")
        text = body
        declared = head[5:].split(";")[0].strip()
    else:
        declared = ""

    cleaned = "".join(text.split())
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图片 base64 无法解码") from exc

    mime = _detect_mime(raw) if raw else (declared or "image/jpeg")
    return cleaned, mime


class VisionClient:
    """智谱视觉模型客户端（经 zai SDK 的 ZhipuAiClient 调用）。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.zhipu_api_key
        self._model = model or settings.zhipu_vision_model
        self._client = ZhipuAiClient(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "zhipu"

    @property
    def available(self) -> bool:
        """是否具备可用配置（视图层据此决定返回 503 还是尝试调用）。"""
        return bool(self._api_key and self._model)

    async def analyze(
        self,
        image_base64: str,
        prompt: str,
    ) -> str:
        """解析图片，返回模型输出的文本。

        `image_base64` 可为纯 base64 或完整 data URL；提示词为空时用场景化默认值。
        失败时抛异常（由 API 层转 500），不做静默降级——用户上传了图却收到
        一句"处理失败"之外的空回答，比报错更难排查。
        """
        cleaned, mime = _decode_image(image_base64)

        response = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            # 前缀必须与实际字节匹配（见 _detect_mime 的说明）
                            "image_url": {"url": f"data:{mime};base64,{cleaned}"},
                        },
                        {
                            "type": "text",
                            "text": (prompt or "").strip() or DEFAULT_PROMPT,
                        },
                    ],
                }
            ],
        )

        return response.choices[0].message.content or ""