from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
import httpx

# Создаем MCP-сервер
mcp = FastMCP("MCP Image Generation Server")


# Определяем инструмент generate-image с декоратором @mcp.tool()
@mcp.tool()
async def generate_image(
        prompt: str,
        style: str = "DEFAULT",
        width: int = 1024,
        height: int = 768,
        ctx: Context[ServerSession, None] = None
) -> dict:
    """Generate an image based on the prompt by calling an external API."""
    await ctx.info(f"Starting image generation for prompt: {prompt} (style: {style}, size: {width}x{height})")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://34bb7a428966.ngrok-free.app/api/generate-image",
                json={
                    "prompt": prompt,
                    "style": style,
                    "width": width,
                    "height": height
                }
            )
            response.raise_for_status()
            api_response = response.json()
            await ctx.debug(f"Image generated successfully: {api_response}")
            return api_response  # Возвращаем dict в формате {"success": true, "message": ..., "imageUrl": ..., "fileName": ...}
        except httpx.HTTPStatusError as e:
            error_msg = f"API error: {e.response.status_code} - {e.response.text}"
            await ctx.error(error_msg)
            return {"success": False, "message": error_msg}
        except Exception as e:
            error_msg = f"Internal error: {str(e)}"
            await ctx.error(error_msg)
            return {"success": False, "message": error_msg}
