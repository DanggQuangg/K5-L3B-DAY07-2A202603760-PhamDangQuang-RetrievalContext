"""VisRAG (Visual Document Retrieval-Augmented Generation) Module.

Combines high-resolution PDF page rendering with Vision Language Models (VLMs)
to inspect visual artifacts (figures, flowcharts, tables, diagrams) that cannot
be adequately captured by standard text extraction.
"""
import base64
import io
import json
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
import pypdfium2 as pdfium

load_dotenv()


class VisualDocumentInspector:
    """Renders PDF pages into images and performs multi-modal visual inspection."""

    def __init__(
        self,
        pdf_path: str = "data/NIST_AI_100_1.pdf",
        cache_dir: str = "artifacts/page_cache",
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found for Vision LLM.")

        if base_url:
            self.base_url = base_url
        elif os.getenv("OPENROUTER_API_KEY"):
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.base_url = None

        self.model_name = model_name or os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Open PDF document
        self.doc = pdfium.PdfDocument(str(self.pdf_path))
        self.total_pages = len(self.doc)

    def render_page_to_jpeg_bytes(self, physical_page: int, scale: float = 1.5) -> bytes:
        """Render physical page (1-based index) into JPEG bytes, with local caching."""
        if physical_page < 1 or physical_page > self.total_pages:
            raise ValueError(f"Page number {physical_page} out of bounds (1..{self.total_pages})")

        cached_file = self.cache_dir / f"page_{physical_page:03d}.jpg"
        if cached_file.exists():
            return cached_file.read_bytes()

        # Render with pypdfium2 (0-based index)
        page = self.doc[physical_page - 1]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        img_bytes = buf.getvalue()

        # Cache on disk
        cached_file.write_bytes(img_bytes)
        return img_bytes

    def inspect_visual_page(self, physical_page: int, query: str) -> dict[str, Any]:
        """Perform VLM visual analysis on a specific PDF page image."""
        img_bytes = self.render_page_to_jpeg_bytes(physical_page)
        b64_image = base64.b64encode(img_bytes).decode("utf-8")

        prompt = (
            f"You are a specialized Visual Document Inspection agent analyzing Page {physical_page} of NIST AI RMF 1.0.\n"
            f"Target Query / Focus: {query}\n\n"
            "Instructions:\n"
            "1. Carefully inspect all visual diagrams, figures, charts, tables, flowcharts, or structural layouts on this page.\n"
            "2. Extract the exact text, node labels, arrows, lifecycle stages, actor roles, or table columns related to the query.\n"
            "3. Provide a clear, factual, and strictly grounded summary of what is visually shown on this page.\n"
            "4. If this page does NOT contain any relevant visual figure or information matching the query, state that clearly."
        )

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=400,
            temperature=0.0
        )

        visual_findings = resp.choices[0].message.content or ""
        chunk_id = f"vis_page_{physical_page:02d}"

        return {
            "id": chunk_id,
            "source_id": "nist_ai_rmf_1_0",
            "source": str(self.pdf_path).replace("\\", "/"),
            "page": physical_page,
            "query": query,
            "text": f"[VISUAL INSPECTION - PAGE {physical_page} - NIST AI RMF]\n{visual_findings.strip()}",
            "visual_summary": visual_findings.strip(),
            "metadata": {
                "type": "visual_page_inspection",
                "physical_page": physical_page,
                "strategy": "visrag"
            }
        }
