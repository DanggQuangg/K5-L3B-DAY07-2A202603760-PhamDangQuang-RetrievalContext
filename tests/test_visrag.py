"""Unit tests for VisRAG and Multimodal Agent components."""
import unittest
from unittest.mock import MagicMock, patch
from src.visrag import VisualDocumentInspector
from src.visrag_agent import VisRAGController, run_visrag_agent


class VisRAGTests(unittest.TestCase):
    """Test VisRAG inspector and agent loop."""

    @patch("src.visrag.pdfium.PdfDocument")
    @patch("src.visrag.OpenAI")
    def test_inspector_initialization(self, mock_openai, mock_pdf):
        mock_pdf_instance = MagicMock()
        mock_pdf_instance.__len__.return_value = 48
        mock_pdf.return_value = mock_pdf_instance

        inspector = VisualDocumentInspector(pdf_path="data/NIST_AI_100_1.pdf", api_key="test_key")
        self.assertEqual(inspector.total_pages, 48)

    def test_agent_abstains_on_empty_citations(self):
        controller = MagicMock()
        controller.decide.return_value = {
            "thought": "No evidence found",
            "action": "answer",
            "answer": "Không đủ bằng chứng để trả lời.",
            "citations": []
        }
        retriever = MagicMock()
        inspector = MagicMock()

        res = run_visrag_agent(
            question="Unanswerable query",
            controller=controller,
            text_retriever=retriever,
            visual_inspector=inspector
        )
        self.assertEqual(res["status"], "abstain")
        self.assertEqual(res["citations"], [])

    def test_agent_routes_to_visual_inspection_and_cites(self):
        controller = MagicMock()
        # Turn 1: Inspect visual page 16
        controller.decide.side_effect = [
            {
                "thought": "Need to look at Figure 3 on page 16",
                "action": "inspect_visual_page",
                "page": 16,
                "visual_query": "AI actors"
            },
            # Turn 2: Final answer citing vis_page_16
            {
                "thought": "Visual inspection provided enough detail",
                "action": "answer",
                "answer": "According to the visual diagram [vis_page_16], AI actors are separated into stages.",
                "citations": ["vis_page_16"]
            }
        ]

        retriever = MagicMock()
        inspector = MagicMock()
        inspector.inspect_visual_page.return_value = {
            "id": "vis_page_16",
            "source_id": "nist_ai_rmf_1_0",
            "source": "data/NIST_AI_100_1.pdf",
            "page": 16,
            "text": "[VISUAL INSPECTION - PAGE 16]\nFigure 3 outlines AI lifecycle stages.",
            "metadata": {"type": "visual_page_inspection"}
        }

        res = run_visrag_agent(
            question="What is shown in Figure 3?",
            controller=controller,
            text_retriever=retriever,
            visual_inspector=inspector
        )

        self.assertEqual(res["status"], "answered")
        self.assertIn("vis_page_16", res["citations"])
        self.assertEqual(res["visual_pages_inspected"], [16])
        self.assertEqual(inspector.inspect_visual_page.call_count, 1)


if __name__ == "__main__":
    unittest.main()
