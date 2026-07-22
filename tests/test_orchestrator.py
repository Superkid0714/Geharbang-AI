import unittest
from unittest.mock import patch

from src.chat.orchestrator import answer_chat
from src.chat.schemas import ChatDomain


class ChatOrchestratorTest(unittest.TestCase):
    @patch("src.chat.orchestrator.answer_general_chat")
    @patch("src.chat.orchestrator.route_chat_query")
    def test_unclear_query_asks_for_clarification_without_llm(
        self,
        mock_route_chat_query,
        mock_answer_general_chat,
    ) -> None:
        from src.chat.schemas import RouteDecision

        mock_route_chat_query.return_value = RouteDecision(
            ChatDomain.UNCLEAR,
            1.0,
            "gibberish",
        )

        response, state = answer_chat("ㅁㄴㅇㄹ")

        mock_answer_general_chat.assert_not_called()
        self.assertEqual(response.domain, ChatDomain.UNCLEAR)
        self.assertIn("조금 더 풀어서", response.answer)
        self.assertIn("같이 고민해볼게요", response.answer)
        self.assertIsNone(state.active_domain)

    @patch("src.chat.orchestrator.answer_jeju_travel")
    @patch("src.chat.orchestrator.has_jeju_travel_index", return_value=True)
    @patch("src.chat.orchestrator.route_chat_query")
    def test_jeju_travel_query_uses_pdf_rag(
        self,
        mock_route_chat_query,
        _mock_has_index,
        mock_answer_jeju_travel,
    ) -> None:
        from src.chat.schemas import RouteDecision

        mock_route_chat_query.return_value = RouteDecision(
            ChatDomain.JEJU_TRAVEL,
            0.96,
            "keyword",
        )
        mock_answer_jeju_travel.return_value = "제주 동쪽 코스를 안내해 드릴게요."

        response, state = answer_chat("제주 동쪽 여행 코스 추천해줘")

        mock_answer_jeju_travel.assert_called_once_with("제주 동쪽 여행 코스 추천해줘")
        self.assertEqual(response.domain, ChatDomain.JEJU_TRAVEL)
        self.assertEqual(state.active_domain, ChatDomain.JEJU_TRAVEL)

    @patch("src.chat.orchestrator.answer_general_chat")
    @patch("src.chat.orchestrator.route_chat_query")
    def test_out_of_scope_query_is_answered_by_general_llm(
        self,
        mock_route_chat_query,
        mock_answer_general_chat,
    ) -> None:
        from src.chat.schemas import RouteDecision

        mock_route_chat_query.return_value = RouteDecision(
            ChatDomain.OUT_OF_SCOPE,
            0.91,
            "unrelated",
        )
        mock_answer_general_chat.return_value = "양자역학은 미시 세계를 설명하는 이론입니다."

        response, state = answer_chat("양자역학을 설명해줘")

        mock_answer_general_chat.assert_called_once_with("양자역학을 설명해줘")
        self.assertEqual(response.answer, "양자역학은 미시 세계를 설명하는 이론입니다.")
        self.assertEqual(response.domain, ChatDomain.OUT_OF_SCOPE)
        self.assertIsNone(state.active_domain)

    @patch("src.chat.orchestrator.answer_general_chat")
    @patch("src.chat.orchestrator.route_chat_query")
    def test_greeting_does_not_call_general_llm(
        self,
        mock_route_chat_query,
        mock_answer_general_chat,
    ) -> None:
        from src.chat.schemas import RouteDecision

        mock_route_chat_query.return_value = RouteDecision(
            ChatDomain.GREETING,
            1.0,
            "greeting",
        )

        response, _ = answer_chat("안녕하세요")

        mock_answer_general_chat.assert_not_called()
        self.assertEqual(response.domain, ChatDomain.GREETING)
        self.assertIn("같이 고민해볼까요", response.answer)
        self.assertIn("숙소도, 스텝 생활도, 제주 여행도", response.answer)


if __name__ == "__main__":
    unittest.main()
