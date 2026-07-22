import os
import unittest
from unittest.mock import patch

from src.chat.router import route_chat_query
from src.chat.schemas import ChatDomain, RouteDecision


class ChatRouterTest(unittest.TestCase):
    def route(self, query: str, active_domain: ChatDomain | None = None) -> ChatDomain:
        return route_chat_query(query, active_domain, use_llm_fallback=False).domain

    def test_guesthouse_queries(self) -> None:
        self.assertEqual(self.route("애월에 조용한 게하 추천해줘"), ChatDomain.GUESTHOUSE)
        self.assertEqual(self.route("여성 도미토리 객실 가격 알려줘"), ChatDomain.GUESTHOUSE)

    def test_staff_step_queries_win_over_guesthouse_word(self) -> None:
        self.assertEqual(self.route("제주 게하 스텝 공고 찾아줘"), ChatDomain.STAFF_STEP)
        self.assertEqual(self.route("게스트하우스에서 한 달 일하고 싶어"), ChatDomain.STAFF_STEP)
        self.assertEqual(self.route("애월 근무 조건과 휴무 알려줘"), ChatDomain.STAFF_STEP)

    def test_reserved_domains(self) -> None:
        self.assertEqual(self.route("협재 흑돼지 맛집 알려줘"), ChatDomain.JEJU_TRAVEL)
        self.assertEqual(self.route("게하르방에서 찜은 어떻게 해?"), ChatDomain.GEHARBANG_SERVICE)
        self.assertEqual(self.route("게하르방에서 스텝 지원은 어떻게 해?"), ChatDomain.GEHARBANG_SERVICE)
        self.assertEqual(self.route("지원서 수정은 어디서 해?"), ChatDomain.GEHARBANG_SERVICE)

    def test_follow_up_keeps_active_domain(self) -> None:
        self.assertEqual(
            self.route("그 공고 근무 조건은?", ChatDomain.STAFF_STEP),
            ChatDomain.STAFF_STEP,
        )
        self.assertEqual(
            self.route("거기 가격은?", ChatDomain.GUESTHOUSE),
            ChatDomain.GUESTHOUSE,
        )

    def test_unknown_query_is_out_of_scope(self) -> None:
        self.assertEqual(self.route("양자역학을 설명해줘"), ChatDomain.OUT_OF_SCOPE)

    @patch("src.chat.router._route_with_gemini")
    def test_generic_interview_worry_is_not_staff_step(self, mock_route_with_gemini) -> None:
        mock_route_with_gemini.return_value = RouteDecision(
            ChatDomain.STAFF_STEP,
            0.91,
            "면접이라는 단어를 스텝 질문으로 오인",
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            decision = route_chat_query("다음 주 면접인데 너무 긴장돼. 어떻게 준비할까?")

        self.assertEqual(decision.domain, ChatDomain.OUT_OF_SCOPE)
        self.assertEqual(decision.reason, "generic_work_or_personal_question")

    def test_gibberish_query_is_unclear(self) -> None:
        for query in ["ㅋㅋㅋㅋㅋㅋ", "ㅁㄴㅇㄹ", "asdfasdf", "!@#$%^"]:
            with self.subTest(query=query):
                self.assertEqual(self.route(query), ChatDomain.UNCLEAR)


if __name__ == "__main__":
    unittest.main()
