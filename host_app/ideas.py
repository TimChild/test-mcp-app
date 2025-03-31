from pydantic import BaseModel


class AnswerStream:
    async def astream(self):
        pass


class AnswerEvent(BaseModel):
    pass


async def call_assistant(question: str, user_id: str) -> AnswerStream:
    pass


class Answer:
    def add(self, event: AnswerEvent):
        pass


def test_call():
    answer: Answer = Answer()
    events: list[AnswerEvent] = []
    async for event in call_assistant().astream():
        assert isinstance(event, AnswerEvent)
        events.append(event)
        answer.add(event)

    assert answer == Answer()
