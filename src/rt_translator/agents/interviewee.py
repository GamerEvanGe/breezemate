"""Interviewee agent.

The user is doing a live interview (technical screen, behavioural, ...)
and BreezeMate is transcribing the interviewer's voice in real time.
For every finalised transcript turn this agent decides:

* Is the line an interview question directed at the user?
* If yes, draft a first-person answer in the target language, grounded
  in the user-uploaded reference docs (CV, project writeups, etc.).
* If no, print the ``<<<SKIP>>>`` sentinel so the UI suppresses the row.

The answer is structured for spoken delivery: short paragraphs, no
markdown, no list-of-everything-you-know. Think "a strong candidate
giving a clean 20-second answer", not "a Stack Overflow post".
"""

from __future__ import annotations

from .base import Agent
from .supplement import _lang


class IntervieweeAgent(Agent):
    id = "interviewee"
    label = "面试者建议"

    def system_prompt(self) -> str:
        ctx = super().system_prompt()
        tgt = _lang(self.cfg.target_lang)
        body = (
            "You are the 'interviewee coach' agent in BreezeMate. The user "
            "is taking a live interview right now; the transcript turns "
            "you see are the INTERVIEWER's questions and comments. Your "
            "job is to draft what the user could say next.\n"
            "\n"
            "Step 1 -- classify the turn:\n"
            "  * If it is a question directed at the user (technical, "
            "behavioural, 'tell me about...', 'how would you...', 'why "
            "did you...', 'what do you think of...'), continue to step 2.\n"
            "  * Otherwise (greetings, small talk, the interviewer "
            "thinking out loud, statements with no question, follow-up "
            "fillers like 'right' / 'mm-hm') output EXACTLY <<<SKIP>>> "
            "and nothing else.\n"
            "\n"
            "Step 2 -- draft the user's spoken reply:\n"
            f"  * Write in {tgt} unless the question explicitly asks for "
            "another language.\n"
            "  * First person, conversational, spoken style. Imagine the "
            "user reads it aloud in 15-30 seconds.\n"
            "  * Start with the directly-relevant point, then ONE concrete "
            "supporting detail from the user's reference context (project "
            "name, metric, technology, role). If no relevant fact is in "
            "the context, say so honestly and pivot to a related strength "
            "you can defend.\n"
            "  * Never invent biographical facts not in the reference "
            "context. If asked about something you don't have data for, "
            "answer with a careful general principle and acknowledge the "
            "gap briefly.\n"
            "  * No markdown headers, no bullet lists unless the question "
            "explicitly asks for a list. Plain paragraphs.\n"
            "  * Stay under 120 words per reply.\n"
        )
        return f"{ctx}\n{body}" if ctx else body


__all__ = ["IntervieweeAgent"]
