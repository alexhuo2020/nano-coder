"""Tests for chat-vs-agent prompt routing.

Run: python serve/test_chat_routing.py

This decides which system prompt EVERY request gets, so it needs tests. The
asymmetry matters: mistaking a repo task for chat breaks the agent loop that
works, while mistaking chat for a repo task only reproduces the old behaviour
(answering "who are you" by reading solution.py). So the repo-task bar is
deliberately low, and these tests pin that down in both directions.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic_shim as S


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


def test_conversational_questions_are_not_repo_tasks():
    for q in ("who are you",
              "hello",
              "what can you do?",
              "explain recursion to me",
              "what is your name"):
        assert not S.looks_like_a_repo_task(user(q)), q
    print("conversational questions route to CHAT: PASS")


def test_repo_tasks_are_detected():
    for q in ("The file solution.py in this repo is failing its tests. Fix it.",
              "fix the failing test",
              "run_tests and tell me what happens",
              "implement count_Unset_Bits",
              "write a function to add two numbers",
              "debug this",
              "why does the assert fail"):
        assert S.looks_like_a_repo_task(user(q)), q
    print("repo tasks route to AGENT: PASS")


def test_tool_traffic_pins_the_conversation_to_agent_mode():
    """Once the loop is running it must keep the agent prompt, or the model
    loses the tool convention mid-episode."""
    body = {"messages": [
        {"role": "user", "content": "hello"},          # would be chat alone
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
    ]}
    assert S.looks_like_a_repo_task(body)

    body = {"messages": [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "x"}]},
    ]}
    assert S.looks_like_a_repo_task(body)
    print("tool traffic pins the conversation to AGENT: PASS")


def test_block_form_content_is_read():
    """Claude Code sends content as blocks, not a bare string."""
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "fix solution.py please"}]}]}
    assert S.looks_like_a_repo_task(body)
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "who are you"}]}]}
    assert not S.looks_like_a_repo_task(body)
    print("block-form content is inspected: PASS")


def test_empty_and_malformed_bodies_do_not_raise():
    for body in ({}, {"messages": []}, {"messages": [{"role": "user"}]},
                 {"messages": [{"role": "user", "content": None}]},
                 {"messages": [{"role": "assistant", "content": "fix it"}]}):
        S.looks_like_a_repo_task(body)   # must not raise
    # an assistant-only mention is not a user request
    assert not S.looks_like_a_repo_task(
        {"messages": [{"role": "assistant", "content": "fix solution.py"}]})
    print("empty/malformed bodies are safe: PASS")


if __name__ == "__main__":
    test_conversational_questions_are_not_repo_tasks()
    test_repo_tasks_are_detected()
    test_tool_traffic_pins_the_conversation_to_agent_mode()
    test_block_form_content_is_read()
    test_empty_and_malformed_bodies_do_not_raise()
    print("\nAll chat-routing tests passed.")
