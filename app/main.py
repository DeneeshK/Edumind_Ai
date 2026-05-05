"""
app/main.py
EduMind entry point.

Usage:
  New student:      python -m app.main --new
  Returning student: python -m app.main --student-id <id>
"""

from __future__ import annotations

import asyncio
import argparse
import uuid

from dotenv import load_dotenv
load_dotenv()

from db.postgres import init_db, close_db
from core.student_model import StudentState
from agents.orchestrator import OrchestratorAgent
from loguru import logger


async def main():
    parser = argparse.ArgumentParser(description="EduMind Adaptive Learning System")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--new", action="store_true", help="Start as new student")
    group.add_argument("--student-id", type=str, help="Returning student ID")
    args = parser.parse_args()

    await init_db()

    try:
        if args.new:
            student_id = str(uuid.uuid4())
            print(f"\n📋 Your student ID (save this!): {student_id}\n")
            state = StudentState(
                student_id=student_id,
                domain="",
                goal="",
                pace="medium",
            )
            orchestrator = OrchestratorAgent(state)
            await orchestrator.run_session(student_id, is_new=True)

        else:
            student_id = args.student_id
            try:
                state = await StudentState.load(student_id)
            except ValueError:
                print(f"❌ Student '{student_id}' not found. Use --new for first session.")
                return
            orchestrator = OrchestratorAgent(state)
            await orchestrator.run_session(student_id, is_new=False)

    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
