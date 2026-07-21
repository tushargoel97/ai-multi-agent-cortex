import asyncio
import logging

logger = logging.getLogger(__name__)
tasks: set[asyncio.Task] = set()


def start(coroutine, *, label: str) -> None:
    task = asyncio.create_task(coroutine, name=label)
    tasks.add(task)

    def finished(done: asyncio.Task) -> None:
        tasks.discard(done)
        if not done.cancelled() and (error := done.exception()) is not None:
            logger.error("Background task %s failed: %s", label, error)

    task.add_done_callback(finished)
