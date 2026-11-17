from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta

import pytest

from apscheduler import JobOutcome, JobReleased
from apscheduler._structures import Job, JobResult, Task
from apscheduler.datastores.memory import MemoryDataStore
from apscheduler.eventbrokers.local import LocalEventBroker


@pytest.fixture
async def lease_store() -> AsyncGenerator[MemoryDataStore]:
    async with AsyncExitStack() as stack:
        broker = LocalEventBroker()
        await broker.start(stack, logging.getLogger("test"))
        store = MemoryDataStore()
        await store.start(stack, broker, logging.getLogger("test"))
        await store.add_task(
            Task(id="lease_task", func="asyncio:sleep", job_executor="async")
        )
        yield store


@pytest.mark.anyio
async def test_expired_lease_is_not_reacquired_by_same_running_scheduler(
    lease_store: MemoryDataStore,
) -> None:
    job = Job(task_id="lease_task", executor="async")
    await lease_store.add_job(job)
    acquired = await lease_store.acquire_jobs("worker1", timedelta(seconds=30), 1)
    assert len(acquired) == 1

    acquired[0].acquired_until = datetime.now(UTC) - timedelta(seconds=1)
    assert await lease_store.acquire_jobs("worker1", timedelta(seconds=30), 1) == []
    assert (await lease_store.get_task("lease_task")).running_jobs == 1

    await lease_store.release_job(
        "worker1", acquired[0], JobResult.from_job(job, JobOutcome.success)
    )
    assert (await lease_store.get_task("lease_task")).running_jobs == 0
    assert await lease_store.get_jobs() == []


@pytest.mark.anyio
async def test_old_owner_cannot_complete_reacquired_job(
    lease_store: MemoryDataStore,
) -> None:
    job = Job(
        task_id="lease_task",
        executor="async",
        result_expiration_time=timedelta(minutes=1),
    )
    await lease_store.add_job(job)
    old = await lease_store.acquire_jobs("worker1", timedelta(seconds=30), 1)
    assert len(old) == 1
    old[0].acquired_until = datetime.now(UTC) - timedelta(seconds=1)

    new = await lease_store.acquire_jobs("worker2", timedelta(seconds=30), 1)
    assert len(new) == 1
    assert (await lease_store.get_task("lease_task")).running_jobs == 1

    await lease_store.release_job(
        "worker1", old[0], JobResult.from_job(job, JobOutcome.error)
    )
    assert len(await lease_store.get_jobs()) == 1
    assert (await lease_store.get_task("lease_task")).running_jobs == 1
    assert await lease_store.get_job_result(job.id) is None

    await lease_store.release_job(
        "worker2", new[0], JobResult.from_job(job, JobOutcome.success)
    )
    assert await lease_store.get_jobs() == []
    assert (await lease_store.get_task("lease_task")).running_jobs == 0
    recorded = await lease_store.get_job_result(job.id)
    assert recorded is not None and recorded.outcome is JobOutcome.success


@pytest.mark.anyio
async def test_late_release_after_cleanup_is_idempotent(
    lease_store: MemoryDataStore,
) -> None:
    job = Job(
        task_id="lease_task",
        executor="async",
        result_expiration_time=timedelta(minutes=1),
    )
    await lease_store.add_job(job)
    acquired = await lease_store.acquire_jobs("worker1", timedelta(seconds=30), 1)
    acquired[0].acquired_until = datetime.now(UTC) - timedelta(seconds=1)
    await lease_store.cleanup()

    assert await lease_store.get_jobs() == []
    assert (await lease_store.get_task("lease_task")).running_jobs == 0

    await lease_store.release_job(
        "worker1", acquired[0], JobResult.from_job(job, JobOutcome.success)
    )
    assert (await lease_store.get_task("lease_task")).running_jobs == 0
    recorded = await lease_store.get_job_result(job.id)
    assert recorded is not None and recorded.outcome is JobOutcome.abandoned


@pytest.mark.anyio
async def test_scheduler_never_starts_duplicate_of_its_running_job() -> None:
    from unittest.mock import patch

    from anyio import Event, fail_after, sleep

    from apscheduler import AsyncScheduler

    store = MemoryDataStore()
    started = Event()
    finish = Event()
    repeated = Event()
    starts = 0

    async def slow_job() -> None:
        nonlocal starts
        starts += 1
        started.set()
        await finish.wait()

    async def wake_job() -> None:
        pass

    async with AsyncScheduler(data_store=store, max_concurrent_jobs=3) as scheduler:
        await scheduler.configure_task("slow_job", func=slow_job)
        await scheduler.configure_task("wake_job", func=wake_job)
        await scheduler.add_job("slow_job")
        await scheduler.start_in_background()
        with fail_after(2):
            await started.wait()

        original = next(iter(scheduler._running_jobs))

        async def duplicate_acquire(
            _store: MemoryDataStore,
            scheduler_id: str,
            lease_duration: timedelta,
            limit: int | None = None,
        ) -> list[Job]:
            repeated.set()
            return [original]

        with patch.object(MemoryDataStore, "acquire_jobs", duplicate_acquire):
            await scheduler.add_job("wake_job")
            with fail_after(2):
                await repeated.wait()
            await sleep(0.05)
            assert starts == 1, "the job already active in this process ran again"

        released = Event()
        with scheduler.subscribe(lambda _: released.set(), {JobReleased}):
            finish.set()
            with fail_after(2):
                await released.wait()

        await scheduler.stop()
