import asyncio
from nanobot.utils.vault_lock import get_vault_lock

async def test_same_slug_returns_same_lock():
    assert get_vault_lock("alice") is get_vault_lock("alice")

async def test_different_slugs_independent():
    assert get_vault_lock("alice") is not get_vault_lock("bob")

async def test_lock_serializes_critical_section():
    order = []
    async def worker(tag, delay):
        async with get_vault_lock("v"):
            order.append(("start", tag))
            await asyncio.sleep(delay)
            order.append(("end", tag))
    await asyncio.gather(worker("a", 0.02), worker("b", 0.0))
    assert order[0] == ("start", "a") and order[1] == ("end", "a")
