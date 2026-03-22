import asyncio
from src.infrastructures.database.session import AsyncSessionLocal
from src.core.security import get_password_hash
from src.repositories.user import UserRepository


async def seed_demo_user():
    async with AsyncSessionLocal() as session:
        repo = UserRepository(session)
        existing_user = await repo.get_by_username("johndoe")

        if existing_user:
            print("Demo user already exists")
            return

        await repo.create({
            "username": "johndoe",
            "email": "johndoe@example.com",
            "hashed_password": get_password_hash("secret"),
            "disabled": False,
        })
        print("Demo user created: username=johndoe, password=secret")

async def main():
    from src.infrastructures.database.session import init_db

    print("Initializing database...")
    await init_db()
    print("Database initialized")

    print("Seeding demo user...")
    await seed_demo_user()
    print("Seed completed")

if __name__ == "__main__":
    asyncio.run(main())
