import asyncio
from sqlalchemy import select
from src.infrastructure.database.session import AsyncSessionLocal
from src.infrastructure.database.models import UserModel
from src.core.security import get_password_hash

async def seed_demo_user():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(UserModel).where(UserModel.username == "johndoe"))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            print("Demo user already exists")
            return
        
        demo_user = UserModel(
            username="johndoe",
            email="johndoe@example.com",
            full_name="John Doe",
            hashed_password=get_password_hash("secret"),
            disabled=False
        )
        
        session.add(demo_user)
        await session.commit()
        print("✅ Demo user created: username=johndoe, password=secret")

async def main():
    from src.core.database import init_db
    
    print("Initializing database...")
    await init_db()
    print("✅ Database initialized")
    
    print("Seeding demo user...")
    await seed_demo_user()
    print("✅ Seed completed")

if __name__ == "__main__":
    asyncio.run(main())
