"""enforce unique email on users

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-22 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Normalize email casing so uniqueness is consistent.
    op.execute("UPDATE users SET email = lower(email)")

    # Keep the oldest account per email, reattach requests from duplicates,
    # then remove duplicate user rows.
    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                email,
                ROW_NUMBER() OVER (
                    PARTITION BY email
                    ORDER BY created_at ASC, id ASC
                ) AS rn,
                FIRST_VALUE(id) OVER (
                    PARTITION BY email
                    ORDER BY created_at ASC, id ASC
                ) AS keep_id
            FROM users
        ),
        duplicates AS (
            SELECT id, keep_id
            FROM ranked
            WHERE rn > 1
        )
        UPDATE requests r
        SET user_id = d.keep_id
        FROM duplicates d
        WHERE r.user_id = d.id
        """
    )

    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                email,
                ROW_NUMBER() OVER (
                    PARTITION BY email
                    ORDER BY created_at ASC, id ASC
                ) AS rn
            FROM users
        )
        DELETE FROM users u
        USING ranked r
        WHERE u.id = r.id AND r.rn > 1
        """
    )

    op.create_unique_constraint("uq_users_email", "users", ["email"])
    op.create_index("ix_users_email", "users", ["email"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_users_email", table_name="users")
    op.drop_constraint("uq_users_email", "users", type_="unique")
