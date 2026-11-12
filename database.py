import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- Load environment variables from .env file ---
load_dotenv()

# --- Read DATABASE_URL from .env ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Engine = connection to PostgreSQL ---
engine = create_engine(DATABASE_URL)

# --- Session = used to talk to the database ---
SessionLocal = sessionmaker(bind=engine)

# --- Base = all our tables will inherit from this ---
Base = declarative_base()


# --- User Table (for login) ---
class UserTable(Base):
    __tablename__ = "users"

    id       = Column(Integer, primary_key=True, index=True)
    email    = Column(String, unique=True, index=True)
    password = Column(String)  # stored as hashed password


# --- Employee Table ---
class EmployeeTable(Base):
    __tablename__ = "employees"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String)
    email      = Column(String)
    department = Column(String)
    role       = Column(String)
    salary     = Column(Float)
    join_date  = Column(Date)


# --- Create all tables if they don't exist ---
Base.metadata.create_all(bind=engine)
