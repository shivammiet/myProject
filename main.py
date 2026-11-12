from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from datetime import date
from database import SessionLocal, EmployeeTable, UserTable
from auth import hash_password, verify_password, create_token, verify_token

# --- App Setup ---
app = FastAPI(title="Employee Portal", version="3.0.0")


# --- Database session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------
# MODELS
# -----------------------------------------------

class Employee(BaseModel):
    name: str       = Field(min_length=2, max_length=50)
    email: EmailStr
    department: str = Field(min_length=2)
    role: str       = Field(min_length=2)
    salary: float   = Field(gt=0)
    join_date: date

    class Config:
        from_attributes = True


class RegisterUser(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)  # password must be 6+ chars


# -----------------------------------------------
# AUTH ROUTES (no login needed)
# -----------------------------------------------

# REGISTER - Create a new user account
@app.post("/register")
def register(user: RegisterUser, db: Session = Depends(get_db)):
    # Check if email already exists
    existing = db.query(UserTable).filter(UserTable.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash the password before saving
    new_user = UserTable(
        email=user.email,
        password=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    return {"message": "Account created successfully!"}


# LOGIN - Get a token
@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Find user by email
    user = db.query(UserTable).filter(UserTable.email == form.username).first()

    # Check password
    if not user or not verify_password(form.password, user.password):
        raise HTTPException(status_code=401, detail="Wrong email or password")

    # Create and return token
    token = create_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}


# -----------------------------------------------
# EMPLOYEE ROUTES (login required 🔐)
# -----------------------------------------------

# CREATE - Add a new employee
@app.post("/employees")
def create_employee(employee: Employee, db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    new_emp = EmployeeTable(
        name=employee.name,
        email=employee.email,
        department=employee.department,
        role=employee.role,
        salary=employee.salary,
        join_date=employee.join_date
    )
    db.add(new_emp)
    db.commit()
    db.refresh(new_emp)
    return new_emp


# READ ALL - Get all employees
@app.get("/employees")
def get_all_employees(db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    return db.query(EmployeeTable).all()


# READ ONE - Get employee by ID
@app.get("/employees/{id}")
def get_employee(id: int, db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    emp = db.query(EmployeeTable).filter(EmployeeTable.id == id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp


# UPDATE - Update employee by ID
@app.put("/employees/{id}")
def update_employee(id: int, updated: Employee, db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    emp = db.query(EmployeeTable).filter(EmployeeTable.id == id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    emp.name       = updated.name
    emp.email      = updated.email
    emp.department = updated.department
    emp.role       = updated.role
    emp.salary     = updated.salary
    emp.join_date  = updated.join_date

    db.commit()
    db.refresh(emp)
    return emp


# DELETE - Remove employee by ID
@app.delete("/employees/{id}")
def delete_employee(id: int, db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    emp = db.query(EmployeeTable).filter(EmployeeTable.id == id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    db.delete(emp)
    db.commit()
    return {"message": f"Employee {id} deleted successfully"}
