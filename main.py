from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import secrets
import os
import multiprocessing
import contextlib

from endpoints import router
from job_store import init_db
from worker import worker_loop

# Initialize DB on import (or startup)
init_db()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    queue = multiprocessing.Queue()
    app.state.queue = queue
    
    # Start worker(s)
    # User requested ability to scale later. We can easily change '1' to N or env var.
    num_workers = int(os.getenv("NUM_WORKERS", "1"))
    processes = []
    
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker_loop, args=(queue,))
        p.start()
        processes.append(p)
        
    yield
    
    # Shutdown
    for p in processes:
        p.terminate()
        p.join()

app = FastAPI(lifespan=lifespan)
security = HTTPBasic()

APP_PASSWORD = os.getenv("APP_PASSWORD")

def check_password(credentials: HTTPBasicCredentials = Depends(security)):
    correct_password = APP_PASSWORD

    is_correct = secrets.compare_digest(
        credentials.password,
        correct_password
    )

    if not is_correct:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# Home page (protected)
@app.get("/")
def home(_: None = Depends(check_password)):
    return FileResponse("static/index.html")

@app.get("/guide")
def guide(_: None = Depends(check_password)):
    return FileResponse("static/guide.html")


# API routes (protected)
app.include_router(
    router,
    dependencies=[Depends(check_password)]
)


#app.mount("/results", StaticFiles(directory="results"), name="results")
