from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import endpoints
from endpoints import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

#app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
