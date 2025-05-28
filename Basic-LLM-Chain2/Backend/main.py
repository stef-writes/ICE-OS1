import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from routers import debug_router, graph_management_router, script_execution_router, database_crud_router

# Create FastAPI app
app = FastAPI()

# Configure CORS
allow_origin_regex = r"^^https?://(localhost|127\.0\.0\.1)(:\d+)?$$" # Added $ to fix regex

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# --- API Routes ---
@app.get("/")
def read_root():
    return {"message": "ScriptChain Backend Running"}

# Include routers
# It's good practice to ensure that the paths in your routers don't clash
# and that the prefix helps in organizing them.
app.include_router(graph_management_router.router, prefix="/api/graph", tags=["Graph Management"])
app.include_router(script_execution_router.router, prefix="/api/execute", tags=["Script Execution"])
app.include_router(database_crud_router.router, prefix="/api/db", tags=["Database CRUD"])
app.include_router(debug_router.router, prefix="/api/debug", tags=["Debug"])


# --- Run Server ---
if __name__ == "__main__":
    print("Starting backend server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)