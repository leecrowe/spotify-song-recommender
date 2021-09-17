import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.ml import prediction


# Instantiate fastAPI with appropriate descriptors
app = FastAPI(
    title='Spotify App API',
    description="Predicts songs",
    version="1.0",
    docs_url='/docs'
)

# Instantiate templates path
templates = Jinja2Templates(directory="app/templates")

# Mounts static files to specific routes for easier reference
app.mount("/assets",
          StaticFiles(directory="app/templates/assets"),
          name="assets"
          )

app.mount("/images",
          StaticFiles(directory="app/templates/images"),
          name="images"
          )

app.mount("/data",
          StaticFiles(directory="app/data/"),
          name="data"
          )

# Route for laning page
@app.get('/', response_class=HTMLResponse)
def display_index(request: Request):
    """Displays index.html from templates when user loads root URL"""
    return templates.TemplateResponse('index.html', {"request": request})

# Include router from prediction model
app.include_router(prediction.router, tags=['Machine Learning'])

# Allow access between the front end and back end from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Run `uvicorn app.main:app` to start ASGI server
if __name__ == '__main__':
    uvicorn.run(app)