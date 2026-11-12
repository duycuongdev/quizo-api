from fastapi import APIRouter

from app.api.v1 import auth, categories, documents, health, questions, quizzes

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(categories.router)
api_router.include_router(questions.router)
api_router.include_router(quizzes.router)
api_router.include_router(documents.router)
