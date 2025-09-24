"""
Clarification router for handling dynamic user interactions.
"""

import asyncio
import uuid
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/clarification", tags=["clarification"])

# Store for pending clarification requests
# In production, this should be replaced with Redis or database
pending_clarifications: Dict[str, Dict] = {}


class ClarificationRequest(BaseModel):
    """Request model for clarification"""
    question: str
    preset_answers: List[str]
    timeout_seconds: Optional[int] = 120


class ClarificationResponse(BaseModel):
    """Response model for clarification"""
    clarification_id: str
    answer: str
    is_preset: bool
    preset_index: Optional[int] = None


class ClarificationStatus(BaseModel):
    """Status model for clarification"""
    clarification_id: str
    status: str  # "pending", "answered", "timeout", "error"
    question: str
    preset_answers: List[str]
    answer: Optional[str] = None
    created_at: str


@router.post("/request", response_model=Dict[str, str])
async def create_clarification_request(request: ClarificationRequest):
    """
    Create a new clarification request.
    Returns the clarification_id that can be used to check status and submit answers.
    """
    clarification_id = str(uuid.uuid4())
    
    pending_clarifications[clarification_id] = {
        "question": request.question,
        "preset_answers": request.preset_answers,
        "status": "pending",
        "answer": None,
        "created_at": asyncio.get_event_loop().time(),
        "timeout_seconds": request.timeout_seconds,
        "future": asyncio.Future()
    }
    
    logger.info(f"Created clarification request {clarification_id}: {request.question}")
    
    return {"clarification_id": clarification_id}


@router.get("/status/{clarification_id}", response_model=ClarificationStatus)
async def get_clarification_status(clarification_id: str):
    """Get the status of a clarification request"""
    
    if clarification_id not in pending_clarifications:
        raise HTTPException(status_code=404, detail="Clarification request not found")
    
    clarification = pending_clarifications[clarification_id]
    
    # Check for timeout
    current_time = asyncio.get_event_loop().time()
    elapsed = current_time - clarification["created_at"]
    
    if elapsed > clarification["timeout_seconds"] and clarification["status"] == "pending":
        clarification["status"] = "timeout"
        if not clarification["future"].done():
            clarification["future"].set_exception(TimeoutError("Clarification request timed out"))
    
    return ClarificationStatus(
        clarification_id=clarification_id,
        status=clarification["status"],
        question=clarification["question"],
        preset_answers=clarification["preset_answers"],
        answer=clarification["answer"]
    )


@router.post("/answer/{clarification_id}")
async def submit_clarification_answer(clarification_id: str, response: ClarificationResponse):
    """Submit an answer to a clarification request"""
    
    if clarification_id not in pending_clarifications:
        raise HTTPException(status_code=404, detail="Clarification request not found")
    
    clarification = pending_clarifications[clarification_id]
    
    if clarification["status"] != "pending":
        raise HTTPException(status_code=400, detail="Clarification request is no longer pending")
    
    # Update the clarification
    clarification["answer"] = response.answer
    clarification["status"] = "answered"
    
    # Resolve the future to unblock the waiting agent
    if not clarification["future"].done():
        clarification["future"].set_result(response.answer)
    
    logger.info(f"Answered clarification {clarification_id}: {response.answer}")
    
    return {"message": "Answer submitted successfully"}


@router.get("/pending")
async def get_pending_clarifications():
    """Get all pending clarification requests"""
    pending = []
    current_time = asyncio.get_event_loop().time()
    
    for clar_id, clarification in pending_clarifications.items():
        if clarification["status"] == "pending":
            # Check timeout
            elapsed = current_time - clarification["created_at"]
            if elapsed > clarification["timeout_seconds"]:
                clarification["status"] = "timeout"
                if not clarification["future"].done():
                    clarification["future"].set_exception(TimeoutError("Clarification request timed out"))
            else:
                pending.append({
                    "clarification_id": clar_id,
                    "question": clarification["question"],
                    "preset_answers": clarification["preset_answers"],
                    "elapsed_seconds": elapsed
                })
    
    # Only log when there are pending clarifications to reduce log spam
    if pending:
        logger.info(f"Found {len(pending)} pending clarification(s)")
    
    return {"pending_clarifications": pending}


@router.delete("/cleanup")
async def cleanup_old_clarifications():
    """Clean up old completed/timeout clarifications"""
    current_time = asyncio.get_event_loop().time()
    to_remove = []
    
    for clar_id, clarification in pending_clarifications.items():
        elapsed = current_time - clarification["created_at"]
        # Remove clarifications older than 10 minutes
        if elapsed > 600:
            to_remove.append(clar_id)
    
    for clar_id in to_remove:
        del pending_clarifications[clar_id]
    
    return {"removed_count": len(to_remove)}


async def wait_for_clarification_answer(clarification_id: str, timeout_seconds: int = 120) -> str:
    """
    Wait for a clarification answer. This is called by the agent.
    Returns the user's answer or raises TimeoutError.
    """
    if clarification_id not in pending_clarifications:
        raise ValueError("Clarification request not found")
    
    clarification = pending_clarifications[clarification_id]
    
    try:
        # Wait for the future to be resolved (when user submits answer)
        answer = await asyncio.wait_for(clarification["future"], timeout=timeout_seconds)
        return answer
    except asyncio.TimeoutError:
        clarification["status"] = "timeout"
        raise TimeoutError("Clarification request timed out")


def create_web_interaction_callback():
    """
    Create a callback function for WebInteraction that uses the clarification API.
    This function will be used by the agent to request clarifications.
    """
    
    async def async_clarification_callback(question: str, preset_answers: List[str]) -> str:
        """Async callback that creates a clarification request and waits for answer"""
        
        # Create clarification request
        clarification_id = str(uuid.uuid4())
        
        pending_clarifications[clarification_id] = {
            "question": question,
            "preset_answers": preset_answers,
            "status": "pending", 
            "answer": None,
            "created_at": asyncio.get_event_loop().time(),
            "timeout_seconds": 120,
            "future": asyncio.Future()
        }
        
        logger.info(f"Agent requested clarification {clarification_id}: {question}")
        
        # Wait for answer
        try:
            answer = await wait_for_clarification_answer(clarification_id, 120)
            return answer
        except TimeoutError:
            logger.warning(f"Clarification {clarification_id} timed out")
            # Return a default answer on timeout
            return "I don't know"
    
    return async_clarification_callback