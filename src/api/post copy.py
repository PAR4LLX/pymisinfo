import json
import uuid
from datetime import datetime
from enum import Enum


class MisinfoState(Enum):
    TRUE = 0
    FAKE = 1
    NOT_CHECKED = 2


class PostModel:
    def __init__(self, id, message, username, misinfo_state, submitted_date):
        self.id = id
        self.message = message
        self.username = username
        self.misinfo_state = misinfo_state
        self.submitted_date = submitted_date


class MisinformationReport:
    """
    Represents the result of the AI analysis for a single post.
    Includes model confidence and submission date for downstream systems.
    """
    def __init__(self, post_id, misinfo_state, confidence: float, date_submitted):
        self.post_id = post_id
        self.misinfo_state = misinfo_state
        self.confidence = confidence
        self.date_submitted = date_submitted

    def to_dict(self):
        # Encode enum state as integer
        if self.misinfo_state == MisinfoState.FAKE:
            state = 0
        elif self.misinfo_state == MisinfoState.TRUE:
            state = 1
        else:
            state = 2

        return {
            "post_id": str(self.post_id),
            "misinfo_state": state,
            "confidence": round(float(self.confidence), 4),
            "date_submitted": self.date_submitted.isoformat(),
        }


def post_from_json(json_str):
    """Deserialize a JSON post message from the queue into a PostModel."""
    data = json.loads(json_str)

    id = uuid.UUID(data["id"])
    message = data["message"]
    username = data.get("username", "unknown")
    submitted_date = datetime.fromisoformat(data["date"])

    # Gracefully handle missing or invalid misinfo_state
    misinfo_state_int = int(data.get("misinfo_state", 2))  # default 2 = NOT_CHECKED

    if misinfo_state_int == 0:
        misinfo_state = MisinfoState.FAKE
    elif misinfo_state_int == 1:
        misinfo_state = MisinfoState.TRUE
    else:
        misinfo_state = MisinfoState.NOT_CHECKED

    return PostModel(id, message, username, misinfo_state, submitted_date)

