import torch
import uuid

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, session_data):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = session_data
        return session_id

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def delete_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

session_manager = SessionManager()  # Singleton instance