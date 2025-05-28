from .chain import ScriptChain # Adjusted import
from callbacks import LoggingCallback # Adjusted import

# --- ScriptChain Storage ---
# We'll use a dictionary to store separate ScriptChain instances for each session
script_chain_store = {}

# Helper function to get or create a ScriptChain for a session
def get_script_chain(session_id):
    """Get or create a ScriptChain instance for the given session ID."""
    if session_id not in script_chain_store:
        print(f"Creating new ScriptChain for session {session_id}")
        chain = ScriptChain()
        chain.add_callback(LoggingCallback())
        script_chain_store[session_id] = chain
    return script_chain_store[session_id] 