"""Test file for vibelint LLM analysis."""


# Poor architectural example with multiple issues
class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.user_data = []
        self.config = {"db_host": "localhost"}

    def connect(self):
        # Bad: mixing UI concerns with database logic
        print("Connecting to database...")
        self.connection = "fake_connection"

    def get_user(self, id):
        # Bad: no error handling, poor separation of concerns
        for user in self.user_data:
            if user["id"] == id:
                return user
        return None

    def save_user_to_file(self, user, filename):
        # Bad: violates single responsibility principle
        with open(filename, "w") as f:
            f.write(str(user))
        print(f"User saved to {filename}")

    def validate_email(self, email):
        # Bad: business logic mixed with data access
        if "@" not in email:
            print("Invalid email!")
            return False
        return True


# Poor naming and structure
def process_stuff(data):
    result = []
    for item in data:
        if item is not None:
            result.append(item * 2)
    return result


# Global variables (bad practice)
GLOBAL_CACHE = {}
current_user = None
