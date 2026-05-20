# services/logging.py
import json
from pathlib import Path


class LoggingService:
    def __init__(self, log_file):
        self.log_file = log_file

    def save(self, entry):
        if not Path(self.log_file).exists():
            with open(self.log_file, "w") as f:
                json.dump([], f)

        with open(self.log_file, "r") as f:
            logs = json.load(f)

        logs.append(entry)

        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=2)