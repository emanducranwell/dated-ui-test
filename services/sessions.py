# services/sessions.py
import sqlite3
from datetime import datetime


class SessionsService:
    def __init__(self, db_path):
        self.db_path = db_path

    def record_visit(self, prior_visit=False):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO sessions(time, prior_visit) VALUES(?, ?)",
            (datetime.now(), prior_visit),
        )
        con.commit()
        con.close()  