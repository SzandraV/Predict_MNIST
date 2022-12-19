import sqlite3
from datetime import datetime
import pandas as pd

class MLModel_DB():
    """Class definition to ML(MachineLearning) model Testing
    """

    def __init__(self, source: str):
        """This is the initializer of the Class
            Opens the database file or creates if does not exist.

        Args:
            source (str): The source file's url (filename)
        """
        self.source = source
        self.db = sqlite3.connect(source)
        self.cur = self.db.cursor()

    def create_db(self):
        """This method creates the DataBase's structure:
            Table "histories" (PK id, time, model, param, score)
        """
        self.cur.execute('''CREATE TABLE IF NOT EXISTS histories(
                    id integer PRIMARY KEY AUTOINCREMENT NOT NULL,
                    time text NOT NULL,
                    model text NOT NULL,
                    param text NOT NULL,
                    score float)''')
        self.db.commit()

    def create_log(self, model_name: str, param_list: str, score: float) -> bool:
        self.cur = self.db.cursor()
        time = str(datetime.now())
        data = [(time, model_name, param_list, score)]
        self.cur.executemany("""INSERT INTO histories
                            (time, model, param, score)
                            VALUES(?, ?, ?, ?)""", data)
        self.db.commit()
        return True

    def log_query(self) -> pd.DataFrame:
        self.cur = self.db.cursor()
        df_hist = pd.read_sql(f"""SELECT time, model, param, score
                    FROM histories""", self.db)
        return df_hist
