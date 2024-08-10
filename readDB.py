import sqlite3
conn = sqlite3.connect('faces.db')
c = conn.cursor()
c.execute("SELECT id, encoding FROM unique_faces")
rows = c.fetchall()
print(rows)