"""Accounts (salted PBKDF2 passwords) and saved chat history. SQLite, same file as the orders."""
import hashlib, hmac, json, re, secrets, sqlite3, time
from orders import _conn

def init():
    with _conn() as c:
        c.execute("create table if not exists users (username text primary key, salt text, pw text, created text)")
        c.execute("create table if not exists messages (id integer primary key autoincrement, username text, chat_id text, ts real, payload text)")
        c.execute("create index if not exists ix_msg on messages(username, chat_id)")

_hash = lambda pw, salt: hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()

def register(user, pw):
    """-> (username, error)"""
    user = user.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,20}", user): return None, "Username: 3-20 letters, numbers or _"
    if len(pw) < 6: return None, "Password must be at least 6 characters"
    salt = secrets.token_hex(16)
    try:
        with _conn() as c: c.execute("insert into users values (?,?,?,?)", (user, salt, _hash(pw, salt), time.ctime()))
    except sqlite3.IntegrityError: return None, "That username is taken"
    return user, None

def login(user, pw):
    user = user.strip().lower()
    with _conn() as c: r = c.execute("select salt, pw from users where username=?", (user,)).fetchone()
    return (user, None) if r and hmac.compare_digest(_hash(pw, r["salt"]), r["pw"]) else (None, "Wrong username or password")

def save(user, chat_id, m):
    with _conn() as c: c.execute("insert into messages (username, chat_id, ts, payload) values (?,?,?,?)", (user, chat_id, time.time(), json.dumps(m, default=str)))

def load(user, chat_id):
    with _conn() as c: return [json.loads(r[0]) for r in c.execute("select payload from messages where username=? and chat_id=? order by id", (user, chat_id))]

def chats(user, limit=15):
    """-> [(chat_id, title)] newest first; title = first question."""
    with _conn() as c:
        ids = [r[0] for r in c.execute("select chat_id from messages where username=? group by chat_id order by max(id) desc limit ?", (user, limit))]
        return [(i, json.loads(c.execute("select payload from messages where username=? and chat_id=? order by id limit 1", (user, i)).fetchone()[0])["text"][:28]) for i in ids]

def delete_chat(user, chat_id):
    with _conn() as c: c.execute("delete from messages where username=? and chat_id=?", (user, chat_id))