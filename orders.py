"""Demo order system (SQLite). Sample data only: Amazon has no public API for real customer orders."""
import os, random, sqlite3, time, datetime as dt

DB_FILE = "orders.db"
RETURN_DAYS = 30
# (status, ordered days ago, shipped days ago, delivered days ago, ETA in N days from today)
PLANS = [("Processing", 0, None, None, 4), ("Shipped", 2, 1, None, 3), ("Out for delivery", 4, 3, None, 0),
         ("Delivered", 8, 7, 5, None), ("Delivered", 50, 49, 45, None)]

def _conn():
    c = sqlite3.connect(DB_FILE); c.row_factory = sqlite3.Row
    return c

def _day(n):
    return None if n is None else (dt.date.today() - dt.timedelta(days=n)).isoformat()

def init_db():
    with _conn() as c:
        c.execute("""create table if not exists orders (order_id text primary key, user text, asin text, title text,
                     price real, qty int, status text, order_date text, ship_date text, delivered_date text,
                     eta text, carrier text, tracking text, return_id text)""")
        c.execute("create table if not exists tickets (ticket_id integer primary key autoincrement, user text, reason text, created text)")

def seed_user(user, get_products):
    """Give an account 5 sample orders built from real catalogue products (does nothing if it already has orders)."""
    with _conn() as c:
        if c.execute("select 1 from orders where user=?", (user,)).fetchone(): return
        n = c.execute("select coalesce(max(cast(substr(order_id,4) as int)), 1000) from orders").fetchone()[0]
        rng = random.Random(user)
        for (status, o_ago, s_ago, d_ago, eta_in), p in zip(PLANS, rng.sample(get_products(), len(PLANS))):
            n += 1
            c.execute("insert into orders values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (f"ORD{n}", user, p["asin"], p["title"], float(p["price"]), 1, status, _day(o_ago), _day(s_ago), _day(d_ago),
                       None if eta_in is None else _day(-eta_in), "UPS" if n % 2 else "Amazon Logistics",
                       f"TBA{rng.randint(10**9, 10**10 - 1)}" if s_ago is not None else None, None))

def reset(user, get_products):
    """Restore one account's sample orders (accounts and chat history are untouched)."""
    with _conn() as c: c.execute("delete from orders where user=?", (user,))
    seed_user(user, get_products)

def list_orders(user):
    with _conn() as c:
        return [dict(r) for r in c.execute("select * from orders where user=? order by order_date desc, order_id", (user,))]

def get_order(user, order_id):
    oid = str(order_id).strip().upper().replace(" ", "")
    with _conn() as c:
        r = c.execute("select * from orders where order_id=? and user=?", (oid, user)).fetchone()
    return dict(r) if r else None

def track(o):
    ev = [f"{o['order_date']}: Order placed"]
    if o["ship_date"]: ev.append(f"{o['ship_date']}: Shipped with {o['carrier']} (tracking {o['tracking']})")
    if o["status"] == "Out for delivery": ev.append(f"{dt.date.today().isoformat()}: Out for delivery, arriving today")
    if o["delivered_date"]: ev.append(f"{o['delivered_date']}: Delivered")
    if o["status"] == "Cancelled": ev.append("Order cancelled")
    if o["return_id"]: ev.append(f"Return {o['return_id']} requested")
    return {"order_id": o["order_id"], "product": o["title"], "status": o["status"], "estimated_delivery": o["eta"], "events": ev}

def return_check(o):
    """-> (eligible, explanation)"""
    if o["return_id"]: return False, f"A return is already in progress (return ID {o['return_id']})."
    if o["status"] == "Cancelled": return False, "This order was cancelled, so there is nothing to return."
    if not o["delivered_date"]: return False, "This order has not been delivered yet. A return can be started after delivery."
    days = (dt.date.today() - dt.date.fromisoformat(o["delivered_date"])).days
    if days > RETURN_DAYS: return False, f"It was delivered {days} days ago, so the {RETURN_DAYS}-day return window has ended."
    return True, f"Eligible: delivered {days} days ago, return window open until {(dt.date.fromisoformat(o['delivered_date']) + dt.timedelta(days=RETURN_DAYS)).isoformat()}."

def cancel_check(o):
    s = o["status"]
    if s == "Processing": return True, "The order has not shipped yet, so it can be cancelled."
    if s == "Cancelled": return False, "This order is already cancelled."
    if s in ("Shipped", "Out for delivery"): return False, "It has already shipped and can no longer be cancelled. You can refuse delivery or return it after it arrives."
    return False, "It has already been delivered. Use a return instead."

def perform_return(user, order_id, reason):
    o = get_order(user, order_id)
    ok, why = return_check(o) if o else (False, "Order not found.")
    if not ok: return {"ok": False, "message": why}
    rid = f"RMA{int(time.time()) % 100000:05d}"
    with _conn() as c:
        c.execute("update orders set status='Return requested', return_id=? where order_id=?", (rid, o["order_id"]))
    return {"ok": True, "return_id": rid, "message": f"Return requested for {o['title']} (order {o['order_id']}). Return ID: {rid}. "
            "Send the item back using the return label; the refund is issued after the return is received and approved."}

def perform_cancel(user, order_id):
    o = get_order(user, order_id)
    ok, why = cancel_check(o) if o else (False, "Order not found.")
    if not ok: return {"ok": False, "message": why}
    with _conn() as c:
        c.execute("update orders set status='Cancelled' where order_id=?", (o["order_id"],))
    return {"ok": True, "message": f"Order {o['order_id']} ({o['title']}) has been cancelled."}

def create_ticket(user, reason):
    with _conn() as c:
        cur = c.execute("insert into tickets (user, reason, created) values (?,?,?)", (user, reason, dt.datetime.now().isoformat(timespec="seconds")))
    return f"TKT{cur.lastrowid:04d}"