import json
import random
import uuid
from datetime import datetime, timedelta

PAGE_PATHS = ["/home", "/product", "/checkout"]
EVENTS = ["page_view", "click", "scroll", "mousemove", "add_to_cart"]
PURCHASE_EVENT = "purchase"


def generate_event(user_id, session_id, event_time, event_type):
    return {
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": event_time.isoformat(),
        "event_type": event_type,
        "page": random.choice(["/product", "/checkout"]) if event_type == PURCHASE_EVENT else random.choice(PAGE_PATHS),
        "scroll_depth": random.randint(0, 100),
        "x": random.randint(0, 1920),
        "y": random.randint(0, 1080),
    }


def generate_session_events(churn_session=False):
    event_count = random.randint(4, 12)
    events = []
    for _ in range(event_count):
        if random.random() < (0.15 if churn_session else 0.35):
            events.append(PURCHASE_EVENT)
        else:
            events.append(random.choice(EVENTS))
    if not churn_session and PURCHASE_EVENT not in events:
        events[-1] = PURCHASE_EVENT
    return events


def create_session(user_id, session_start, churn_session=False):
    session_id = str(uuid.uuid4())
    events = []
    event_types = generate_session_events(churn_session=churn_session)
    for index, event_type in enumerate(event_types):
        event_time = session_start + timedelta(seconds=index * random.randint(10, 45))
        events.append(generate_event(user_id, session_id, event_time, event_type))
    return events


def generate_user_sessions(user_id, churn_user=False):
    session_data = []
    if churn_user:
        session_count = random.randint(1, 2)
        last_session_offset = random.randint(16, 30)
        last_session_time = datetime.now() - timedelta(days=last_session_offset, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        for i in range(session_count):
            session_time = last_session_time - timedelta(days=i * random.randint(3, 7), hours=random.randint(0, 5))
            session_data.extend(create_session(user_id, session_time, churn_session=True))
    else:
        session_count = random.randint(2, 5)
        base_days_ago = random.randint(0, 8)
        last_session_time = datetime.now() - timedelta(days=base_days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        for i in range(session_count):
            session_time = last_session_time - timedelta(days=i * random.randint(1, 4), hours=random.randint(0, 5))
            session_data.extend(create_session(user_id, session_time, churn_session=False))
    return session_data


data = []
user_count = 120
churn_rate = 0.25

for i in range(1, user_count + 1):
    user_id = f"U{i}"
    churn_user = random.random() < churn_rate
    data.extend(generate_user_sessions(user_id, churn_user=churn_user))

with open("user_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
