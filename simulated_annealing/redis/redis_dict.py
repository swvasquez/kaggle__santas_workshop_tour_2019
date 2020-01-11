import redis


r = redis.Redis(
    host='localhost',
    port=6379,
    charset="utf-8",
    decode_responses=True
)

print(r.dbsize())