TASKS = {
    "bugfix_sql_join": '''下面这段 Python 代码有一个 bug。请找出并修复，并解释原因。

```python
def get_users_with_orders(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT u.id, u.name, o.total
        FROM users u
        LEFT JOIN orders o ON o.user_id = u.id
        WHERE o.total > 1000
        ORDER BY o.total DESC
    """)
    return cur.fetchall()
```
''',
    "impl_sliding_window": "用 Python 实现一个函数 `max_sum_sliding_window(nums, k)`，返回长度为 k 的连续子数组的最大和。要求 O(n) 时间、O(1) 额外空间（不含返回），并处理 k > len(nums) 的边界。给出代码。",
    "refactor_sqlish_fragile": "下面的代码是对的但很脆弱，请重构为更健壮的版本：使用 dataclass、类型标注、清晰的错误处理，并拆分成合理函数。保持行为不变。\n\n```python\nrules = {}\ndef add(name, cond, action):\n    rules[name] = (cond, action)\ndef run(data):\n    for name, (cond, action) in rules.items():\n        if cond(data):\n            action(data)\n```",
    "logic_augmented_req": "在下面需求中，你只有一次输出代码的机会，必须一次性正确：\n1. 读取 stdin 每行一个整数，直到 EOF\n2. 输出一个 JSON: {\"count\": n, \"sum\": s, \"avg\": round(s/n, 2)}\n3. n=0 时输出 {\"error\": \"empty\"}，仍是合法 JSON\n请给出 Python 代码。",
    "math_taxi_fare": "一个城市出租车计价规则：起步价 13 元含 3 公里；超过 3 公里每公里 2.3 元；夜间（23:00-05:00）加收 20%；等待费每分钟 1 元。实现函数 `fare(distance_km, night, wait_minutes)` 返回应付款（四舍五入到分）。不要假设任何未给参数。",
}

BENCH_PROMPTS = {}
for name, q in TASKS.items():
    BENCH_PROMPTS[name] = q