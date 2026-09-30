"""Single-process Socket.IO server; state is held in memory."""
bind = "0.0.0.0:5000"
worker_class = "gthread"
workers = 1
threads = 100
timeout = 120
keepalive = 5
max_requests = 0
accesslog = "-"
errorlog = "-"
loglevel = "info"
preload_app = False
