queue workers are opt-in, and off by default

Adding "real workers" in LangSmith Deployment (the current name for LangGraph Platform) means enabling a dedicated queue tier — separate pods that only execute runs:

queue:
enabled: true               # OFF by default
number_of_queue_workers: 8

Until you flip queue.enabled, there are no worker pods at all — the API pods run the queue in-process, exactly as [lifespan.py:123](http://lifespan.py:123) does locally. That's the piece worth knowing: the default deployment has no dedicated workers, so "add more workers" isn't a dial you turn up, it's a tier you turn on.

Autoscaling is likewise disabled by default, and the two tiers scale on different signals — API on request volume, queue on pending run count (lg_api_num_pending_runs, the metric from §4):

api.autoscaling.minReplicas: 15      # high-load example from the docs
api.autoscaling.maxReplicas: 25
queue.autoscaling.minReplicas: 10
queue.autoscaling.maxReplicas: 20

The sizing formula confirms last turn's model

The docs' own capacity formula is literally the worker × job structure I d

available_jobs = number_of_queue_workers × N_JOBS_PER_WORKER
number_of_queue_workers = throughput/sec × avg_run_seconds / N_JOBS_PER_WORKER