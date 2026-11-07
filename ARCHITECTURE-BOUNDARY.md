# HashBurst AI integration boundary

This repository supplies specialist AI/model capabilities. For mining, it may provide reinforcement-learning, prediction, reward and parameter-selection engines to `hashburst/mining-accelerator`.

It does not own HVM consensus, APoW validity/reward rules, node lifecycle or host actuation. Mining decisions are bounded by policy from `hashburst/node`; mining telemetry and mining-specific optimization belong to `hashburst/mining-accelerator`.

Models must return auditable proposals with model/policy version and input provenance. They must not claim measured energy or verified carbon assets from CPU-time proxies.
