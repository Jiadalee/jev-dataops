# Security

This is a single-host operator workbench, not a hardened multi-tenant SaaS. Shared-token users can access all data in an instance. Keep it on loopback or place it behind authenticated HTTPS ingress with explicit host, disk and request limits.

Report vulnerabilities through GitHub's private security reporting feature if enabled, or contact the maintainer privately through their GitHub profile. Do not publish secrets or exploitable customer data in an issue.

Provider keys are read only from environment variables. Live selection sends normalized content fields to the chosen provider; automatic privacy redaction is not provided. Uploads, audit files, response caches and checkpoints remain private runtime artifacts and are excluded from source distributions and Git by default.

Model loading disables remote Python code and requires safetensors weights. The operator remains responsible for selecting and licensing a model, updating dependencies and validating generated artifacts before deployment.
