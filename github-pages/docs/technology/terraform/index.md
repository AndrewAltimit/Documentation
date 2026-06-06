---
layout: docs
title: Terraform
permalink: /docs/technology/terraform/
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #5c4ee5 0%, #844fba 100%); color: white; padding: 1.5rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2rem;">Terraform</h1>
  <p style="margin-top: 0.5rem; opacity: 0.9;">Infrastructure as Code: theory and practice</p>
</div>

Terraform manages infrastructure as code: instead of clicking through cloud consoles or writing fragile scripts, you describe servers, networks, and services in declarative configuration files, and Terraform figures out how to create, update, and destroy them to match.

## Why Learn Terraform?

Consider the following scenario: Your team needs to deploy the same application across development, staging, and production environments. Without Infrastructure as Code, you might spend hours clicking through web consoles, hoping you remember every setting. With Terraform, you write the configuration once and deploy it consistently everywhere.

**Terraform helps you:**

- **Eliminate manual errors** - No more forgotten security groups or misconfigured databases
- **Enable team collaboration** - Infrastructure changes go through code review just like application code
- **Recover quickly from disasters** - Rebuild entire environments from version-controlled configurations
- **Track changes over time** - See exactly what changed, when, and why through git history

---

## Quick Navigation

### [Core Concepts](core-concepts.html)
**Best for:** Getting started with Terraform, understanding the basics, writing your first configuration.

Learn the fundamentals of Terraform from installation to your first deployment. This section takes you from zero knowledge to deploying real infrastructure in about 30 minutes.

- Prerequisites and installation
- Terraform Crash Course (Zero to Hero in 30 minutes)
- HCL language basics and variables
- Providers and resource lifecycles

### [State & Modules](state-modules.html)
**Best for:** Working in teams, managing multiple environments, creating reusable infrastructure components.

State is what makes Terraform powerful - it tracks what exists in the real world and calculates the minimal changes needed. Modules let you package and reuse infrastructure patterns across projects.

- Understanding Terraform state and why it matters
- Local vs. remote state backends
- Workspaces for environment management
- Creating and consuming modules

### [Enterprise Patterns](patterns.html)
**Best for:** Production deployments, large teams, compliance requirements, scaling infrastructure.

Real-world case studies and battle-tested patterns from organizations running Terraform at scale. Learn from the challenges others have solved.

- Multi-region disaster recovery patterns
- Security and compliance automation
- Performance optimization techniques
- Testing infrastructure code

### [Advanced Topics](advanced.html)
**Best for:** Power users, platform teams, complex automation scenarios.

Push Terraform's boundaries with meta-programming, dynamic configuration generation, and the latest features.

- Meta-programming and code generation
- Policy as Code integration
- Troubleshooting guide
- Future directions in IaC

---

## Key Capabilities

- **Smart dependencies** — automatically determines the right order to create resources from their references.
- **Reliable updates** — plans and applies the minimal change set to move from current to desired state.
- **Error prevention** — validates and plans configuration, catching mistakes before they reach production.

---

## Choosing Your Tool: Terraform vs OpenTofu

In 2023, HashiCorp changed Terraform's license from open-source to BSL (Business Source License). The community responded by creating OpenTofu, a fully open-source fork. Here is how they compare:

| Aspect | Terraform | OpenTofu |
|--------|-----------|----------|
| **License** | BSL (Business Source License) | MPL 2.0 (Open Source) |
| **Maintained by** | HashiCorp | Linux Foundation |
| **Command** | `terraform` | `tofu` |
| **Compatibility** | Original tool | Drop-in replacement |
| **Enterprise features** | Terraform Cloud/Enterprise | Community-driven alternatives |

**When to use Terraform:** You need official HashiCorp support, are already invested in Terraform Cloud, or prefer the stability of the original tool.

**When to use OpenTofu:** You prefer open-source licensing, want community governance, or your organization has licensing concerns with BSL.

Both tools use identical HCL configuration syntax, so skills transfer directly between them.

```bash
# Install OpenTofu
curl -fsSL https://get.opentofu.org/install-opentofu.sh | bash
tofu version
```

---

## See Also

- [AWS Cloud Services](../aws/) - Deploy infrastructure on AWS
- [Kubernetes](../kubernetes/) - Container orchestration
- [Docker](../docker/) - Container fundamentals
- [CI/CD](../ci-cd/) - Continuous integration and deployment
