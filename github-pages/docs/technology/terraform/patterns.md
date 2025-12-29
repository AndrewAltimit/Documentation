---
layout: docs
title: "Terraform: Enterprise Patterns"
permalink: /docs/technology/terraform/patterns.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #5c4ee5 0%, #844fba 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Terraform: Enterprise Patterns</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Real-world case studies and battle-tested patterns for scaling Terraform across teams and organizations.</p>
</div>

## Real-World Case Studies

Learning from how other organizations solve infrastructure challenges can save you weeks of trial and error. These case studies showcase common patterns you will encounter as your Terraform usage matures.

---

### Case Study 1: Multi-Region Disaster Recovery

**The Challenge**

A financial services company needed disaster recovery across three AWS regions. If the primary region failed, traffic should automatically route to the standby.

**Why This Pattern Matters**

Multi-region deployments are common for:
- Financial services (regulatory requirements)
- Global applications (latency reduction)
- High-availability systems (business continuity)

**The Solution**

The key insight: use modules to ensure each region is configured identically, and Route53 for automatic failover.

```hcl
# Same module for primary and standby regions
module "primary" {
  source = "./modules/regional-infrastructure"
  region = "us-east-1"
  role   = "primary"
}

module "standby" {
  source = "./modules/regional-infrastructure"
  region = "us-west-2"
  role   = "standby"
}

# Route53 handles automatic failover
resource "aws_route53_health_check" "primary" {
  fqdn              = module.primary.alb_dns_name
  type              = "HTTPS"
  failure_threshold = 3
}
```

**Lessons Learned**:
- Modules ensure consistency across regions
- Test failover regularly (automate it)
- Monitor cross-region replication lag
- Define RTO/RPO upfront - they drive architecture decisions

---

### Case Study 2: Self-Service Developer Platform

**The Challenge**

A tech company with 1000+ developers needed to provide Kubernetes clusters on demand while maintaining security and controlling costs.

**Why This Pattern Matters**

As organizations grow, platform teams cannot manually provision infrastructure for every request. Self-service platforms let developers move fast while maintaining guardrails.

**The Solution**

Use `for_each` to dynamically create clusters from a configuration map, with standardized settings enforced through the module.

```hcl
# Each team gets their own cluster from the same module
module "team_clusters" {
  source   = "./modules/k8s-platform"
  for_each = var.team_configs

  cluster_name = each.key
  node_count   = each.value.node_count

  # Enforce standards through the module
  security_policies = ["restricted"]
  cost_quotas       = each.value.budget
}
```

**Key patterns used:**
- `for_each` for dynamic resource creation
- Modules to enforce standards
- Quotas to control costs
- Built-in monitoring and alerting

**Lessons Learned**:
- Standardization enables self-service without chaos
- Cost controls are essential at scale (hibernation schedules, quotas)
- Build monitoring in from day one, not after problems occur

---

### Case Study 3: Compliance-Driven Healthcare Infrastructure

**The Challenge**

A healthcare provider needed HIPAA-compliant infrastructure with encryption everywhere, audit trails, and guaranteed data retention.

**Why This Pattern Matters**

Regulated industries (healthcare, finance, government) have strict requirements. Building compliance into your Terraform modules means every deployment is compliant by default.

**The Solution**

Create modules that enforce compliance requirements - encryption, logging, and retention policies are not optional.

```hcl
# Module enforces HIPAA requirements by default
module "patient_data" {
  source = "./modules/compliant-storage"

  bucket_name = "patient-records"

  # These are enforced, not optional
  encryption_required   = true
  access_logging        = true
  retention_days        = 2555  # 7 years for HIPAA
  block_public_access   = true
}

# All API calls logged to immutable storage
resource "aws_cloudtrail" "audit" {
  name                       = "hipaa-audit"
  enable_log_file_validation = true
  kms_key_id                 = aws_kms_key.audit.arn
}
```

**Key compliance patterns:**
- Encryption enforced at module level (not optional)
- Immutable audit logs (cannot be deleted)
- Data subnets isolated from internet
- Automated compliance checking

**Lessons Learned**:
- Build compliance into modules, not as an afterthought
- Automate compliance checking (run daily)
- Immutable logs are critical for audits
- "Enforce, don't trust" - make secure defaults mandatory

---

### Case Study 4: Zero-Downtime Microservices Migration

**The Challenge**

Migrate a monolithic application to microservices without any downtime. The monolith handles $10M in daily transactions.

**Why This Pattern Matters**

The "strangler fig" pattern lets you gradually replace a monolith piece by piece, rather than a risky big-bang rewrite.

**The Solution**

Use API Gateway with weighted routing to gradually shift traffic from monolith to microservices.

```hcl
# Start with 100% to monolith, gradually shift
resource "aws_api_gateway_stage" "prod" {
  stage_name = "prod"

  # Initially: 90% monolith, 10% new service
  # Gradually increase microservice percentage
  canary_settings {
    percent_traffic = 10  # To new microservice
  }
}
```

**Migration phases:**
1. Deploy new microservice alongside monolith
2. Route 10% of traffic to microservice (canary)
3. Monitor errors and latency
4. If healthy, increase to 50%, then 100%
5. Repeat for next service

**Lessons Learned**:
- Gradual migration reduces risk dramatically
- Monitoring is critical during transition (set up dashboards first)
- Always have rollback capability
- Service mesh helps manage inter-service communication

---

### Key Takeaways from These Case Studies

| Area | Pattern | Why It Works |
|------|---------|--------------|
| **Modules** | Opinionated with sensible defaults | Consistency without configuration burden |
| **State** | Split by team/service boundary | Reduces blast radius, enables parallelism |
| **Security** | Encrypt by default, audit everything | Compliance becomes automatic |
| **Operations** | Progressive rollouts with monitoring | Catch problems before they affect everyone |

---

## Performance at Scale

As your infrastructure grows beyond a few hundred resources, Terraform operations can slow down noticeably. Understanding why helps you optimize effectively.

### Common Performance Bottlenecks

| Bottleneck | Symptom | Solution |
|------------|---------|----------|
| Large state files | Slow init, plan, apply | Split state by team/service |
| API rate limits | "Rate exceeded" errors | Reduce parallelism, add retries |
| Sequential dependencies | Cannot parallelize | Restructure resources |
| Remote state latency | Slow operations | Use regional backends |

### Practical Optimization Techniques

**1. Split state by domain**

Instead of one massive state file, split by team or service:

```
infrastructure/
  networking/     # VPCs, subnets (platform team)
  compute/        # EC2, EKS (app teams)
  databases/      # RDS, DynamoDB (data team)
```

**2. Use `-parallelism` wisely**

```bash
# Default is 10. Increase for faster applies, decrease if hitting rate limits
terraform apply -parallelism=20
```

**3. Target specific resources**

When debugging or making small changes, target specific resources:

```bash
terraform plan -target=module.database
terraform apply -target=aws_instance.web
```

**4. Reduce refresh time**

Skip refresh when you know nothing changed externally:

```bash
terraform plan -refresh=false
```

---

## Enterprise Patterns

When Terraform grows from dozens to thousands of resources across multiple teams and accounts, new challenges emerge. This section covers patterns that scale.

### Multi-Account Organization

Large organizations typically structure AWS accounts by:
- **Environment** (dev, staging, prod)
- **Team** (platform, app-team-1, data)
- **Purpose** (security, audit, networking)

```hcl
# Define accounts systematically
module "organization" {
  source = "./modules/aws-organization"

  organizational_units = {
    production  = { accounts = ["prod-us", "prod-eu"] }
    development = { accounts = ["dev-team1", "dev-team2"] }
    security    = { accounts = ["audit", "logging"] }
  }
}
```

### Module Governance

At scale, you need governance around modules:

| Practice | Why It Matters |
|----------|----------------|
| Version pinning | Prevents breaking changes |
| Deprecation policy | Give teams time to migrate |
| Testing requirements | Catch issues before deployment |
| Documentation standards | Enable self-service |

### Key Scaling Patterns

**1. Hierarchical State Management**

Split state into layers that different teams own:

```
Layer 1: Account baseline (platform team)
    |
Layer 2: Networking (platform team)
    |
Layer 3: Applications (app teams)
```

Each layer reads outputs from the layer above via `terraform_remote_state`.

**2. GitOps Workflow**

Automate Terraform in CI/CD:
- Run `terraform plan` on every PR
- Require approval for production applies
- Add cost estimation (Infracost) and security scanning (Checkov)
- Post results as PR comments

**3. Policy as Code**

Use tools like OPA (Open Policy Agent) or Sentinel to enforce rules:
- Allowed instance types per environment
- Required tags on all resources
- Cost thresholds that block excessive spending

---

## Security and Compliance

When infrastructure becomes code, security practices from software development apply directly.

### Security Scanning Tools

Run these tools in CI/CD before any apply reaches production:

| Tool | What It Checks | How to Run |
|------|----------------|------------|
| **Checkov** | Misconfigurations, compliance | `checkov -d .` |
| **tfsec** | Security issues | `tfsec .` |
| **Terrascan** | Policy violations | `terrascan scan` |
| **Infracost** | Cost estimates | `infracost breakdown` |

### Secret Management

Never store secrets in Terraform files or state. Instead:

```hcl
# Read secrets from AWS Secrets Manager
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "prod/database/password"
}

resource "aws_db_instance" "main" {
  password = data.aws_secretsmanager_secret_version.db_password.secret_string
}
```

For more advanced scenarios, use HashiCorp Vault for dynamic, short-lived credentials.

---

## Testing Infrastructure Code

Testing catches problems before they affect production. Here is the testing pyramid for Terraform:

### Testing Levels

| Level | What It Tests | Tools |
|-------|---------------|-------|
| **Validation** | Syntax, types | `terraform validate` |
| **Static Analysis** | Security, best practices | Checkov, tfsec |
| **Unit Tests** | Module logic | Terratest, pytest |
| **Integration Tests** | Real infrastructure | Terratest |

### Practical Testing Example

Use Terratest (Go) for integration tests:

```go
func TestWebServerModule(t *testing.T) {
    opts := &terraform.Options{
        TerraformDir: "./modules/webserver",
        Vars: map[string]interface{}{
            "instance_type": "t3.micro",
        },
    }

    defer terraform.Destroy(t, opts)
    terraform.InitAndApply(t, opts)

    // Verify the instance is reachable
    ip := terraform.Output(t, opts, "public_ip")
    http_helper.HttpGetWithRetry(t, "http://"+ip, nil, 200, "OK", 30, 5*time.Second)
}
```

**When to test:**
- Always validate and run static analysis in CI
- Run unit tests on every PR
- Run integration tests nightly or before releases (they are slow and cost money)

