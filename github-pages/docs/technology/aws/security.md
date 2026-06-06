---
layout: docs
title: AWS Security & Identity
permalink: /docs/technology/aws/security.html
hide_title: true
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "shield-alt"
---

<div class="hero-section" style="background: linear-gradient(135deg, #ff9900 0%, #ffb84d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">AWS Security & Identity</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Comprehensive guide to IAM, Security Hub, KMS, WAF, and security best practices for protecting your AWS infrastructure.</p>
</div>

<div class="intro-card">
  <p class="lead-text">Security in AWS isn't a feature you add later - it's woven into every decision from day one. This guide covers identity (IAM), data protection (KMS), threat detection (Security Hub, GuardDuty), and edge protection (WAF), building from foundational principles up to automated, continuous compliance.</p>
</div>

## The Problem Security Solves

In an on-premises data center, a locked door and a firewall went a long way: an attacker had to physically reach the building or breach the network perimeter. In the cloud, every resource has a public API endpoint, and a single leaked access key or an over-permissive policy can expose your entire account from anywhere on the internet. Security in AWS is therefore less about a perimeter and more about *who can do what to which resource* — and proving it.

The good news: AWS gives you fine-grained controls for exactly this, and most of them cost nothing to turn on. The hard part is knowing which control addresses which risk. This guide is organized around four questions:

| Question | Risk it addresses | Primary services |
|----------|-------------------|------------------|
| **Who can act, and on what?** | Stolen credentials, privilege escalation | IAM, IAM Identity Center, Organizations |
| **Is the data readable if stolen?** | Data exfiltration, lost disks | KMS, encryption defaults |
| **Are we being attacked right now?** | Active intrusion, misconfiguration | GuardDuty, Security Hub, Config |
| **Can attacker traffic even reach us?** | DDoS, injection, scraping | WAF, Shield, Security Groups |

### Where the Line Is Drawn: Shared Responsibility

AWS secures the cloud (physical data centers, hypervisors, the network backbone). You secure what is *in* the cloud (identities, configuration, data, application code). Almost every breach you read about lives on the customer side of that line — a public S3 bucket, a hardcoded key, an `0.0.0.0/0` rule. Internalizing this division tells you where to spend your effort.

### The Security Maturity Path

Most teams progress through these stages. You do not need to reach the end on day one, but you should know what each stage protects against:

1. **Basic protection**: MFA on the root account, individual IAM users/roles, no long-lived keys in code.
2. **Defense in depth**: Private subnets, encryption at rest and in transit, CloudTrail logging on.
3. **Automated compliance**: Config rules and Security Hub flag drift; GuardDuty watches for active threats.
4. **Zero trust**: Assume breach. Verify every request, scope every credential, segment every network.

---

## IAM: Who Can Do What

IAM (Identity and Access Management) is the foundation everything else rests on. Get IAM wrong and no amount of encryption or threat detection will save you. Get it right and most attacks have nowhere to go.

### The Four IAM Building Blocks

| Concept | What it is | When to use it |
|---------|-----------|----------------|
| **User** | A long-lived identity for a human or legacy app | Rare today — prefer Identity Center for humans |
| **Group** | A collection of users sharing permissions | Apply policies to many users at once (e.g. `Developers`) |
| **Role** | A temporary identity that anything can assume | The default for applications, EC2, Lambda, cross-account access |
| **Policy** | A JSON document granting or denying actions | Attached to users, groups, or roles to define permissions |

**The single most important rule:** prefer **roles** over **users with access keys**. A role hands out short-lived, automatically-rotated credentials; a user's access key sits in a config file or environment variable indefinitely, waiting to leak. EC2 instances, Lambda functions, and CI/CD pipelines should all assume roles, never carry static keys.

### Anatomy of an IAM Policy

Every permission decision comes down to a policy document. Read this one as "allow reading objects from a specific bucket, but only over TLS":

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "ReadAppBucketOverTLS",
    "Effect": "Allow",
    "Action": ["s3:GetObject"],
    "Resource": "arn:aws:s3:::my-app-bucket/*",
    "Condition": {
      "Bool": { "aws:SecureTransport": "true" }
    }
  }]
}
```

- **Effect**: `Allow` or `Deny`. An explicit `Deny` always wins over any `Allow`.
- **Action**: The API calls being permitted (`service:Operation`). Resist `s3:*`.
- **Resource**: The exact ARNs the actions apply to. Resist `"Resource": "*"`.
- **Condition**: Extra constraints — source IP, MFA present, encryption in transit, etc.

### Least Privilege in Practice

Least privilege means granting only the permissions actually needed, then widening only when something breaks. The progression below shows the journey teams typically take:

| Stage | What it looks like | Why it is risky / better |
|-------|--------------------|--------------------------|
| **Bad** | Every developer has `AdministratorAccess` | One leaked key compromises everything |
| **Better** | A shared `PowerUserAccess` role (no IAM rights) | Limits self-escalation, but still broad |
| **Best** | Per-team policies scoped to specific services and resources | Blast radius of any one credential stays small |

<div class="tip-card">
  <h4>Tighten policies with real usage data</h4>
  <p>Do not guess at permissions. IAM Access Analyzer can generate a least-privilege policy from the actual CloudTrail history of what a role used. Start broad in dev, then let the data tell you what to remove before production.</p>
</div>

### Common IAM Pitfalls

<div class="notice--warning">
  <h4>Common Pitfalls</h4>
  <ul>
    <li><strong>Using the root account for daily work:</strong> The root user can do anything, including closing the account. Lock it behind MFA, create an admin role, and never use root again except for the handful of tasks that require it.</li>
    <li><strong>Long-lived access keys in code:</strong> Keys committed to git or baked into AMIs are the #1 source of breaches. Use roles; if you must use keys, rotate them and scan repos with tools like git-secrets.</li>
    <li><strong>Wildcards everywhere:</strong> <code>"Action": "*"</code> on <code>"Resource": "*"</code> is administrator access by another name. Scope both.</li>
    <li><strong>Forgetting MFA:</strong> A password alone is one phished email away from compromise. Enforce MFA on all human identities via an IAM policy condition.</li>
  </ul>
</div>

---

## Encryption: Unreadable If Stolen

Encryption is your fallback for when a control fails — if a disk, snapshot, or bucket leaks, encrypted data is just noise without the key. AWS makes this nearly free to enable, so the only wrong choice is leaving it off.

| Layer | What to enable | How |
|-------|----------------|-----|
| **At rest** | Default encryption on S3, EBS, RDS, DynamoDB | Toggle in console/IaC; transparent to your app |
| **In transit** | TLS for every connection | Enforce with the `aws:SecureTransport` condition shown above |
| **Key management** | KMS keys with automatic rotation | Use AWS-managed keys to start; customer-managed keys when you need control over policy and rotation |

**Why KMS instead of managing keys yourself?** KMS handles generation, storage, rotation, and access logging of keys, and integrates directly with S3/EBS/RDS so encryption is transparent. Every use of a key is recorded in CloudTrail, giving you an audit trail of who decrypted what.

**Real-world scenario**: A healthcare startup turns on default encryption and KMS rotation from day one. When the HIPAA audit arrives, encryption-at-rest is already in place across every store — turning what is often months of remediation into a checkbox.

---

## Detection: Are We Being Attacked?

Prevention is never perfect, so you also need to *see* what is happening. Three services cover the spectrum, and they complement rather than replace each other:

| Service | What it answers | What it watches |
|---------|-----------------|-----------------|
| **GuardDuty** | "Is something malicious happening?" | CloudTrail, VPC flow, DNS logs analyzed by ML for threats |
| **Config** | "Did a resource drift from policy?" | Resource configuration changes against rules (e.g. "no public buckets") |
| **Security Hub** | "What is my overall posture?" | Aggregates GuardDuty + Config + standards (CIS, PCI-DSS) into one score |

Think of it as layers: **Config** catches misconfiguration, **GuardDuty** catches active intrusion, and **Security Hub** is the dashboard that rolls both up against a compliance benchmark so you get a single prioritized findings list instead of scattered alarms.

### Security Hub: Your Compliance Command Center

Security Hub continuously evaluates your environment against industry standards (CIS, PCI-DSS, AWS Foundational Security Best Practices) and produces a real-time compliance score, so manual quarterly security reviews become continuous automated ones. The Terraform below shows the core wiring — enabling Security Hub, subscribing to standards, and routing high-severity findings to an automated remediation Lambda. The full configuration (GuardDuty, Macie, Inspector, Config rules, WAF) follows; treat it as a reference you adapt, not a block to copy wholesale.

```hcl
# security-hub.tf - Security Hub configuration and custom checks

# Enable Security Hub
resource "aws_securityhub_account" "main" {
  depends_on = [
    aws_organizations_organization.main
  ]
}

# Enable security standards
resource "aws_securityhub_standards_subscription" "cis" {
  standards_arn = "arn:aws:securityhub:${var.aws_region}::standards/cis-aws-foundations-benchmark/v/1.4.0"
  
  depends_on = [aws_securityhub_account.main]
}

resource "aws_securityhub_standards_subscription" "pci_dss" {
  standards_arn = "arn:aws:securityhub:${var.aws_region}::standards/pci-dss/v/3.2.1"
  
  depends_on = [aws_securityhub_account.main]
}

resource "aws_securityhub_standards_subscription" "aws_foundational" {
  standards_arn = "arn:aws:securityhub:${var.aws_region}::standards/aws-foundational-security-best-practices/v/1.0.0"
  
  depends_on = [aws_securityhub_account.main]
}

# Custom Security Hub action
resource "aws_securityhub_action_target" "remediate" {
  name        = "Remediate"
  identifier  = "Remediate"
  description = "Trigger automated remediation"
  
  depends_on = [aws_securityhub_account.main]
}

# Lambda for custom security checks
resource "aws_lambda_function" "security_checker" {
  filename         = "security_checker.zip"
  function_name    = "${var.environment}-custom-security-checks"
  role            = aws_iam_role.security_checker.arn
  handler         = "index.handler"
  runtime         = "python3.12"
  timeout         = 900
  memory_size     = 3008
  
  environment {
    variables = {
      SECURITY_HUB_PRODUCT_ARN = "arn:aws:securityhub:${var.aws_region}:${data.aws_caller_identity.current.account_id}:product/${data.aws_caller_identity.current.account_id}/default"
      ENVIRONMENT              = var.environment
    }
  }
  
  vpc_config {
    subnet_ids         = var.private_subnet_ids
    security_group_ids = [aws_security_group.lambda.id]
  }
}

# EventBridge rule to trigger security checks
resource "aws_cloudwatch_event_rule" "security_checks" {
  name                = "${var.environment}-security-checks"
  description         = "Trigger custom security checks"
  schedule_expression = "rate(6 hours)"
}

resource "aws_cloudwatch_event_target" "security_checker" {
  rule      = aws_cloudwatch_event_rule.security_checks.name
  target_id = "SecurityChecker"
  arn       = aws_lambda_function.security_checker.arn
}

# Config Rules for compliance
resource "aws_config_config_rule" "encrypted_volumes" {
  name = "${var.environment}-encrypted-volumes"
  
  source {
    owner             = "AWS"
    source_identifier = "ENCRYPTED_VOLUMES"
  }
  
  depends_on = [aws_config_configuration_recorder.main]
}

resource "aws_config_config_rule" "restricted_ssh" {
  name = "${var.environment}-restricted-ssh"
  
  source {
    owner             = "AWS"
    source_identifier = "INCOMING_SSH_DISABLED"
  }
  
  depends_on = [aws_config_configuration_recorder.main]
}

resource "aws_config_config_rule" "s3_bucket_encryption" {
  name = "${var.environment}-s3-bucket-encryption"
  
  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_SERVER_SIDE_ENCRYPTION_ENABLED"
  }
  
  depends_on = [aws_config_configuration_recorder.main]
}

# Custom Config rule with Lambda
resource "aws_config_config_rule" "custom_security_check" {
  name = "${var.environment}-custom-security-check"
  
  source {
    owner             = "LAMBDA"
    source_identifier = aws_lambda_function.config_rule_evaluator.arn
    
    source_detail {
      message_type = "ConfigurationItemChangeNotification"
    }
    
    source_detail {
      message_type = "OversizedConfigurationItemChangeNotification"
    }
  }
  
  depends_on = [aws_config_configuration_recorder.main]
}

# GuardDuty configuration
resource "aws_guardduty_detector" "main" {
  enable                       = true
  finding_publishing_frequency = "FIFTEEN_MINUTES"
  
  datasources {
    s3_logs {
      enable = true
    }
    kubernetes {
      audit_logs {
        enable = true
      }
    }
    malware_protection {
      scan_ec2_instance_with_findings {
        ebs_volumes {
          enable = true
        }
      }
    }
  }
}

# GuardDuty threat intelligence sets
resource "aws_s3_bucket" "threat_intel" {
  bucket = "${var.environment}-threat-intel-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "threat_intel" {
  bucket = aws_s3_bucket.threat_intel.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_object" "threat_list" {
  bucket  = aws_s3_bucket.threat_intel.id
  key     = "threat-lists/malicious-ips.txt"
  content = file("${path.module}/threat-lists/malicious-ips.txt")
  etag    = filemd5("${path.module}/threat-lists/malicious-ips.txt")
}

resource "aws_guardduty_threatintelset" "malicious_ips" {
  activate    = true
  detector_id = aws_guardduty_detector.main.id
  format      = "TXT"
  location    = "s3://${aws_s3_bucket.threat_intel.id}/${aws_s3_object.threat_list.key}"
  name        = "malicious-ips"
  
  depends_on = [aws_s3_object.threat_list]
}

# GuardDuty member accounts
resource "aws_guardduty_member" "member_accounts" {
  for_each = var.member_accounts
  
  account_id                 = each.value.account_id
  detector_id                = aws_guardduty_detector.main.id
  email                      = each.value.email
  invite                     = true
  invitation_message         = "You are invited to join GuardDuty"
  disable_email_notification = false
}

# Inspector v2
resource "aws_inspector2_enabler" "main" {
  account_ids    = [data.aws_caller_identity.current.account_id]
  resource_types = ["EC2", "ECR", "LAMBDA"]
}

# Macie for S3 data protection
resource "aws_macie2_account" "main" {
  finding_publishing_frequency = "FIFTEEN_MINUTES"
  status                       = "ENABLED"
}

resource "aws_macie2_classification_job" "s3_scan" {
  job_type = "ONE_TIME"
  name     = "${var.environment}-s3-sensitive-data-scan"
  
  s3_job_definition {
    bucket_definitions {
      account_id = data.aws_caller_identity.current.account_id
      buckets    = [aws_s3_bucket.data.id]
    }
  }
  
  depends_on = [aws_macie2_account.main]
}

# Access Analyzer
resource "aws_accessanalyzer_analyzer" "main" {
  analyzer_name = "${var.environment}-access-analyzer"
  type          = "ACCOUNT"  # or "ORGANIZATION"
  
  tags = {
    Environment = var.environment
  }
}

# Systems Manager compliance
resource "aws_ssm_association" "patch_baseline" {
  name = "AWS-RunPatchBaseline"
  
  targets {
    key    = "tag:Environment"
    values = [var.environment]
  }
  
  schedule_expression = "cron(0 2 ? * SUN *)"
  
  parameters = {
    Operation    = "Install"
    RebootOption = "RebootIfNeeded"
  }
}

# KMS key policies for security
data "aws_iam_policy_document" "kms_key_policy" {
  statement {
    sid    = "Enable IAM User Permissions"
    effect = "Allow"
    
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
    
    actions   = ["kms:*"]
    resources = ["*"]
  }
  
  statement {
    sid    = "Allow use of the key for encryption"
    effect = "Allow"
    
    principals {
      type        = "Service"
      identifiers = [
        "logs.${var.aws_region}.amazonaws.com",
        "s3.amazonaws.com",
        "rds.amazonaws.com"
      ]
    }
    
    actions = [
      "kms:Decrypt",
      "kms:GenerateDataKey",
      "kms:CreateGrant",
      "kms:DescribeKey"
    ]
    
    resources = ["*"]
    
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values = [
        "s3.${var.aws_region}.amazonaws.com",
        "rds.${var.aws_region}.amazonaws.com"
      ]
    }
  }
}

resource "aws_kms_key" "main" {
  description             = "${var.environment} master key"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.kms_key_policy.json
  
  tags = {
    Environment = var.environment
  }
}

# Security automation with EventBridge and Lambda
resource "aws_cloudwatch_event_rule" "security_findings" {
  name        = "${var.environment}-security-findings"
  description = "Capture Security Hub findings for automated remediation"
  
  event_pattern = jsonencode({
    source      = ["aws.securityhub"]
    detail-type = ["Security Hub Findings - Imported"]
    detail = {
      findings = {
        Severity = {
          Label = ["CRITICAL", "HIGH"]
        }
        Workflow = {
          Status = ["NEW"]
        }
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "remediation" {
  rule      = aws_cloudwatch_event_rule.security_findings.name
  target_id = "RemediationFunction"
  arn       = aws_lambda_function.auto_remediation.arn
}

resource "aws_lambda_function" "auto_remediation" {
  filename         = "auto_remediation.zip"
  function_name    = "${var.environment}-security-auto-remediation"
  role            = aws_iam_role.remediation.arn
  handler         = "index.handler"
  runtime         = "python3.12"
  timeout         = 300
  
  environment {
    variables = {
      ENVIRONMENT = var.environment
    }
  }
}

# WAF rules for application protection
resource "aws_wafv2_web_acl" "main" {
  name  = "${var.environment}-waf-acl"
  scope = "REGIONAL"  # or "CLOUDFRONT"
  
  default_action {
    allow {}
  }
  
  rule {
    name     = "RateLimitRule"
    priority = 1
    
    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }
    
    action {
      block {}
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }
  }
  
  rule {
    name     = "ManagedRuleGroup"
    priority = 2
    
    override_action {
      none {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "ManagedRuleGroup"
      sampled_requests_enabled   = true
    }
  }
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.environment}-waf-acl"
    sampled_requests_enabled   = true
  }
  
  tags = {
    Environment = var.environment
  }
}
```

---

## Edge Protection: Can Attacker Traffic Reach Us?

The final layer filters traffic before it touches your application. The `aws_wafv2_web_acl` above sits in front of CloudFront or an ALB and drops bad requests at the edge:

| Control | Protects against | Where it sits |
|---------|------------------|---------------|
| **WAF** | SQL injection, XSS, bot scraping, request floods | CloudFront / ALB / API Gateway |
| **Shield Standard** | Common network/transport DDoS (free, always on) | Edge, automatic |
| **Shield Advanced** | Large/sophisticated DDoS, with cost protection | Edge, paid |
| **Security Groups** | Unwanted inbound/outbound to a resource | Per ENI (stateful firewall) |

The WAF rules above combine a **rate-based rule** (block any IP exceeding 2,000 requests in the evaluation window — a cheap defense against scraping and brute force) with an **AWS-managed rule group** (`KnownBadInputs`) that catches common exploit payloads without you maintaining signatures. Start with managed rule groups; write custom rules only for application-specific patterns.

## How the Layers Fit Together

Security is not one control but a stack — a request must clear every layer, and a breach of one is contained by the next:

```mermaid
flowchart TB
    Req([Incoming request]) --> WAF["WAF + Shield<br/>filter malicious traffic"]
    WAF --> SG["Security Group<br/>allow only expected ports"]
    SG --> App["Application / EC2 / Lambda"]
    App --> IAM["IAM role<br/>scoped, temporary credentials"]
    IAM --> Data["Encrypted data<br/>KMS keys, TLS in transit"]
    GD["GuardDuty + Config + Security Hub"] -. "watch every layer" .-> WAF
    GD -. .-> App
    GD -. .-> Data
```

If WAF misses something, the security group still limits exposure. If a credential leaks, IAM scoping limits the blast radius. If data is exfiltrated, encryption renders it useless. And throughout, the detection services watch every layer for anomalies.

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Least Privilege Always</h4>
    <p>Grant the minimum permissions each user or service needs. Prefer scoped custom policies and IAM roles over broad managed policies like AdministratorAccess.</p>
  </div>
  <div class="takeaway-card">
    <h4>Encrypt Everywhere</h4>
    <p>Enable encryption at rest (S3, EBS, RDS) and in transit (TLS). Let KMS handle key management and rotation.</p>
  </div>
  <div class="takeaway-card">
    <h4>Shared Responsibility</h4>
    <p>AWS secures the cloud; you secure what's in it — access control, configuration, and data. Know which side of the line each control sits on.</p>
  </div>
  <div class="takeaway-card">
    <h4>Detect and Automate</h4>
    <p>Turn on GuardDuty and Security Hub for continuous threat detection, and automate remediation so misconfigurations are caught fast.</p>
  </div>
</div>

---

## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Compute Services](compute.html) - EC2 and Lambda security configurations
- [Networking](../networking/) - VPC security and WAF integration
- [Infrastructure as Code](iac.html) - Security automation and compliance monitoring
- [Cybersecurity Guide](../cybersecurity/) - General security concepts
