---
layout: docs
title: AWS Security & Identity
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

Security in AWS isn't a feature you add later - it's woven into every decision from day one. This guide covers IAM, Security Hub, KMS, WAF, and security best practices.

## Security: Your First and Constant Priority

Security in AWS isn't a feature you add later - it's woven into every decision from day one. The good news? AWS provides powerful tools that make security easier than traditional on-premise setups.

### The Security Mindset Evolution

Your security journey typically progresses through these stages:

1. **Basic Protection**: Strong passwords, MFA, and basic IAM policies
2. **Defense in Depth**: Network isolation, encryption, and logging
3. **Automated Compliance**: Continuous monitoring and automated remediation
4. **Zero Trust Architecture**: Assume breach, verify everything

### Core Security Principles That Save You Later

#### Principle of Least Privilege
Give users and services only the permissions they need, nothing more. It's tempting to grant broad permissions for convenience, but this creates massive risk.

**Example progression**:
- Bad: Give developers AdministratorAccess
- Better: Create a PowerUserAccess role without IAM permissions
- Best: Custom policies granting exactly what each team needs

#### Encryption Everywhere
AWS makes encryption easy - use it for everything:
- **At Rest**: S3, EBS, RDS all support transparent encryption
- **In Transit**: TLS/SSL for all communications
- **Key Management**: AWS KMS handles the complexity of key rotation

**Real-world scenario**: A healthcare startup encrypts patient data by default. When they undergo HIPAA compliance audit, encryption is already in place, saving months of remediation work.

### Security Hub: Your Compliance Command Center

Security Hub continuously monitors your AWS environment against industry standards (CIS, PCI-DSS, HIPAA). Instead of manual security reviews, you get real-time compliance scores.

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
  runtime         = "python3.9"
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
  runtime         = "python3.9"
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


## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Compute Services](compute.html) - EC2 and Lambda security configurations
- [Networking](../networking.html) - VPC security and WAF integration
- [Infrastructure & Operations](infrastructure.html) - Security automation and compliance monitoring
- [Cybersecurity Guide](../cybersecurity.html) - General security concepts
