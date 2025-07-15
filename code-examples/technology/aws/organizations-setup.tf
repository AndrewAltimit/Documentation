# AWS Organizations configuration
# This demonstrates a complete multi-account organization setup with security controls

# Enable AWS Organizations
resource "aws_organizations_organization" "main" {
  aws_service_access_principals = [
    "cloudtrail.amazonaws.com",
    "config.amazonaws.com",
    "sso.amazonaws.com",
    "controltower.amazonaws.com"
  ]

  enabled_policy_types = [
    "SERVICE_CONTROL_POLICY",
    "TAG_POLICY",
    "BACKUP_POLICY"
  ]

  feature_set = "ALL"
}

# Create organizational units
resource "aws_organizations_organizational_unit" "security" {
  name      = "Security"
  parent_id = aws_organizations_organization.main.roots[0].id
}

resource "aws_organizations_organizational_unit" "production" {
  name      = "Production"
  parent_id = aws_organizations_organization.main.roots[0].id
}

resource "aws_organizations_organizational_unit" "development" {
  name      = "Development"
  parent_id = aws_organizations_organization.main.roots[0].id
}

# Create member accounts
resource "aws_organizations_account" "log_archive" {
  name      = "Log Archive"
  email     = "log-archive@company.com"
  parent_id = aws_organizations_organizational_unit.security.id
  
  # Automatically assume role in the new account
  role_name = "OrganizationAccountAccessRole"

  # Use lifecycle to prevent accidental deletion
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_organizations_account" "audit" {
  name      = "Audit"
  email     = "audit@company.com"
  parent_id = aws_organizations_organizational_unit.security.id
  role_name = "OrganizationAccountAccessRole"

  lifecycle {
    prevent_destroy = true
  }
}

# Service Control Policy
resource "aws_organizations_policy" "require_encryption" {
  name        = "RequireEncryption"
  description = "Require encryption for all services"
  type        = "SERVICE_CONTROL_POLICY"

  content = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Deny"
        Action = [
          "ec2:RunInstances"
        ]
        Resource = "*"
        Condition = {
          Bool = {
            "ec2:Encrypted" = "false"
          }
        }
      },
      {
        Effect = "Deny"
        Action = [
          "s3:PutObject"
        ]
        Resource = "*"
        Condition = {
          StringNotEquals = {
            "s3:x-amz-server-side-encryption" = "AES256"
          }
        }
      }
    ]
  })
}

# Attach SCP to OUs
resource "aws_organizations_policy_attachment" "production_scp" {
  policy_id = aws_organizations_policy.require_encryption.id
  target_id = aws_organizations_organizational_unit.production.id
}

resource "aws_organizations_policy_attachment" "development_scp" {
  policy_id = aws_organizations_policy.require_encryption.id
  target_id = aws_organizations_organizational_unit.development.id
}

# Cross-account role for management
data "aws_iam_policy_document" "cross_account_assume_role" {
  statement {
    effect = "Allow"
    
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
    
    actions = ["sts:AssumeRole"]
    
    condition {
      test     = "StringEquals"
      variable = "sts:ExternalId"
      values   = [var.external_id]
    }
  }
}

resource "aws_iam_role" "cross_account_admin" {
  provider = aws.member_account
  
  name               = "CrossAccountAdminRole"
  assume_role_policy = data.aws_iam_policy_document.cross_account_assume_role.json
  
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AdministratorAccess"
  ]
}

# Output organization details
output "organization_id" {
  value       = aws_organizations_organization.main.id
  description = "The ID of the organization"
}

output "organization_arn" {
  value       = aws_organizations_organization.main.arn
  description = "The ARN of the organization"
}

output "account_ids" {
  value = {
    log_archive = aws_organizations_account.log_archive.id
    audit       = aws_organizations_account.audit.id
  }
  description = "Map of account names to IDs"
}