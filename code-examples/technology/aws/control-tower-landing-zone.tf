# AWS Control Tower configuration with Terraform
# This demonstrates Control Tower Landing Zone setup with Account Factory for Terraform (AFT)

# Variables for landing zone configuration
variable "landing_zone_config" {
  description = "Landing zone configuration"
  type = object({
    version = string
    organizational_units = list(object({
      name = string
      accounts = list(object({
        name  = string
        email = string
      }))
    }))
  })
  
  default = {
    version = "3.0"
    organizational_units = [
      {
        name = "Security"
        accounts = [
          {
            name  = "Log Archive"
            email = "log-archive@company.com"
          },
          {
            name  = "Audit"
            email = "audit@company.com"
          }
        ]
      },
      {
        name = "Production"
        accounts = [
          {
            name  = "Production-App"
            email = "prod-app@company.com"
          }
        ]
      },
      {
        name = "Development"
        accounts = [
          {
            name  = "Dev-Sandbox"
            email = "dev-sandbox@company.com"
          }
        ]
      }
    ]
  }
}

# Control Tower Landing Zone (using AFT - Account Factory for Terraform)
module "control_tower_account_factory" {
  source = "aws-ia/control_tower_account_factory/aws"
  version = "1.0.0"

  # Core configuration
  ct_management_account_id    = data.aws_caller_identity.current.account_id
  log_archive_account_id      = aws_organizations_account.log_archive.id
  audit_account_id            = aws_organizations_account.audit.id
  aft_management_account_id   = aws_organizations_account.aft_management.id
  ct_home_region              = var.aws_region
  
  # AFT Feature flags
  aft_feature_cloudtrail_data_events      = true
  aft_feature_enterprise_support          = false
  aft_feature_delete_default_vpcs_enabled = true
  
  # Terraform distribution
  terraform_distribution = "tfc"
  terraform_token        = var.terraform_cloud_token
  terraform_org_name     = var.terraform_cloud_org
  
  # VCS configuration
  vcs_provider                                  = "github"
  account_request_repo_name                     = "${var.github_org}/aft-account-request"
  global_customizations_repo_name               = "${var.github_org}/aft-global-customizations"
  account_customizations_repo_name              = "${var.github_org}/aft-account-customizations"
  account_provisioning_customizations_repo_name = "${var.github_org}/aft-account-provisioning-customizations"
}

# Baseline security services for all accounts
module "security_baseline" {
  source = "./modules/security-baseline"
  
  for_each = toset([
    for ou in var.landing_zone_config.organizational_units : 
    for account in ou.accounts : account.name
  ])
  
  providers = {
    aws = aws.member_accounts[each.key]
  }
  
  enable_cloudtrail   = true
  enable_config       = true
  enable_guardduty    = true
  enable_securityhub  = true
  enable_access_analyzer = true
  
  # CloudTrail configuration
  cloudtrail_bucket_name = "${each.key}-cloudtrail-${data.aws_caller_identity.current.account_id}"
  
  # Config configuration
  config_bucket_name = "${each.key}-config-${data.aws_caller_identity.current.account_id}"
  config_recorder_name = "${each.key}-recorder"
  
  # GuardDuty configuration
  guardduty_finding_publishing_frequency = "FIFTEEN_MINUTES"
  
  # SecurityHub standards
  enable_cis_standard         = true
  enable_pci_dss_standard     = true
  enable_aws_foundational_standard = true
}

# IAM password policy for all accounts
resource "aws_iam_account_password_policy" "strict" {
  for_each = toset([
    for ou in var.landing_zone_config.organizational_units : 
    for account in ou.accounts : account.name
  ])
  
  provider = aws.member_accounts[each.key]
  
  minimum_password_length        = 14
  require_lowercase_characters   = true
  require_numbers                = true
  require_uppercase_characters   = true
  require_symbols                = true
  allow_users_to_change_password = true
  max_password_age               = 90
  password_reuse_prevention      = 24
}