---
layout: docs
title: "AWS Cost Optimization"
hide_title: true
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "server"
---

<div class="hero-section" style="background: linear-gradient(135deg, #ff9900 0%, #ffb84d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">AWS Cost Optimization</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Understand what drives your bill, then put automated guardrails — budgets, anomaly detection, lifecycle policies, and Spot — in place to keep it predictable.</p>
</div>

The cloud's pay-as-you-go model is powerful, but without proper management, costs can spiral. The key is understanding how pricing works and implementing automated controls from the start. This page walks through where cloud spend comes from and gives you ready-to-adapt Terraform and boto3 building blocks for keeping it under control.

---

## Cost Optimization: Spending Smart in the Cloud

### The Cost Evolution Pattern

Most teams follow this cost optimization journey:

1. **Shock Phase**: First AWS bill surprises everyone
2. **Panic Cuts**: Turning off resources randomly
3. **Understanding**: Learning what actually drives costs
4. **Optimization**: Right-sizing and automated management
5. **Mastery**: Costs become predictable and optimized

### Understanding Your Bill

AWS costs break down into three main categories:

#### Compute Costs
- **On-Demand**: Like hotel rooms - flexible but expensive
- **Reserved Instances**: Like apartment leases - cheaper with commitment
- **Spot Instances**: Like last-minute deals - up to 90% off but can be interrupted
- **Savings Plans**: Flexible commitment across instance types

**Real example**: A startup's API servers cost $5,000/month on-demand. After analyzing usage patterns, they buy Reserved Instances for baseline capacity and use Spot for batch processing, reducing costs to $2,000/month.

#### Storage Costs
- **S3 Storage Classes**: Match storage to access patterns
  - Standard: Frequently accessed data
  - Infrequent Access: 50% cheaper for archived data
  - Glacier: 90% cheaper for long-term archives
- **Lifecycle Policies**: Automatically move data to cheaper storage

**Real example**: A photo sharing app automatically moves photos older than 30 days to Infrequent Access, and after 1 year to Glacier. Storage costs drop 70% with no user impact.

#### Data Transfer Costs
- **Within Region**: Free between services
- **Cross-Region**: Charged per GB
- **Internet Egress**: Most expensive

### Advanced Cost Management Tools

```hcl
# cost-optimization.tf - Cost management and optimization

# Cost anomaly detection
resource "aws_ce_anomaly_monitor" "main" {
  name              = "${var.environment}-cost-anomaly-monitor"
  monitor_type      = "DIMENSIONAL"
  monitor_dimension = "SERVICE"
}

resource "aws_ce_anomaly_subscription" "main" {
  name      = "${var.environment}-cost-anomaly-subscription"
  threshold = 100.0  # USD
  frequency = "DAILY"

  monitor_arn_list = [
    aws_ce_anomaly_monitor.main.arn
  ]

  subscriber {
    type    = "EMAIL"
    address = var.cost_alert_email
  }

  subscriber {
    type    = "SNS"
    address = aws_sns_topic.cost_alerts.arn
  }
}

# Budget alerts
resource "aws_budgets_budget" "monthly" {
  name              = "${var.environment}-monthly-budget"
  budget_type       = "COST"
  limit_amount      = var.monthly_budget_limit
  limit_unit        = "USD"
  time_unit         = "MONTHLY"
  time_period_start = "2024-01-01_00:00"

  cost_types {
    include_credit             = false
    include_discount           = true
    include_other_subscription = true
    include_recurring          = true
    include_refund             = false
    include_subscription       = true
    include_support            = true
    include_tax                = true
    include_upfront            = true
    use_blended                = false
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.cost_alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.cost_alerts.arn]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.cost_alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.cost_alerts.arn]
  }
}

# Service-specific budgets
resource "aws_budgets_budget" "service_budgets" {
  for_each = var.service_budgets

  name              = "${var.environment}-${each.key}-budget"
  budget_type       = "COST"
  limit_amount      = each.value.limit
  limit_unit        = "USD"
  time_unit         = "MONTHLY"
  time_period_start = "2024-01-01_00:00"

  cost_filter {
    name = "Service"
    values = [each.key]
  }

  cost_types {
    include_credit             = false
    include_discount           = true
    include_other_subscription = true
    include_recurring          = true
    include_refund             = false
    include_subscription       = true
    include_support            = false
    include_tax                = true
    include_upfront            = true
    use_blended                = false
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 90
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.cost_alert_email]
  }
}

# Compute Optimizer enrollment
resource "aws_organizations_policy" "compute_optimizer" {
  name        = "ComputeOptimizerEnrollment"
  description = "Enable Compute Optimizer for all accounts"
  type        = "SERVICE_CONTROL_POLICY"

  content = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "compute-optimizer:*"
        Resource = "*"
      }
    ]
  })
}

# Lambda for cost optimization recommendations
resource "aws_lambda_function" "cost_optimizer" {
  filename         = "cost_optimizer.zip"
  function_name    = "${var.environment}-cost-optimizer"
  role            = aws_iam_role.cost_optimizer.arn
  handler         = "index.handler"
  runtime         = "python3.12"
  timeout         = 900
  memory_size     = 3008

  environment {
    variables = {
      SNS_TOPIC_ARN    = aws_sns_topic.cost_recommendations.arn
      S3_BUCKET        = aws_s3_bucket.cost_reports.id
      ENVIRONMENT      = var.environment
    }
  }

  layers = [
    "arn:aws:lambda:${var.aws_region}:336392948345:layer:AWSSDKPandas-Python312:1"
  ]
}

# EventBridge rule for weekly cost analysis
resource "aws_cloudwatch_event_rule" "cost_analysis" {
  name                = "${var.environment}-weekly-cost-analysis"
  description         = "Trigger weekly cost analysis"
  schedule_expression = "cron(0 9 ? * MON *)"
}

resource "aws_cloudwatch_event_target" "cost_optimizer" {
  rule      = aws_cloudwatch_event_rule.cost_analysis.name
  target_id = "CostOptimizer"
  arn       = aws_lambda_function.cost_optimizer.arn
}

# Cost and Usage Report
resource "aws_s3_bucket" "cost_reports" {
  bucket = "${var.environment}-cost-reports-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_policy" "cost_reports" {
  bucket = aws_s3_bucket.cost_reports.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "billingreports.amazonaws.com"
        }
        Action = [
          "s3:GetBucketAcl",
          "s3:GetBucketPolicy"
        ]
        Resource = aws_s3_bucket.cost_reports.arn
      },
      {
        Effect = "Allow"
        Principal = {
          Service = "billingreports.amazonaws.com"
        }
        Action = "s3:PutObject"
        Resource = "${aws_s3_bucket.cost_reports.arn}/*"
      }
    ]
  })
}

resource "aws_cur_report_definition" "main" {
  report_name                = "${var.environment}-cost-usage-report"
  time_unit                  = "DAILY"
  format                     = "Parquet"
  compression                = "Parquet"
  additional_schema_elements = ["RESOURCES"]
  s3_bucket                  = aws_s3_bucket.cost_reports.id
  s3_prefix                  = "cur"
  s3_region                  = var.aws_region
  additional_artifacts       = ["QUICKSIGHT"]
  report_versioning          = "OVERWRITE_REPORT"
}

# Reserved Instance utilization alerts
resource "aws_cloudwatch_metric_alarm" "ri_utilization" {
  alarm_name          = "${var.environment}-low-ri-utilization"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "ReservedInstanceUtilization"
  namespace           = "AWS/CE"
  period              = "86400"  # 24 hours
  statistic           = "Average"
  threshold           = "75"
  alarm_description   = "Reserved Instance utilization below 75%"
  alarm_actions       = [aws_sns_topic.cost_alerts.arn]

  dimensions = {
    Currency = "USD"
  }
}

# Savings Plans utilization alerts
resource "aws_cloudwatch_metric_alarm" "sp_utilization" {
  alarm_name          = "${var.environment}-low-sp-utilization"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "SavingsPlansUtilization"
  namespace           = "AWS/CE"
  period              = "86400"
  statistic           = "Average"
  threshold           = "90"
  alarm_description   = "Savings Plans utilization below 90%"
  alarm_actions       = [aws_sns_topic.cost_alerts.arn]
}

# Cost allocation tags
resource "aws_organizations_policy" "tagging" {
  name        = "MandatoryTaggingPolicy"
  description = "Enforce cost allocation tags"
  type        = "TAG_POLICY"

  content = jsonencode({
    tags = {
      Environment = {
        tag_key = {
          "@@assign" = "Environment"
        }
        tag_value = {
          "@@assign" = ["Production", "Staging", "Development"]
        }
        enforced_for = {
          "@@assign" = ["ec2:instance", "s3:bucket", "rds:db"]
        }
      }
      CostCenter = {
        tag_key = {
          "@@assign" = "CostCenter"
        }
        enforced_for = {
          "@@assign" = ["ec2:*", "s3:*", "rds:*"]
        }
      }
      Project = {
        tag_key = {
          "@@assign" = "Project"
        }
        enforced_for = {
          "@@assign" = ["ec2:*", "s3:*", "rds:*"]
        }
      }
    }
  })
}

# Attach tagging policy to organization
resource "aws_organizations_policy_attachment" "tagging" {
  policy_id = aws_organizations_policy.tagging.id
  target_id = aws_organizations_organization.main.roots[0].id
}

# Instance Scheduler for non-production environments
module "instance_scheduler" {
  source  = "aws-ia/instance-scheduler/aws"
  version = "2.0.0"

  scheduler_frequency = "5"

  schedules = [
    {
      name        = "business-hours"
      description = "Run instances during business hours only"
      timezone    = "America/New_York"

      periods = [
        {
          name        = "weekdays"
          description = "Monday to Friday"
          begintime   = "08:00"
          endtime     = "18:00"
          weekdays    = "mon-fri"
        }
      ]
    }
  ]

  tag_name = "Schedule"
}

# Spot Instance configuration
resource "aws_launch_template" "spot" {
  name_prefix = "${var.environment}-spot-"

  instance_market_options {
    market_type = "spot"

    spot_options {
      max_price                      = "0.5"  # 50% of on-demand price
      spot_instance_type             = "persistent"
      instance_interruption_behavior = "stop"
    }
  }

  tag_specifications {
    resource_type = "instance"

    tags = {
      Environment = var.environment
      InstanceType = "spot"
    }
  }
}

# S3 lifecycle policies for cost optimization
resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "transition-old-logs"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 180
      storage_class = "DEEP_ARCHIVE"
    }

    expiration {
      days = 365
    }
  }

  rule {
    id     = "delete-incomplete-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Athena for cost analysis
resource "aws_athena_database" "cost_analysis" {
  name   = "${var.environment}_cost_analysis"
  bucket = aws_s3_bucket.cost_reports.id
}

resource "aws_athena_workgroup" "cost_analysis" {
  name = "${var.environment}-cost-analysis"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://${aws_s3_bucket.cost_reports.id}/athena-results/"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }
}

# QuickSight for cost visualization
resource "aws_quicksight_data_source" "cost_data" {
  data_source_id = "${var.environment}-cost-data"
  name           = "Cost and Usage Report"

  parameters {
    athena {
      work_group = aws_athena_workgroup.cost_analysis.name
    }
  }

  type = "ATHENA"
}
```

Cost categories and Spot Fleet management can be automated in Python with boto3:

```python
class CostCategoryManager:
    def __init__(self):
        self.ce_client = boto3.client('ce')

    def create_cost_categories(self, tag_schema):
        for key in tag_schema.keys():
            self.ce_client.create_cost_category_definition(
                Name=f'CostCategory-{key}',
                Rules=[
                    {
                        'Value': value,
                        'Rule': {
                            'Tags': {
                                'Key': key,
                                'Values': [value]
                            }
                        }
                    } for value in tag_schema[key]
                ]
            )

# Spot Instance management
class SpotInstanceManager:
    def __init__(self):
        self.ec2 = boto3.client('ec2')

    def create_spot_fleet(self,
                         target_capacity: int,
                         instance_types: List[str],
                         max_price: str,
                         subnets: List[str]) -> str:
        """Create diversified Spot Fleet"""

        # Build launch specifications for each instance type
        launch_specs = []

        for instance_type in instance_types:
            for subnet in subnets:
                launch_specs.append({
                    'InstanceType': instance_type,
                    'ImageId': 'ami-12345678',  # Your AMI
                    'KeyName': 'your-key-pair',
                    'SecurityGroups': [{'GroupId': 'sg-12345678'}],
                    'SubnetId': subnet,
                    'IamInstanceProfile': {
                        'Arn': 'arn:aws:iam::account:instance-profile/role'
                    },
                    'TagSpecifications': [
                        {
                            'ResourceType': 'instance',
                            'Tags': [
                                {'Key': 'Name', 'Value': 'SpotFleet-Instance'},
                                {'Key': 'Type', 'Value': 'Spot'}
                            ]
                        }
                    ]
                })

        response = self.ec2.request_spot_fleet(
            SpotFleetRequestConfig={
                'AllocationStrategy': 'diversified',
                'TargetCapacity': target_capacity,
                'SpotPrice': max_price,
                'IamFleetRole': 'arn:aws:iam::account:role/aws-ec2-spot-fleet-role',
                'LaunchSpecifications': launch_specs,
                'TerminateInstancesWithExpiration': True,
                'Type': 'maintain',
                'ReplaceUnhealthyInstances': True,
                'InstanceInterruptionBehavior': 'terminate',
                'TagSpecifications': [
                    {
                        'ResourceType': 'spot-fleet-request',
                        'Tags': [
                            {'Key': 'Name', 'Value': 'MySpotFleet'}
                        ]
                    }
                ]
            }
        )

        return response['SpotFleetRequestId']
```

---

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Cost Is a Design Constraint</h4>
    <p>Right-size, use the appropriate pricing model (Spot, Savings Plans, Reserved), and set billing alerts early. Optimization is continuous, not a one-time cleanup.</p>
  </div>
  <div class="takeaway-card">
    <h4>Automate the Guardrails</h4>
    <p>Budgets, cost anomaly detection, and lifecycle policies catch overspend without anyone watching a dashboard. Codify them so they ship with every environment.</p>
  </div>
  <div class="takeaway-card">
    <h4>Tag for Attribution</h4>
    <p>Enforced cost-allocation tags (Environment, CostCenter, Project) turn one big bill into per-team, per-project numbers you can actually act on.</p>
  </div>
</div>

---

## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Infrastructure as Code](iac.html) - Codify budgets and Spot launch templates
- [Compute Services](compute.html) - Spot, Reserved, and Savings Plans pricing models
- [Storage Services](storage.html) - S3 storage classes and lifecycle policies
- [Monitoring & Messaging](monitoring.html) - CloudWatch alarms behind cost alerts
