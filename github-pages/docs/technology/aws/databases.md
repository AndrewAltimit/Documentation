---
layout: docs
title: AWS Database Services
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "database"
---

# AWS Database Services

AWS offers managed database services for every use case - from traditional relational databases (RDS) to NoSQL solutions (DynamoDB). This guide covers database selection, configuration, and advanced patterns.

## Database Services Overview

### Amazon RDS - Managed Relational Databases
RDS runs traditional databases (MySQL, PostgreSQL, etc.) but handles the tedious parts - backups, patching, replication. You focus on your schema and queries while AWS keeps the database running smoothly.

**Real-world example**: A SaaS application uses RDS PostgreSQL for customer data. RDS automatically backs up the database nightly, replicates to a standby instance for high availability, and can scale up during busy periods.

### Amazon DynamoDB - NoSQL at Scale
DynamoDB is a NoSQL database designed for applications that need consistent performance at any scale. It can handle millions of requests per second with single-digit millisecond latency.

**Real-world example**: A mobile game uses DynamoDB to store player profiles and game state. Whether 100 or 10 million players are online, DynamoDB maintains consistent performance.

### Amazon Aurora - High-Performance Relational
Aurora is AWS's MySQL and PostgreSQL-compatible database that combines the performance of commercial databases with the cost-effectiveness of open-source. It's up to 5x faster than standard MySQL and 3x faster than standard PostgreSQL.

### Amazon ElastiCache - In-Memory Caching
ElastiCache provides Redis and Memcached for caching frequently accessed data. It dramatically reduces database load and improves response times.

**Use cases**:
- Session storage for web applications
- Caching API responses
- Real-time leaderboards
- Rate limiting

### DynamoDB Advanced Patterns

#### Single Table Design

```hcl
# dynamodb-single-table.tf - Advanced DynamoDB single table design

# Single table for all entities
resource "aws_dynamodb_table" "main" {
  name           = "${var.environment}-main-table"
  billing_mode   = "PAY_PER_REQUEST"  # On-demand billing
  hash_key       = "PK"
  range_key      = "SK"
  
  # Enable streams for event processing
  stream_enabled   = true
  stream_view_type = "NEW_AND_OLD_IMAGES"
  
  # Point-in-time recovery
  point_in_time_recovery {
    enabled = true
  }
  
  # Server-side encryption
  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.dynamodb.arn
  }
  
  # Primary key attributes
  attribute {
    name = "PK"
    type = "S"
  }
  
  attribute {
    name = "SK"
    type = "S"
  }
  
  # GSI1 attributes
  attribute {
    name = "GSI1PK"
    type = "S"
  }
  
  attribute {
    name = "GSI1SK"
    type = "S"
  }
  
  # GSI2 attributes
  attribute {
    name = "GSI2PK"
    type = "S"
  }
  
  attribute {
    name = "GSI2SK"
    type = "S"
  }
  
  # GSI3 attributes for time-based queries
  attribute {
    name = "GSI3PK"
    type = "S"
  }
  
  attribute {
    name = "CreatedAt"
    type = "S"
  }
  
  # Global Secondary Index 1 - Entity lookups
  global_secondary_index {
    name            = "GSI1"
    hash_key        = "GSI1PK"
    range_key       = "GSI1SK"
    projection_type = "ALL"
  }
  
  # Global Secondary Index 2 - Date-based queries
  global_secondary_index {
    name            = "GSI2"
    hash_key        = "GSI2PK"
    range_key       = "GSI2SK"
    projection_type = "ALL"
  }
  
  # Global Secondary Index 3 - Time-series data
  global_secondary_index {
    name            = "GSI3"
    hash_key        = "GSI3PK"
    range_key       = "CreatedAt"
    projection_type = "INCLUDE"
    non_key_attributes = ["EntityType", "Status", "UpdatedAt"]
  }
  
  # TTL for temporary data
  ttl {
    attribute_name = "ExpiresAt"
    enabled        = true
  }
  
  tags = {
    Environment = var.environment
    Purpose     = "Single table design"
  }
}

# DynamoDB autoscaling for provisioned mode (if needed)
resource "aws_appautoscaling_target" "dynamodb_table_read" {
  count              = var.enable_autoscaling ? 1 : 0
  max_capacity       = 40000
  min_capacity       = 5
  resource_id        = "table/${aws_dynamodb_table.main.name}"
  scalable_dimension = "dynamodb:table:ReadCapacityUnits"
  service_namespace  = "dynamodb"
}

resource "aws_appautoscaling_policy" "dynamodb_table_read" {
  count              = var.enable_autoscaling ? 1 : 0
  name               = "DynamoDBReadCapacityUtilization:${aws_appautoscaling_target.dynamodb_table_read[0].resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_read[0].resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_read[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_read[0].service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBReadCapacityUtilization"
    }
    target_value = 70
  }
}

# Lambda function for DynamoDB streams processing
resource "aws_lambda_function" "stream_processor" {
  filename         = "stream_processor.zip"
  function_name    = "${var.environment}-dynamodb-stream-processor"
  role            = aws_iam_role.stream_processor.arn
  handler         = "index.handler"
  runtime         = "python3.9"
  timeout         = 60
  memory_size     = 512
  
  environment {
    variables = {
      ENVIRONMENT    = var.environment
      EVENT_BUS_NAME = aws_cloudwatch_event_bus.main.name
    }
  }
  
  tracing_config {
    mode = "Active"
  }
}

# Event source mapping for DynamoDB streams
resource "aws_lambda_event_source_mapping" "dynamodb_stream" {
  event_source_arn  = aws_dynamodb_table.main.stream_arn
  function_name     = aws_lambda_function.stream_processor.arn
  starting_position = "LATEST"
  
  maximum_batching_window_in_seconds = 10
  parallelization_factor             = 10
  maximum_retry_attempts             = 3
  maximum_record_age_in_seconds      = 3600
  
  # Error handling
  destination_config {
    on_failure {
      destination_arn = aws_sqs_queue.dlq.arn
    }
  }
  
  # Filter criteria
  filter_criteria {
    filter {
      pattern = jsonencode({
        eventName = ["INSERT", "MODIFY"]
        dynamodb = {
          NewImage = {
            EntityType = {
              S = ["Order"]
            }
          }
        }
      })
    }
  }
}

# DynamoDB global tables for multi-region
resource "aws_dynamodb_global_table" "main" {
  count = var.enable_global_tables ? 1 : 0
  
  name = aws_dynamodb_table.main.name
  
  dynamic "replica" {
    for_each = var.global_table_regions
    content {
      region_name = replica.value
      
      # KMS key for each region
      kms_key_arn = data.aws_kms_key.regional[replica.value].arn
    }
  }
}

# DynamoDB Accelerator (DAX) cluster
resource "aws_dax_cluster" "main" {
  count = var.enable_dax ? 1 : 0
  
  cluster_name       = "${var.environment}-dax-cluster"
  iam_role_arn       = aws_iam_role.dax.arn
  node_type          = "dax.r4.large"
  replication_factor = 3
  
  # Encryption
  server_side_encryption {
    enabled = true
  }
  
  # Parameter group
  parameter_group_name = aws_dax_parameter_group.main.name
  
  # Subnet group
  subnet_group_name = aws_dax_subnet_group.main.name
  
  # Security
  security_group_ids = [aws_security_group.dax.id]
  
  # Maintenance window
  maintenance_window = "sun:05:00-sun:06:00"
  
  # Notifications
  notification_topic_arn = aws_sns_topic.dax_notifications.arn
  
  tags = {
    Environment = var.environment
  }
}

# DAX parameter group
resource "aws_dax_parameter_group" "main" {
  count = var.enable_dax ? 1 : 0
  name  = "${var.environment}-dax-params"
  
  parameters {
    name  = "query-ttl-millis"
    value = "600000"  # 10 minutes
  }
  
  parameters {
    name  = "record-ttl-millis"
    value = "300000"  # 5 minutes
  }
}

# Contributor Insights for monitoring
resource "aws_dynamodb_contributor_insights" "main" {
  table_name = aws_dynamodb_table.main.name
}

# CloudWatch alarms for DynamoDB
resource "aws_cloudwatch_metric_alarm" "user_errors" {
  alarm_name          = "${var.environment}-dynamodb-user-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "UserErrors"
  namespace           = "AWS/DynamoDB"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "This metric monitors DynamoDB user errors"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    TableName = aws_dynamodb_table.main.name
  }
}

resource "aws_cloudwatch_metric_alarm" "throttled_requests" {
  alarm_name          = "${var.environment}-dynamodb-throttled-requests"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "UserErrors"
  namespace           = "AWS/DynamoDB"
  period              = "300"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "This metric monitors DynamoDB throttled requests"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    TableName = aws_dynamodb_table.main.name
  }
  
  metric_query {
    id          = "throttled"
    return_data = true
    
    metric {
      metric_name = "UserErrors"
      namespace   = "AWS/DynamoDB"
      period      = 300
      stat        = "Sum"
      
      dimensions = {
        TableName = aws_dynamodb_table.main.name
      }
    }
  }
}
```

## Database Selection Guide

| Use Case | Recommended Service | Why |
|----------|-------------------|-----|
| Traditional applications | RDS (MySQL/PostgreSQL) | Familiar SQL, ACID compliance |
| High-scale web apps | Aurora | Better performance, auto-scaling |
| Real-time, high-throughput | DynamoDB | Millisecond latency at any scale |
| Session storage | ElastiCache Redis | In-memory speed, persistence |
| Time-series data | Timestream | Optimized for temporal queries |
| Graph relationships | Neptune | Native graph traversal |
| Document storage | DocumentDB | MongoDB-compatible |

## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Compute Services](compute.html) - Lambda with DynamoDB patterns
- [Storage Services](storage.html) - S3 for database backups
- [Security](security.html) - Database encryption and access control
- [Infrastructure & Operations](infrastructure.html) - Database IaC and monitoring
- [Database Design Guide](../database-design.html) - General database concepts
