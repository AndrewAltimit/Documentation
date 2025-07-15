---
layout: single
title: AWS Developer's Guide
---

# AWS Developer's Guide

<html><header><link rel="stylesheet" href="https://andrewaltimit.github.io/Documentation/style.css"></header></html>

Amazon Web Services (AWS) is a comprehensive cloud services platform that offers a wide range of services to help developers build, deploy, and manage applications. AWS provides everything from compute and storage resources to machine learning and analytics services. With over 200 fully featured services, AWS enables organizations to build sophisticated applications with increased flexibility, scalability, and reliability.

## Best Practices

- **Security**: Implement the principle of least privilege with IAM, use encryption, and follow AWS security best practices.
- **Cost Optimization**: Leverage auto-scaling, spot instances, and other cost-saving techniques.
- **Backup and Recovery**: Regularly create and test backups to ensure data durability and recoverability.
- **Monitoring and Logging**: Use Amazon CloudWatch, AWS X-Ray, and other monitoring tools to track application performance and diagnose issues.
- **Performance**: Optimize performance by using caching, Content Delivery Networks (CDNs), and other performance-enhancing techniques.
- **Infrastructure as Code**: Use AWS CloudFormation or Terraform to manage your infrastructure as code and maintain version control.

## Resources and Tools

### Documentation

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [AWS Architecture Center](https://aws.amazon.com/architecture/)
- [AWS Whitepapers](https://aws.amazon.com/whitepapers/)
- [AWS Documentation](https://aws.amazon.com/documentation/)

### Getting Started

- [AWS Free Tier](https://aws.amazon.com/free/): Get started with AWS using the free tier, which includes limited access to many AWS services.
- [AWS Training and Certification](https://aws.amazon.com/training/): Access AWS training resources and certification programs to build and validate your AWS knowledge.
- [AWS Blog](https://aws.amazon.com/blogs/aws/): Stay up to date with AWS news, announcements, and best practices.
- [AWS Marketplace](https://aws.amazon.com/marketplace/): Find and deploy pre-built software solutions on AWS.

### SDKs and Libraries

- [AWS SDKs](https://aws.amazon.com/tools/): Use AWS SDKs to interact with AWS services in your preferred programming language.
- [AWS Amplify](https://aws.amazon.com/amplify/): Utilize the Amplify library to simplify building cloud-powered mobile and web applications.
- [AWS CDK](https://aws.amazon.com/cdk/): Use the Cloud Development Kit (CDK) to define cloud infrastructure using familiar programming languages.

### AWS Partner Network

Leverage the [AWS Partner Network (APN)](https://aws.amazon.com/partners/) to find and collaborate with AWS Consulting and Technology Partners who offer a wide range of solutions and expertise to help you get the most out of AWS.

### Community

- [AWS Developer Forums](https://forums.aws.amazon.com/index.jspa): Engage with the AWS developer community to ask questions and share knowledge.
- [AWS User Groups](https://aws.amazon.com/usergroups/): Connect with other AWS users at local events and meetups.
- [AWS re:Invent](https://reinvent.awsevents.com/): Attend AWS's annual global conference for learning, networking, and discovering new services and features.

## Common Solutions

### Serverless Architecture

- Utilize AWS Lambda for compute and Amazon API Gateway for handling HTTP requests.
- Leverage Amazon S3 for static website hosting and object storage.
- Use Amazon DynamoDB for serverless databases.

### Microservices Architecture

- Implement containerized microservices using Amazon ECS or EKS.
- Use Amazon API Gateway for service-to-service communication and API management.
- Leverage Amazon RDS or DynamoDB for database services.

### Big Data and Analytics

- Ingest and process real-time data using Amazon Kinesis Data Streams and Kinesis Data Analytics.
- Use Amazon EMR for batch processing and Amazon Redshift for data warehousing.
- Visualize and analyze data using Amazon QuickSight.

### Machine Learning Pipeline

- Train, deploy, and manage ML models using Amazon SageMaker.
- Use Amazon S3 for storing training datasets and model artifacts.
- Integrate with other AWS services like Lambda, API Gateway, and Kinesis for real-time processing and predictions.

### High Availability and Disaster Recovery

- Design your architecture for high availability by deploying resources across multiple Availability Zones (AZs).
- Use Amazon RDS Multi-AZ deployments, Amazon EFS, and Amazon S3 for durable and highly available storage.
- Leverage AWS services like Amazon Route 53, Elastic Load Balancing (ELB), and Auto Scaling Groups to ensure fault tolerance and load distribution.

### Web Application Hosting

- Host web applications using Amazon EC2 instances behind an Application Load Balancer (ALB).
- Store static assets in Amazon S3 and use Amazon CloudFront for content delivery.
- Utilize Amazon RDS or DynamoDB for database storage.

### Data Processing and ETL

- Ingest data using Amazon Kinesis Data Streams or Firehose.
- Process and transform data using AWS Glue, AWS Data Pipeline, or AWS Step Functions.
- Store processed data in Amazon S3, Amazon RDS, Amazon Redshift, or Amazon Elasticsearch Service.

### Hybrid Cloud Solutions

- Extend your on-premises data center to AWS using AWS Direct Connect or VPN connections.
- Use AWS Storage Gateway and AWS Outposts for hybrid cloud storage and compute solutions.
- Leverage AWS services like Amazon RDS, Amazon WorkSpaces, and Amazon Connect to extend your on-premises solutions to the cloud.

## List of Services

### Compute

- **Amazon EC2**: Elastic Compute Cloud (EC2) provides scalable virtual servers.
- **AWS Lambda**: Serverless compute service for running code without provisioning servers.
- **Amazon ECS**: Elastic Container Service (ECS) is a container orchestration service.
- **Amazon EKS**: Elastic Kubernetes Service (EKS) is a managed Kubernetes service.

### Storage

- **Amazon S3**: Simple Storage Service (S3) is an object storage service.
- **Amazon EBS**: Elastic Block Store (EBS) provides block-level storage volumes for EC2 instances.
- **Amazon EFS**: Elastic File System (EFS) is a managed file storage service.
- **AWS Storage Gateway**: Hybrid storage service connecting on-premises environments to AWS storage.

### Databases

- **Amazon RDS**: Relational Database Service (RDS) is a managed relational database service.
- **Amazon DynamoDB**: Managed NoSQL database service.
- **Amazon ElastiCache**: In-memory data store and cache service.
- **Amazon Redshift**: Managed data warehouse service.

### Networking

- **Amazon VPC**: Virtual Private Cloud (VPC) provides an isolated virtual network within AWS.
- **Amazon Route 53**: Scalable Domain Name System (DNS) web service.
- **AWS Direct Connect**: Dedicated network connection between your on-premises environment and AWS.

### Security

- **AWS Identity and Access Management (IAM)**: Manage user access and permissions.
- **Amazon Cognito**: User authentication and authorization service.
- **AWS Security Hub**: Centralized security management and monitoring.

### Developer Tools

- **AWS CodeCommit**: Managed source control service.
- **AWS CodeBuild**: Managed build service.
- **AWS CodeDeploy**: Managed deployment service.
- **AWS CodePipeline**: Continuous delivery pipeline service.

### Analytics

- **Amazon Kinesis**: Real-time data streaming and processing service.
- **Amazon EMR**: Managed Hadoop framework.
- **Amazon Elasticsearch Service**: Managed Elasticsearch service.
- **Amazon QuickSight**: Business intelligence and data visualization service.

### Machine Learning

- **Amazon SageMaker**: Managed machine learning platform.
- **Amazon Rekognition**: Image and video analysis service.
- **Amazon Comprehend**: Natural language processing (NLP) service.
- **Amazon Lex**: Conversational interfaces and chatbot service.

### Application Integration

- **Amazon SNS**: Simple Notification Service (SNS) is a publish-subscribe messaging service.
- **Amazon SQS**: Simple Queue Service (SQS) is a fully managed message queuing service.
- **AWS Step Functions**: Coordinate distributed applications and microservices using visual workflows.

### IoT and Edge Computing

- **AWS IoT Core**: Managed cloud platform for IoT devices.
- **AWS Greengrass**: Extend AWS services to edge devices for local processing and data management.
- **Amazon FreeRTOS**: IoT operating system for microcontrollers.

### Mobile and Web Development

- **AWS Amplify**: Development platform for building mobile and web applications with built-in authentication, API, storage, and more.
- **AWS App Runner**: Service for building, deploying, and scaling containerized applications quickly.
- **Amazon AppStream 2.0**: Fully managed application streaming service.

### Management and Monitoring

- **Amazon CloudWatch**: Monitor and manage your AWS resources and applications, and set up alarms for specific events.
- **AWS Trusted Advisor**: Optimize your AWS infrastructure with automated best practice checks for cost, performance, security, and fault tolerance.
- **AWS Organizations**: Centrally manage and govern your AWS environment across multiple accounts.

### Migration and Transfer

- **AWS Database Migration Service**: Migrate databases to AWS with minimal downtime.
- **AWS DataSync**: Transfer data to and from AWS quickly and securely.
- **AWS Snow Family**: Use physical devices to transport large amounts of data to and from AWS.

## Advanced Implementation Patterns

### Multi-Account Architecture

#### AWS Organizations Setup

**AWS Organizations enables centralized management of multiple AWS accounts:**

- **Organizational Units (OUs)**: Hierarchical structure for account grouping
- **Service Control Policies (SCPs)**: Preventive guardrails across accounts
- **Cross-Account Roles**: Secure access management between accounts
- **Automated Account Creation**: Programmatic account provisioning

**Key features implemented:**
- Security, Production, and Development OUs
- Encryption enforcement via SCPs
- Log Archive and Audit accounts
- Cross-account administrative access

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/aws/organizations-setup.tf">organizations-setup.tf</a>
</div>

```hcl
# Example usage:
module "organization" {
  source = "./modules/organization"
  
  external_id = "unique-external-id"
  
  # Reference outputs
  organization_id = module.organization.organization_id
  account_ids    = module.organization.account_ids
}
```

#### Control Tower Landing Zone

**AWS Control Tower provides automated multi-account governance:**

- **Account Factory for Terraform (AFT)**: GitOps-based account provisioning
- **Security Baseline**: Automated security service deployment
- **Guardrails**: Preventive and detective controls
- **Landing Zone**: Well-architected multi-account environment

**Key components implemented:**
- Organizational structure with Security, Production, and Development OUs
- Automated security services (CloudTrail, Config, GuardDuty, SecurityHub)
- Strict IAM password policies across all accounts
- Integration with Terraform Cloud for infrastructure management

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/aws/control-tower-landing-zone.tf">control-tower-landing-zone.tf</a>
</div>

```hcl
# Example usage:
module "landing_zone" {
  source = "./modules/control-tower"
  
  aws_region             = "us-east-1"
  terraform_cloud_token  = var.tfc_token
  terraform_cloud_org    = "myorg"
  github_org            = "mycompany"
}
```

### Event-Driven Architecture

#### EventBridge Pattern

**EventBridge enables decoupled event-driven architectures:**

- **Custom Event Bus**: Domain-specific event routing
- **Event Archive**: Replay capability for debugging and recovery
- **Error Handling**: DLQ and SNS notifications for failures
- **Step Functions Integration**: Complex workflow orchestration

**Key components implemented:**
- Order processing pipeline with Lambda functions
- Event filtering and transformation
- Dead letter queue for resilience
- Step Functions for multi-step workflows
- API Gateway integration for event publishing
- Comprehensive error handling and monitoring

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/aws/eventbridge-pattern.tf">eventbridge-pattern.tf</a>
</div>

```hcl
# Example usage:
module "event_driven_system" {
  source = "./modules/eventbridge"
  
  environment = "production"
  aws_region  = "us-east-1"
  
  # Connect to existing resources
  orders_table_arn = aws_dynamodb_table.orders.arn
  kms_key_id      = aws_kms_key.main.id
}
```

### Advanced Lambda Patterns

#### Lambda with Container Images

**Advanced Lambda patterns with containers and enterprise features:**

- **Container Images**: Package Lambda functions as OCI images up to 10GB
- **Lambda Layers**: Share code and dependencies across functions
- **Extensions**: Integrate monitoring and security tools
- **Auto-scaling**: Provisioned concurrency with automatic scaling

**Key features implemented:**
- ECR repository with lifecycle policies
- Container-based Lambda with 10GB ephemeral storage
- AWS Lambda Powertools and OpenTelemetry integration
- Function URLs with CORS configuration
- Traffic shifting with weighted aliases
- X-Ray tracing and CloudWatch Insights
- Provisioned concurrency with auto-scaling
- Async invocation destinations

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/aws/lambda-container-patterns.tf">lambda-container-patterns.tf</a>
</div>

```hcl
# Example usage:
module "lambda_advanced" {
  source = "./modules/lambda-patterns"
  
  environment        = "production"
  aws_region        = "us-east-1"
  private_subnet_ids = module.vpc.private_subnets
  
  # Container image configuration
  ecr_repository_url = aws_ecr_repository.app.repository_url
  image_tag         = "v1.2.3"
}
```

resource "aws_appautoscaling_policy" "lambda_concurrency" {
  name               = "${var.environment}-lambda-concurrency-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.lambda_concurrency.resource_id
  scalable_dimension = aws_appautoscaling_target.lambda_concurrency.scalable_dimension
  service_namespace  = aws_appautoscaling_target.lambda_concurrency.service_namespace
  
  target_tracking_scaling_policy_configuration {
    target_value = 0.7
    
    predefined_metric_specification {
      predefined_metric_type = "LambdaProvisionedConcurrencyUtilization"
    }
    
    scale_in_cooldown  = 180
    scale_out_cooldown = 0
  }
}
```

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

### API Gateway Advanced Patterns

#### REST API with Request Validation

```hcl
# api-gateway.tf - Advanced API Gateway with request validation

# REST API
resource "aws_api_gateway_rest_api" "main" {
  name        = "${var.environment}-advanced-api"
  description = "Advanced REST API with validation and security"
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
  
  # API Policy for resource limits
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = "*"
        Action = "execute-api:Invoke"
        Resource = "*"
      },
      {
        Effect = "Deny"
        Principal = "*"
        Action = "execute-api:Invoke"
        Resource = "*"
        Condition = {
          IpAddressNotEquals = {
            "aws:SourceIp" = var.allowed_ip_ranges
          }
        }
      }
    ]
  })
}

# Request validator
resource "aws_api_gateway_request_validator" "main" {
  name                        = "${var.environment}-validator"
  rest_api_id                 = aws_api_gateway_rest_api.main.id
  validate_request_body       = true
  validate_request_parameters = true
}

# API Models for request/response validation
resource "aws_api_gateway_model" "user" {
  rest_api_id  = aws_api_gateway_rest_api.main.id
  name         = "User"
  content_type = "application/json"
  
  schema = jsonencode({
    "$schema" = "http://json-schema.org/draft-04/schema#"
    type = "object"
    required = ["email", "name"]
    properties = {
      email = {
        type = "string"
        format = "email"
        pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
      }
      name = {
        type = "string"
        minLength = 1
        maxLength = 100
      }
      age = {
        type = "integer"
        minimum = 0
        maximum = 150
      }
      preferences = {
        type = "object"
        properties = {
          notifications = {
            type = "boolean"
          }
          theme = {
            type = "string"
            enum = ["light", "dark", "auto"]
          }
        }
      }
    }
  })
}

resource "aws_api_gateway_model" "error" {
  rest_api_id  = aws_api_gateway_rest_api.main.id
  name         = "Error"
  content_type = "application/json"
  
  schema = jsonencode({
    "$schema" = "http://json-schema.org/draft-04/schema#"
    type = "object"
    required = ["message"]
    properties = {
      message = {
        type = "string"
      }
      code = {
        type = "string"
      }
      requestId = {
        type = "string"
      }
    }
  })
}

# API Resources
resource "aws_api_gateway_resource" "v1" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "v1"
}

resource "aws_api_gateway_resource" "users" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_resource.v1.id
  path_part   = "users"
}

resource "aws_api_gateway_resource" "user" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_resource.users.id
  path_part   = "{userId}"
}

# OPTIONS method for CORS
resource "aws_api_gateway_method" "users_options" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.users.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "users_options" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.users_options.http_method
  type        = "MOCK"
  
  request_templates = {
    "application/json" = "{\"statusCode\": 200}"
  }
}

resource "aws_api_gateway_method_response" "users_options" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.users_options.http_method
  status_code = "200"
  
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
  
  response_models = {
    "application/json" = "Empty"
  }
}

resource "aws_api_gateway_integration_response" "users_options" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.users_options.http_method
  status_code = aws_api_gateway_method_response.users_options.status_code
  
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS,POST,PUT,DELETE'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
}

# POST method with validation
resource "aws_api_gateway_method" "create_user" {
  rest_api_id          = aws_api_gateway_rest_api.main.id
  resource_id          = aws_api_gateway_resource.users.id
  http_method          = "POST"
  authorization        = "CUSTOM"
  authorizer_id        = aws_api_gateway_authorizer.jwt.id
  request_validator_id = aws_api_gateway_request_validator.main.id
  
  request_models = {
    "application/json" = aws_api_gateway_model.user.name
  }
  
  request_parameters = {
    "method.request.header.X-Correlation-ID" = true
  }
}

# Lambda integration
resource "aws_api_gateway_integration" "create_user" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.create_user.http_method
  
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.create_user.invoke_arn
  
  timeout_milliseconds = 29000
  
  request_templates = {
    "application/json" = jsonencode({
      statusCode = 200
    })
  }
}

# Method responses
resource "aws_api_gateway_method_response" "create_user_200" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.create_user.http_method
  status_code = "200"
  
  response_models = {
    "application/json" = aws_api_gateway_model.user.name
  }
  
  response_parameters = {
    "method.response.header.X-Correlation-ID" = true
  }
}

resource "aws_api_gateway_method_response" "create_user_400" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.create_user.http_method
  status_code = "400"
  
  response_models = {
    "application/json" = aws_api_gateway_model.error.name
  }
}

# Custom authorizer
resource "aws_api_gateway_authorizer" "jwt" {
  name                   = "${var.environment}-jwt-authorizer"
  rest_api_id            = aws_api_gateway_rest_api.main.id
  type                   = "TOKEN"
  authorizer_uri         = aws_lambda_function.authorizer.invoke_arn
  authorizer_credentials = aws_iam_role.api_gateway_authorizer.arn
  identity_source        = "method.request.header.Authorization"
  
  # Cache auth results
  authorizer_result_ttl_in_seconds = 300
}

# Lambda authorizer function
resource "aws_lambda_function" "authorizer" {
  filename         = "authorizer.zip"
  function_name    = "${var.environment}-api-authorizer"
  role            = aws_iam_role.lambda_authorizer.arn
  handler         = "index.handler"
  runtime         = "python3.9"
  timeout         = 5
  memory_size     = 256
  
  environment {
    variables = {
      JWT_SECRET        = aws_secretsmanager_secret_version.jwt_secret.secret_string
      ENVIRONMENT       = var.environment
      USER_POOL_ID      = aws_cognito_user_pool.main.id
      TOKEN_USE         = "access"
    }
  }
}

# API deployment
resource "aws_api_gateway_deployment" "main" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  
  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.users.id,
      aws_api_gateway_method.create_user.id,
      aws_api_gateway_integration.create_user.id,
    ]))
  }
  
  lifecycle {
    create_before_destroy = true
  }
  
  depends_on = [
    aws_api_gateway_method.create_user,
    aws_api_gateway_integration.create_user
  ]
}

# API stages
resource "aws_api_gateway_stage" "prod" {
  deployment_id = aws_api_gateway_deployment.main.id
  rest_api_id   = aws_api_gateway_rest_api.main.id
  stage_name    = "prod"
  
  # Enable logging
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId               = "$context.requestId"
      extendedRequestId       = "$context.extendedRequestId"
      ip                      = "$context.identity.sourceIp"
      caller                  = "$context.identity.caller"
      user                    = "$context.identity.user"
      requestTime             = "$context.requestTime"
      httpMethod              = "$context.httpMethod"
      resourcePath            = "$context.resourcePath"
      status                  = "$context.status"
      protocol                = "$context.protocol"
      responseLength          = "$context.responseLength"
      error                   = "$context.error.message"
      integrationLatency      = "$context.integration.latency"
      integrationStatus       = "$context.integration.status"
      integrationErrorMessage = "$context.integrationErrorMessage"
      authorizerError         = "$context.authorizer.error"
    })
  }
  
  # Caching
  cache_cluster_enabled = true
  cache_cluster_size    = "0.5"
  
  # Throttling
  throttle_burst_limit = 5000
  throttle_rate_limit  = 10000
  
  # X-Ray tracing
  xray_tracing_enabled = true
  
  variables = {
    deployed_at = timestamp()
  }
}

# Method settings for all methods
resource "aws_api_gateway_method_settings" "all" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  stage_name  = aws_api_gateway_stage.prod.stage_name
  method_path = "*/*"
  
  settings = {
    metrics_enabled        = true
    logging_level          = "INFO"
    data_trace_enabled     = true
    throttling_burst_limit = 2000
    throttling_rate_limit  = 1000
    caching_enabled        = false
  }
}

# API key and usage plan
resource "aws_api_gateway_api_key" "main" {
  name = "${var.environment}-api-key"
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_api_gateway_usage_plan" "main" {
  name         = "${var.environment}-usage-plan"
  description  = "Usage plan for API clients"
  
  api_stages {
    api_id = aws_api_gateway_rest_api.main.id
    stage  = aws_api_gateway_stage.prod.stage_name
  }
  
  quota_settings {
    limit  = 10000
    period = "DAY"
  }
  
  throttle_settings {
    rate_limit  = 500
    burst_limit = 1000
  }
}

resource "aws_api_gateway_usage_plan_key" "main" {
  key_id        = aws_api_gateway_api_key.main.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.main.id
}

# Custom domain
resource "aws_api_gateway_domain_name" "main" {
  domain_name              = "api.${var.domain_name}"
  regional_certificate_arn = aws_acm_certificate_validation.api.certificate_arn
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
  
  security_policy = "TLS_1_2"
  
  mutual_tls_authentication {
    truststore_uri = "s3://${aws_s3_bucket.truststore.bucket}/truststore.pem"
  }
}

resource "aws_api_gateway_base_path_mapping" "main" {
  api_id      = aws_api_gateway_rest_api.main.id
  stage_name  = aws_api_gateway_stage.prod.stage_name
  domain_name = aws_api_gateway_domain_name.main.domain_name
  base_path   = "v1"
}

# WAF for API Gateway
resource "aws_wafv2_web_acl_association" "api_gateway" {
  resource_arn = aws_api_gateway_stage.prod.arn
  web_acl_arn  = aws_wafv2_web_acl.api.arn
}
```

### Step Functions Advanced Workflows

```hcl
# step-functions.tf - Advanced Step Functions workflows

# Step Functions state machine for order processing
resource "aws_sfn_state_machine" "order_processing" {
  name     = "${var.environment}-order-processing"
  role_arn = aws_iam_role.step_functions.arn
  
  definition = jsonencode({
    Comment = "Advanced order processing workflow with error handling"
    StartAt = "ValidateOrder"
    States = {
      ValidateOrder = {
        Type = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.validate_order.arn
          "Payload.$" = "$"
        }
        ResultPath = "$.validation"
        Retry = [
          {
            ErrorEquals = ["Lambda.ServiceException", "Lambda.AWSLambdaException"]
            IntervalSeconds = 2
            MaxAttempts = 3
            BackoffRate = 2
          }
        ]
        Catch = [
          {
            ErrorEquals = ["ValidationError"]
            Next = "HandleValidationError"
            ResultPath = "$.error"
          },
          {
            ErrorEquals = ["States.ALL"]
            Next = "HandleGeneralError"
            ResultPath = "$.error"
          }
        ]
        Next = "CheckInventory"
      }
      
      CheckInventory = {
        Type = "Task"
        Resource = "arn:aws:states:::aws-sdk:dynamodb:query"
        Parameters = {
          TableName = aws_dynamodb_table.inventory.name
          KeyConditionExpression = "ProductId = :productId"
          ExpressionAttributeValues = {
            ":productId" = {
              "S.$" = "$.order.productId"
            }
          }
        }
        ResultPath = "$.inventory"
        Next = "EvaluateInventory"
      }
      
      EvaluateInventory = {
        Type = "Choice"
        Choices = [
          {
            Variable = "$.inventory.Items[0].Stock.N"
            NumericGreaterThanEquals = 1
            Next = "ProcessPayment"
          }
        ]
        Default = "InsufficientInventory"
      }
      
      InsufficientInventory = {
        Type = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = aws_sns_topic.inventory_alerts.arn
          Message = {
            "orderId.$" = "$.order.orderId"
            "productId.$" = "$.order.productId"
            "message" = "Insufficient inventory"
          }
        }
        Next = "OrderFailed"
      }
      
      ProcessPayment = {
        Type = "Parallel"
        Branches = [
          {
            StartAt = "ChargePayment"
            States = {
              ChargePayment = {
                Type = "Task"
                Resource = "arn:aws:states:::lambda:invoke.waitForTaskToken"
                Parameters = {
                  FunctionName = aws_lambda_function.process_payment.arn
                  Payload = {
                    "order.$" = "$.order"
                    "taskToken.$" = "$$.Task.Token"
                  }
                }
                TimeoutSeconds = 300
                HeartbeatSeconds = 30
                Retry = [
                  {
                    ErrorEquals = ["PaymentDeclined"]
                    MaxAttempts = 2
                    IntervalSeconds = 5
                  }
                ]
                End = true
              }
            }
          },
          {
            StartAt = "SendConfirmationEmail"
            States = {
              SendConfirmationEmail = {
                Type = "Task"
                Resource = "arn:aws:states:::aws-sdk:ses:sendEmail"
                Parameters = {
                  Destination = {
                    ToAddresses = ["$.order.customerEmail"]
                  }
                  Message = {
                    Body = {
                      Html = {
                        Data = "Order confirmation email body"
                      }
                    }
                    Subject = {
                      Data = "Order Confirmation"
                    }
                  }
                  Source = "noreply@example.com"
                }
                ResultPath = "$.emailResult"
                End = true
              }
            }
          },
          {
            StartAt = "LogAnalytics"
            States = {
              LogAnalytics = {
                Type = "Task"
                Resource = "arn:aws:states:::aws-sdk:firehose:putRecordBatch"
                Parameters = {
                  DeliveryStreamName = aws_kinesis_firehose_delivery_stream.analytics.name
                  Records = [{
                    Data = {
                      "orderId.$" = "$.order.orderId"
                      "amount.$" = "$.order.amount"
                      "timestamp.$" = "$$.State.EnteredTime"
                      "eventType" = "order_placed"
                    }
                  }]
                }
                ResultPath = null
                End = true
              }
            }
          }
        ]
        Next = "UpdateInventory"
        ResultPath = "$.paymentResults"
      }
      
      UpdateInventory = {
        Type = "Map"
        ItemsPath = "$.order.items"
        MaxConcurrency = 5
        Parameters = {
          "item.$" = "$$.Map.Item.Value"
          "orderId.$" = "$.order.orderId"
        }
        Iterator = {
          StartAt = "DecrementStock"
          States = {
            DecrementStock = {
              Type = "Task"
              Resource = "arn:aws:states:::dynamodb:updateItem"
              Parameters = {
                TableName = aws_dynamodb_table.inventory.name
                Key = {
                  ProductId = {
                    "S.$" = "$.item.productId"
                  }
                }
                UpdateExpression = "SET #stock = #stock - :quantity, #updated = :timestamp"
                ExpressionAttributeNames = {
                  "#stock" = "Stock"
                  "#updated" = "LastUpdated"
                }
                ExpressionAttributeValues = {
                  ":quantity" = {
                    "N.$" = "$.item.quantity"
                  }
                  ":timestamp" = {
                    "S.$" = "$$.State.EnteredTime"
                  }
                }
                ConditionExpression = "#stock >= :quantity"
                ReturnValues = "ALL_NEW"
              }
              ResultPath = "$.updateResult"
              End = true
            }
          }
        }
        Next = "CompleteOrder"
        ResultPath = "$.inventoryUpdates"
        Catch = [
          {
            ErrorEquals = ["DynamoDB.ConditionalCheckFailedException"]
            Next = "HandleInventoryError"
          }
        ]
      }
      
      CompleteOrder = {
        Type = "Task"
        Resource = "arn:aws:states:::batch:submitJob.sync"
        Parameters = {
          JobName = "CompleteOrderJob"
          JobQueue = aws_batch_job_queue.main.name
          JobDefinition = aws_batch_job_definition.complete_order.name
          Parameters = {
            "orderId.$" = "$.order.orderId"
          }
        }
        Next = "OrderSuccess"
      }
      
      OrderSuccess = {
        Type = "Succeed"
      }
      
      OrderFailed = {
        Type = "Fail"
        Cause = "Order processing failed"
      }
      
      HandleValidationError = {
        Type = "Task"
        Resource = aws_lambda_function.handle_error.arn
        Parameters = {
          "error.$" = "$.error"
          "order.$" = "$.order"
          "errorType" = "validation"
        }
        Next = "OrderFailed"
      }
      
      HandleInventoryError = {
        Type = "Task"
        Resource = aws_lambda_function.handle_error.arn
        Parameters = {
          "error.$" = "$.error"
          "order.$" = "$.order"
          "errorType" = "inventory"
        }
        Next = "OrderFailed"
      }
      
      HandleGeneralError = {
        Type = "Task"
        Resource = aws_lambda_function.handle_error.arn
        Parameters = {
          "error.$" = "$.error"
          "order.$" = "$.order"
          "errorType" = "general"
        }
        Next = "OrderFailed"
      }
    }
  })
  
  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.step_functions.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }
  
  tracing_configuration {
    enabled = true
  }
  
  tags = {
    Environment = var.environment
    Purpose     = "Order processing workflow"
  }
}

# Express workflow for synchronous execution
resource "aws_sfn_state_machine" "express_workflow" {
  name     = "${var.environment}-express-workflow"
  role_arn = aws_iam_role.step_functions.arn
  type     = "EXPRESS"
  
  definition = jsonencode({
    Comment = "Express workflow for fast synchronous execution"
    StartAt = "TransformData"
    States = {
      TransformData = {
        Type = "Task"
        Resource = "arn:aws:states:::aws-sdk:lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.transform_data.arn
          Payload = {
            "input.$" = "$"
          }
        }
        OutputPath = "$.Payload"
        Next = "EnrichData"
      }
      
      EnrichData = {
        Type = "Task"
        Resource = aws_lambda_function.enrich_data.arn
        TimeoutSeconds = 5
        Next = "ReturnResult"
      }
      
      ReturnResult = {
        Type = "Pass"
        Parameters = {
          "transformedData.$" = "$"
          "processedAt.$" = "$$.State.EnteredTime"
          "executionName.$" = "$$.Execution.Name"
        }
        End = true
      }
    }
  })
}

# CloudWatch Logs for Step Functions
resource "aws_cloudwatch_log_group" "step_functions" {
  name              = "/aws/vendedlogs/states/${var.environment}-order-processing"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.logs.arn
}

# CloudWatch alarms for Step Functions
resource "aws_cloudwatch_metric_alarm" "step_functions_failed" {
  alarm_name          = "${var.environment}-step-functions-executions-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "ExecutionsFailed"
  namespace           = "AWS/States"
  period              = "300"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "This metric monitors failed Step Functions executions"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    StateMachineArn = aws_sfn_state_machine.order_processing.arn
  }
}

# EventBridge rule to trigger Step Functions
resource "aws_cloudwatch_event_rule" "trigger_workflow" {
  name        = "${var.environment}-trigger-order-workflow"
  description = "Trigger order processing workflow"
  
  event_pattern = jsonencode({
    source      = ["order.service"]
    detail-type = ["Order Placed"]
  })
}

resource "aws_cloudwatch_event_target" "step_functions" {
  rule      = aws_cloudwatch_event_rule.trigger_workflow.name
  target_id = "StepFunctionsTarget"
  arn       = aws_sfn_state_machine.order_processing.arn
  role_arn  = aws_iam_role.events_step_functions.arn
  
  input_transformer {
    input_paths = {
      order = "$.detail.order"
    }
    input_template = jsonencode({
      order = "<order>"
    })
  }
}
```

### Container Orchestration with ECS/Fargate

```hcl
# ecs-fargate.tf - Container orchestration with ECS and Fargate

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.environment}-cluster"
  
  configuration {
    execute_command_configuration {
      kms_key_id = aws_kms_key.ecs.id
      logging    = "OVERRIDE"
      
      log_configuration {
        cloud_watch_encryption_enabled = true
        cloud_watch_log_group_name     = aws_cloudwatch_log_group.ecs_exec.name
      }
    }
  }
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Environment = var.environment
  }
}

# Capacity providers for mixing Fargate and Fargate Spot
resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name
  
  capacity_providers = [
    "FARGATE",
    "FARGATE_SPOT"
  ]
  
  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = "FARGATE"
  }
  
  default_capacity_provider_strategy {
    base              = 0
    weight            = 50
    capacity_provider = "FARGATE_SPOT"
  }
}

# Task Definition
resource "aws_ecs_task_definition" "app" {
  family                   = "${var.environment}-app"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn
  
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"  # or "ARM64" for Graviton2
  }
  
  container_definitions = jsonencode([
    {
      name      = "app"
      image     = "${aws_ecr_repository.app.repository_url}:latest"
      essential = true
      
      portMappings = [
        {
          containerPort = 8080
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        }
      ]
      
      secrets = [
        {
          name      = "DB_PASSWORD"
          valueFrom = aws_secretsmanager_secret.db_password.arn
        },
        {
          name      = "API_KEY"
          valueFrom = aws_ssm_parameter.api_key.arn
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
      
      linuxParameters = {
        initProcessEnabled = true
      }
      
      # FireLens logging
      firelensConfiguration = {
        type = "fluentbit"
        options = {
          enable-ecs-log-metadata = "true"
          config-file-type        = "file"
          config-file-value       = "/fluent-bit/configs/parse-json.conf"
        }
      }
      
      # ECS Exec
      linuxParameters = {
        initProcessEnabled = true
      }
      
      # Resource limits
      ulimits = [
        {
          name      = "nofile"
          softLimit = 65536
          hardLimit = 65536
        }
      ]
      
      # Mount points for EFS
      mountPoints = [
        {
          sourceVolume  = "efs-storage"
          containerPath = "/data"
          readOnly      = false
        }
      ]
    },
    {
      name             = "xray-daemon"
      image            = "public.ecr.aws/xray/aws-xray-daemon:latest"
      essential        = false
      memoryReservation = 256
      
      portMappings = [
        {
          containerPort = 2000
          protocol      = "udp"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_xray.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "xray"
        }
      }
    }
  ])
  
  # EFS volume
  volume {
    name = "efs-storage"
    
    efs_volume_configuration {
      file_system_id          = aws_efs_file_system.app.id
      root_directory          = "/"
      transit_encryption      = "ENABLED"
      transit_encryption_port = 2999
      
      authorization_config {
        access_point_id = aws_efs_access_point.app.id
        iam             = "ENABLED"
      }
    }
  }
  
  # Ephemeral storage
  ephemeral_storage {
    size_in_gib = 30
  }
  
  tags = {
    Environment = var.environment
  }
}

# ECS Service
resource "aws_ecs_service" "app" {
  name                               = "${var.environment}-app-service"
  cluster                            = aws_ecs_cluster.main.id
  task_definition                    = aws_ecs_task_definition.app.arn
  desired_count                      = var.app_count
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  health_check_grace_period_seconds  = 60
  launch_type                        = "FARGATE"
  platform_version                   = "LATEST"
  
  deployment_controller {
    type = "ECS"  # or "CODE_DEPLOY" for blue/green
  }
  
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  
  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_service.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 8080
  }
  
  service_registries {
    registry_arn = aws_service_discovery_service.app.arn
  }
  
  # Capacity provider strategy to use Fargate Spot
  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
    base              = 1
  }
  
  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 4
    base              = 0
  }
  
  enable_ecs_managed_tags = true
  propagate_tags          = "SERVICE"
  
  tags = {
    Environment = var.environment
  }
  
  lifecycle {
    ignore_changes = [desired_count]
  }
  
  depends_on = [
    aws_lb_listener.app,
    aws_iam_role_policy_attachment.ecs_task
  ]
}

# Auto Scaling
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# CPU-based auto scaling
resource "aws_appautoscaling_policy" "ecs_cpu" {
  name               = "${var.environment}-ecs-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    
    target_value       = 70.0
    scale_in_cooldown  = 180
    scale_out_cooldown = 60
  }
}

# Memory-based auto scaling
resource "aws_appautoscaling_policy" "ecs_memory" {
  name               = "${var.environment}-ecs-memory-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    
    target_value       = 75.0
    scale_in_cooldown  = 180
    scale_out_cooldown = 60
  }
}

# Custom metric scaling
resource "aws_appautoscaling_policy" "ecs_requests" {
  name               = "${var.environment}-ecs-requests-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    customized_metric_specification {
      metric_name = "RequestsPerTask"
      namespace   = "${var.environment}/Application"
      statistic   = "Average"
      unit        = "Count"
      
      dimensions {
        name  = "ServiceName"
        value = aws_ecs_service.app.name
      }
    }
    
    target_value       = 1000.0
    scale_in_cooldown  = 180
    scale_out_cooldown = 60
  }
}

# Scheduled scaling
resource "aws_appautoscaling_scheduled_action" "scale_up_morning" {
  name               = "${var.environment}-scale-up-morning"
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  schedule           = "cron(0 8 * * ? *)"  # 8 AM UTC daily
  timezone           = "America/New_York"
  
  scalable_target_action {
    min_capacity = 5
    max_capacity = 20
  }
}

# Service Discovery
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "${var.environment}.local"
  description = "Private DNS namespace for service discovery"
  vpc         = var.vpc_id
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_service_discovery_service" "app" {
  name = "app"
  
  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    
    dns_records {
      ttl  = 10
      type = "A"
    }
    
    routing_policy = "MULTIVALUE"
  }
  
  health_check_custom_config {
    failure_threshold = 1
  }
}

# CloudWatch Logs
resource "aws_cloudwatch_log_group" "ecs_app" {
  name              = "/ecs/${var.environment}/app"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.logs.arn
}

resource "aws_cloudwatch_log_group" "ecs_xray" {
  name              = "/ecs/${var.environment}/xray"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "ecs_exec" {
  name              = "/ecs/${var.environment}/exec"
  retention_in_days = 7
  kms_key_id        = aws_kms_key.logs.arn
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "ecs" {
  dashboard_name = "${var.environment}-ecs-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ServiceName", aws_ecs_service.app.name, "ClusterName", aws_ecs_cluster.main.name],
            [".", "MemoryUtilization", ".", ".", ".", "."]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "ECS Service Utilization"
        }
      },
      {
        type   = "metric"
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", aws_lb.main.arn_suffix],
            [".", "RequestCount", ".", ".", { stat = "Sum" }],
            [".", "HTTPCode_Target_4XX_Count", ".", ".", { stat = "Sum" }],
            [".", "HTTPCode_Target_5XX_Count", ".", ".", { stat = "Sum" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "Load Balancer Metrics"
        }
      }
    ]
  })
}
```

## Performance Optimization Strategies

### CloudFront and Edge Computing

```hcl
# cloudfront.tf - CloudFront distribution with Lambda@Edge

# S3 bucket for static content
resource "aws_s3_bucket" "static_content" {
  bucket = "${var.environment}-static-content-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "static_content" {
  bucket = aws_s3_bucket.static_content.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "static_content" {
  bucket = aws_s3_bucket.static_content.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3.id
    }
  }
}

# CloudFront Origin Access Control
resource "aws_cloudfront_origin_access_control" "main" {
  name                              = "${var.environment}-oac"
  description                       = "Origin Access Control for S3"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# Lambda@Edge functions
data "archive_file" "lambda_edge_viewer_request" {
  type        = "zip"
  source_file = "lambda-edge/viewer-request.js"
  output_path = "lambda-edge-viewer-request.zip"
}

resource "aws_lambda_function" "edge_viewer_request" {
  provider         = aws.us_east_1  # Lambda@Edge must be in us-east-1
  filename         = data.archive_file.lambda_edge_viewer_request.output_path
  function_name    = "${var.environment}-edge-viewer-request"
  role            = aws_iam_role.lambda_edge.arn
  handler         = "viewer-request.handler"
  runtime         = "nodejs18.x"
  timeout         = 5
  memory_size     = 128
  publish         = true
  
  source_code_hash = data.archive_file.lambda_edge_viewer_request.output_base64sha256
}

data "archive_file" "lambda_edge_origin_response" {
  type        = "zip"
  source_file = "lambda-edge/origin-response.js"
  output_path = "lambda-edge-origin-response.zip"
}

resource "aws_lambda_function" "edge_origin_response" {
  provider         = aws.us_east_1
  filename         = data.archive_file.lambda_edge_origin_response.output_path
  function_name    = "${var.environment}-edge-origin-response"
  role            = aws_iam_role.lambda_edge.arn
  handler         = "origin-response.handler"
  runtime         = "nodejs18.x"
  timeout         = 5
  memory_size     = 128
  publish         = true
  
  source_code_hash = data.archive_file.lambda_edge_origin_response.output_base64sha256
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "${var.environment} CloudFront distribution"
  default_root_object = "index.html"
  aliases             = [var.domain_name, "www.${var.domain_name}"]
  price_class         = "PriceClass_200"  # US, Canada, Europe, Asia
  
  # S3 origin
  origin {
    domain_name              = aws_s3_bucket.static_content.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.main.id
    origin_id                = "S3-${aws_s3_bucket.static_content.id}"
    
    origin_shield {
      enabled              = true
      origin_shield_region = var.aws_region
    }
  }
  
  # ALB origin for dynamic content
  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "ALB-${aws_lb.main.id}"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
      
      origin_keepalive_timeout = 60
      origin_read_timeout      = 60
    }
    
    custom_header {
      name  = "X-Origin-Verify"
      value = random_password.origin_verify.result
    }
  }
  
  # Origin group for failover
  origin_group {
    origin_id = "origin-group-1"
    
    failover_criteria {
      status_codes = [500, 502, 503, 504]
    }
    
    member {
      origin_id = "ALB-${aws_lb.main.id}"
    }
    
    member {
      origin_id = "ALB-${aws_lb.secondary.id}"
    }
  }
  
  # Default cache behavior
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${aws_s3_bucket.static_content.id}"
    
    forwarded_values {
      query_string = false
      headers      = ["Origin", "Access-Control-Request-Method", "Access-Control-Request-Headers"]
      
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 86400
    max_ttl                = 31536000
    compress               = true
    
    # Lambda@Edge associations
    lambda_function_association {
      event_type   = "viewer-request"
      lambda_arn   = aws_lambda_function.edge_viewer_request.qualified_arn
      include_body = false
    }
    
    lambda_function_association {
      event_type   = "origin-response"
      lambda_arn   = aws_lambda_function.edge_origin_response.qualified_arn
      include_body = false
    }
  }
  
  # Cache behavior for API
  ordered_cache_behavior {
    path_pattern     = "/api/*"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "origin-group-1"
    
    cache_policy_id            = aws_cloudfront_cache_policy.api.id
    origin_request_policy_id   = aws_cloudfront_origin_request_policy.api.id
    response_headers_policy_id = aws_cloudfront_response_headers_policy.security.id
    
    viewer_protocol_policy = "https-only"
    compress               = true
  }
  
  # Cache behavior for static assets
  ordered_cache_behavior {
    path_pattern     = "/static/*"
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${aws_s3_bucket.static_content.id}"
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 604800   # 7 days
    max_ttl                = 31536000  # 1 year
    compress               = true
    
    forwarded_values {
      query_string = false
      
      cookies {
        forward = "none"
      }
    }
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.main.certificate_arn
    minimum_protocol_version = "TLSv1.2_2021"
    ssl_support_method       = "sni-only"
  }
  
  web_acl_id = aws_wafv2_web_acl.cloudfront.arn
  
  logging_config {
    include_cookies = false
    bucket          = aws_s3_bucket.cloudfront_logs.bucket_domain_name
    prefix          = "cloudfront/"
  }
  
  tags = {
    Environment = var.environment
  }
  
  depends_on = [
    aws_s3_bucket_policy.static_content
  ]
}

# Cache policy for API
resource "aws_cloudfront_cache_policy" "api" {
  name        = "${var.environment}-api-cache-policy"
  comment     = "Cache policy for API endpoints"
  default_ttl = 0
  max_ttl     = 3600
  min_ttl     = 0
  
  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_gzip   = true
    enable_accept_encoding_brotli = true
    
    cookies_config {
      cookie_behavior = "all"
    }
    
    headers_config {
      header_behavior = "whitelist"
      headers {
        items = [
          "Authorization",
          "CloudFront-Viewer-Country",
          "CloudFront-Is-Mobile-Viewer",
          "CloudFront-Is-Tablet-Viewer",
          "CloudFront-Is-Desktop-Viewer"
        ]
      }
    }
    
    query_strings_config {
      query_string_behavior = "all"
    }
  }
}

# Origin request policy
resource "aws_cloudfront_origin_request_policy" "api" {
  name    = "${var.environment}-api-origin-request-policy"
  comment = "Origin request policy for API"
  
  cookies_config {
    cookie_behavior = "all"
  }
  
  headers_config {
    header_behavior = "allViewer"
  }
  
  query_strings_config {
    query_string_behavior = "all"
  }
}

# Response headers policy
resource "aws_cloudfront_response_headers_policy" "security" {
  name    = "${var.environment}-security-headers-policy"
  comment = "Security headers policy"
  
  security_headers_config {
    content_type_options {
      override = true
    }
    
    frame_options {
      frame_option = "DENY"
      override     = true
    }
    
    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = true
    }
    
    xss_protection {
      mode_block = true
      protection = true
      override   = true
    }
    
    strict_transport_security {
      access_control_max_age_sec = 63072000
      include_subdomains         = true
      preload                    = true
      override                   = true
    }
    
    content_security_policy {
      content_security_policy = "default-src 'self'; img-src 'self' data: https:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
      override                = true
    }
  }
  
  cors_config {
    access_control_allow_credentials = true
    
    access_control_allow_headers {
      items = ["*"]
    }
    
    access_control_allow_methods {
      items = ["GET", "HEAD", "PUT", "POST", "DELETE", "OPTIONS"]
    }
    
    access_control_allow_origins {
      items = ["https://${var.domain_name}"]
    }
    
    access_control_max_age_sec = 86400
    origin_override            = true
  }
  
  custom_headers_config {
    items {
      header   = "X-Environment"
      value    = var.environment
      override = false
    }
  }
}

# CloudFront monitoring
resource "aws_cloudwatch_metric_alarm" "4xx_errors" {
  alarm_name          = "${var.environment}-cloudfront-4xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "4xxErrorRate"
  namespace           = "AWS/CloudFront"
  period              = "300"
  statistic           = "Average"
  threshold           = "5"
  alarm_description   = "This metric monitors CloudFront 4xx error rate"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DistributionId = aws_cloudfront_distribution.main.id
  }
}

# Real-time logs
resource "aws_cloudfront_realtime_log_config" "main" {
  name = "${var.environment}-realtime-logs"
  
  endpoint {
    stream_type = "Kinesis"
    
    kinesis_stream_config {
      role_arn   = aws_iam_role.cloudfront_logs.arn
      stream_arn = aws_kinesis_stream.cloudfront_logs.arn
    }
  }
  
  fields = [
    "timestamp",
    "c-ip",
    "sc-status",
    "cs-uri-stem",
    "cs-user-agent",
    "cs-referer",
    "x-edge-location",
    "x-edge-request-id",
    "x-edge-response-result-type",
    "time-taken"
  ]
  
  sampling_rate = 1  # 1% sampling
}
```

### RDS Performance Insights

```hcl
# rds-performance.tf - RDS with Performance Insights and monitoring

# RDS subnet group
resource "aws_db_subnet_group" "main" {
  name       = "${var.environment}-db-subnet-group"
  subnet_ids = var.private_subnet_ids
  
  tags = {
    Environment = var.environment
  }
}

# RDS parameter group with performance optimizations
resource "aws_db_parameter_group" "optimized" {
  name   = "${var.environment}-mysql-optimized"
  family = "mysql8.0"
  
  # Query performance parameters
  parameter {
    name  = "slow_query_log"
    value = "1"
  }
  
  parameter {
    name  = "long_query_time"
    value = "1"
  }
  
  parameter {
    name  = "log_queries_not_using_indexes"
    value = "1"
  }
  
  parameter {
    name  = "performance_schema"
    value = "1"
  }
  
  # InnoDB optimization
  parameter {
    name  = "innodb_buffer_pool_size"
    value = "{DBInstanceClassMemory*3/4}"
  }
  
  parameter {
    name  = "innodb_log_file_size"
    value = "1073741824"  # 1GB
  }
  
  parameter {
    name  = "innodb_flush_log_at_trx_commit"
    value = "2"
  }
  
  parameter {
    name  = "innodb_flush_method"
    value = "O_DIRECT"
  }
  
  # Connection optimization
  parameter {
    name  = "max_connections"
    value = "1000"
  }
  
  tags = {
    Environment = var.environment
  }
}

# RDS instance with Performance Insights
resource "aws_db_instance" "main" {
  identifier = "${var.environment}-database"
  
  # Engine configuration
  engine                      = "mysql"
  engine_version              = "8.0.33"
  auto_minor_version_upgrade  = true
  allow_major_version_upgrade = false
  
  # Instance configuration
  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  storage_encrypted = true
  kms_key_id        = aws_kms_key.rds.arn
  storage_type      = "gp3"
  iops              = var.db_iops
  
  # Database configuration
  db_name  = var.db_name
  username = var.db_username
  password = aws_secretsmanager_secret_version.db_password.secret_string
  port     = 3306
  
  # Network configuration
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false
  
  # Parameter and option groups
  parameter_group_name = aws_db_parameter_group.optimized.name
  option_group_name    = aws_db_option_group.main.name
  
  # Backup configuration
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # High availability
  multi_az               = true
  deletion_protection    = true
  skip_final_snapshot    = false
  final_snapshot_identifier = "${var.environment}-database-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  # Performance Insights
  enabled_cloudwatch_logs_exports = ["error", "general", "slowquery"]
  
  performance_insights_enabled          = true
  performance_insights_kms_key_id       = aws_kms_key.rds.arn
  performance_insights_retention_period = 7  # days
  
  # Enhanced monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
  
  tags = {
    Environment = var.environment
  }
}

# RDS Proxy for connection pooling
resource "aws_db_proxy" "main" {
  name                   = "${var.environment}-rds-proxy"
  engine_family          = "MYSQL"
  auth {
    auth_scheme = "SECRETS"
    secret_arn  = aws_secretsmanager_secret.db_password.arn
  }
  
  role_arn               = aws_iam_role.rds_proxy.arn
  vpc_subnet_ids         = var.private_subnet_ids
  
  require_tls                    = true
  idle_client_timeout            = 1800
  max_connections_percent        = 100
  max_idle_connections_percent   = 50
  connection_borrow_timeout      = 120
  
  target {
    db_instance_identifier = aws_db_instance.main.id
  }
  
  tags = {
    Environment = var.environment
  }
}

# CloudWatch dashboard for RDS Performance Insights
resource "aws_cloudwatch_dashboard" "rds_performance" {
  dashboard_name = "${var.environment}-rds-performance-insights"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", aws_db_instance.main.id],
            [".", "DatabaseConnections", ".", "."],
            [".", "FreeableMemory", ".", ".", { stat = "Average" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "RDS Resource Utilization"
        }
      },
      {
        type   = "metric"
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/RDS", "ReadLatency", "DBInstanceIdentifier", aws_db_instance.main.id],
            [".", "WriteLatency", ".", "."],
            [".", "ReadThroughput", ".", ".", { stat = "Sum" }],
            [".", "WriteThroughput", ".", ".", { stat = "Sum" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "RDS I/O Performance"
        }
      },
      {
        type   = "metric"
        width  = 24
        height = 6
        
        properties = {
          metrics = [
            ["AWS/RDS", "DBLoad", "DBInstanceIdentifier", aws_db_instance.main.id],
            [".", "DBLoadCPU", ".", "."],
            [".", "DBLoadNonCPU", ".", "."]
          ]
          period = 60
          stat   = "Average"
          region = var.aws_region
          title  = "Performance Insights - Database Load"
        }
      }
    ]
  })
}

# CloudWatch alarms for RDS
resource "aws_cloudwatch_metric_alarm" "rds_cpu" {
  alarm_name          = "${var.environment}-rds-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors RDS CPU utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }
}

resource "aws_cloudwatch_metric_alarm" "rds_connections" {
  alarm_name          = "${var.environment}-rds-high-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  
  metric_query {
    id          = "connections_percentage"
    expression  = "(connections / max_connections) * 100"
    label       = "Connection Percentage"
    return_data = true
  }
  
  metric_query {
    id = "connections"
    
    metric {
      metric_name = "DatabaseConnections"
      namespace   = "AWS/RDS"
      period      = 300
      stat        = "Average"
      
      dimensions = {
        DBInstanceIdentifier = aws_db_instance.main.id
      }
    }
  }
  
  metric_query {
    id = "max_connections"
    
    metric {
      metric_name = "MaxConnections"
      namespace   = "AWS/RDS"
      period      = 300
      stat        = "Average"
      
      dimensions = {
        DBInstanceIdentifier = aws_db_instance.main.id
      }
    }
  }
  
  threshold           = 80
  alarm_description   = "RDS connection usage above 80%"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# Lambda function for Performance Insights analysis
resource "aws_lambda_function" "pi_analyzer" {
  filename         = "pi_analyzer.zip"
  function_name    = "${var.environment}-performance-insights-analyzer"
  role            = aws_iam_role.pi_analyzer.arn
  handler         = "index.handler"
  runtime         = "python3.9"
  timeout         = 300
  memory_size     = 1024
  
  environment {
    variables = {
      DB_INSTANCE_ID = aws_db_instance.main.id
      SNS_TOPIC_ARN  = aws_sns_topic.alerts.arn
    }
  }
  
  layers = [
    "arn:aws:lambda:${var.aws_region}:336392948345:layer:AWSSDKPandas-Python39:1"
  ]
}

# EventBridge rule to trigger Performance Insights analysis
resource "aws_cloudwatch_event_rule" "pi_analysis" {
  name                = "${var.environment}-pi-analysis"
  description         = "Trigger Performance Insights analysis"
  schedule_expression = "rate(1 hour)"
}

resource "aws_cloudwatch_event_target" "pi_analyzer" {
  rule      = aws_cloudwatch_event_rule.pi_analysis.name
  target_id = "PIAnalyzer"
  arn       = aws_lambda_function.pi_analyzer.arn
}

resource "aws_lambda_permission" "pi_analyzer" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pi_analyzer.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.pi_analysis.arn
}

# RDS automated backups to another region
resource "aws_db_instance_automated_backups_replication" "cross_region" {
  count                      = var.enable_cross_region_backup ? 1 : 0
  source_db_instance_arn     = aws_db_instance.main.arn
  kms_key_id                 = data.aws_kms_key.backup_region.arn
  retention_period           = 7
  
  provider = aws.backup_region
}

# Database Activity Streams
resource "aws_db_activity_stream" "main" {
  count = var.enable_activity_stream ? 1 : 0
  
  resource_arn                        = aws_db_instance.main.arn
  mode                                = "async"
  kms_key_id                          = aws_kms_key.activity_stream.id
  engine_native_audit_fields_included = true
  
  depends_on = [aws_db_instance.main]
}
```

## Security Best Practices

### AWS Security Hub and Compliance

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

## Cost Optimization Strategies

### Advanced Cost Management

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
  runtime         = "python3.9"
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
    "arn:aws:lambda:${var.aws_region}:336392948345:layer:AWSSDKPandas-Python39:1"
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

## Emerging Services and Research

### AWS Quantum Computing and Braket

```hcl
# quantum-computing.tf - AWS Braket quantum computing resources

# S3 bucket for quantum task results
resource "aws_s3_bucket" "quantum_results" {
  bucket = "${var.environment}-quantum-results-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "quantum_results" {
  bucket = aws_s3_bucket.quantum_results.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "quantum_results" {
  bucket = aws_s3_bucket.quantum_results.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# IAM role for Braket
resource "aws_iam_role" "braket" {
  name = "${var.environment}-braket-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "braket.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "braket" {
  name = "${var.environment}-braket-policy"
  role = aws_iam_role.braket.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.quantum_results.arn,
          "${aws_s3_bucket.quantum_results.arn}/*"
        ]
      }
    ]
  })
}

# Lambda function for quantum circuit execution
resource "aws_lambda_function" "quantum_processor" {
  filename         = "quantum_processor.zip"
  function_name    = "${var.environment}-quantum-processor"
  role            = aws_iam_role.quantum_lambda.arn
  handler         = "index.handler"
  runtime         = "python3.9"
  timeout         = 900
  memory_size     = 3008
  
  environment {
    variables = {
      QUANTUM_RESULTS_BUCKET = aws_s3_bucket.quantum_results.id
      ENVIRONMENT           = var.environment
    }
  }
  
  layers = [
    "arn:aws:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:layer:AmazonBraket:1"
  ]
}

# EventBridge rule for quantum task monitoring
resource "aws_cloudwatch_event_rule" "quantum_task_state_change" {
  name        = "${var.environment}-quantum-task-state-change"
  description = "Capture Braket quantum task state changes"
  
  event_pattern = jsonencode({
    source      = ["aws.braket"]
    detail-type = ["Braket Task State Change"]
    detail = {
      status = ["COMPLETED", "FAILED"]
    }
  })
}

resource "aws_cloudwatch_event_target" "quantum_notification" {
  rule      = aws_cloudwatch_event_rule.quantum_task_state_change.name
  target_id = "QuantumNotification"
  arn       = aws_sns_topic.quantum_notifications.arn
}

# SNS topic for quantum task notifications
resource "aws_sns_topic" "quantum_notifications" {
  name = "${var.environment}-quantum-notifications"
  
  kms_master_key_id = aws_kms_key.sns.id
}

# Lambda for quantum-classical hybrid algorithms
resource "aws_lambda_function" "quantum_hybrid" {
  filename         = "quantum_hybrid.zip"
  function_name    = "${var.environment}-quantum-hybrid-optimizer"
  role            = aws_iam_role.quantum_lambda.arn
  handler         = "vqe_optimizer.handler"
  runtime         = "python3.9"
  timeout         = 900
  memory_size     = 10240  # 10GB for optimization tasks
  
  ephemeral_storage {
    size = 10240  # 10GB
  }
  
  environment {
    variables = {
      QUANTUM_DEVICE_ARN = var.quantum_device_arn
      S3_BUCKET         = aws_s3_bucket.quantum_results.id
      MAX_ITERATIONS    = "100"
    }
  }
  
  vpc_config {
    subnet_ids         = var.private_subnet_ids
    security_group_ids = [aws_security_group.lambda.id]
  }
}

# Step Functions for quantum workflow orchestration
resource "aws_sfn_state_machine" "quantum_workflow" {
  name     = "${var.environment}-quantum-workflow"
  role_arn = aws_iam_role.step_functions.arn
  
  definition = jsonencode({
    Comment = "Quantum-Classical Hybrid Workflow"
    StartAt = "PrepareQuantumCircuit"
    States = {
      PrepareQuantumCircuit = {
        Type = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.quantum_processor.arn
          Payload = {
            "action" = "prepare_circuit"
            "parameters.$" = "$.circuit_params"
          }
        }
        ResultPath = "$.circuit"
        Next = "SubmitQuantumTask"
      }
      
      SubmitQuantumTask = {
        Type = "Task"
        Resource = "arn:aws:states:::aws-sdk:braket:createQuantumTask"
        Parameters = {
          DeviceArn = var.quantum_device_arn
          OutputS3Bucket = aws_s3_bucket.quantum_results.id
          OutputS3KeyPrefix = "tasks/"
          Shots = 1000
          Action = {
            "BraketSchemaHeader" = {
              "name" = "braket.ir.jaqcd.program"
              "version" = "1.0"
            }
            "instructions.$" = "$.circuit.instructions"
          }
        }
        ResultPath = "$.quantumTask"
        Next = "WaitForQuantumTask"
      }
      
      WaitForQuantumTask = {
        Type = "Wait"
        Seconds = 30
        Next = "GetQuantumTaskStatus"
      }
      
      GetQuantumTaskStatus = {
        Type = "Task"
        Resource = "arn:aws:states:::aws-sdk:braket:getQuantumTask"
        Parameters = {
          "QuantumTaskArn.$" = "$.quantumTask.QuantumTaskArn"
        }
        Next = "CheckTaskComplete"
      }
      
      CheckTaskComplete = {
        Type = "Choice"
        Choices = [
          {
            Variable = "$.Status"
            StringEquals = "COMPLETED"
            Next = "ProcessResults"
          },
          {
            Variable = "$.Status"
            StringEquals = "FAILED"
            Next = "HandleFailure"
          }
        ]
        Default = "WaitForQuantumTask"
      }
      
      ProcessResults = {
        Type = "Task"
        Resource = aws_lambda_function.quantum_hybrid.arn
        Parameters = {
          "action" = "process_results"
          "results.$" = "$.OutputS3Uri"
        }
        End = true
      }
      
      HandleFailure = {
        Type = "Fail"
        Cause = "Quantum task failed"
      }
    }
  })
}
```

### Machine Learning on AWS

```hcl
# sagemaker.tf - SageMaker MLOps infrastructure

# SageMaker execution role
resource "aws_iam_role" "sagemaker_execution" {
  name = "${var.environment}-sagemaker-execution-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
  
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  ]
}

# Feature Store
resource "aws_sagemaker_feature_group" "main" {
  feature_group_name             = "${var.environment}-feature-group"
  record_identifier_feature_name = "record_id"
  event_time_feature_name        = "event_time"
  role_arn                       = aws_iam_role.sagemaker_execution.arn
  
  feature_definition {
    feature_name = "record_id"
    feature_type = "String"
  }
  
  feature_definition {
    feature_name = "event_time"
    feature_type = "String"
  }
  
  feature_definition {
    feature_name = "user_id"
    feature_type = "String"
  }
  
  feature_definition {
    feature_name = "feature_1"
    feature_type = "Fractional"
  }
  
  feature_definition {
    feature_name = "feature_2"
    feature_type = "Fractional"
  }
  
  feature_definition {
    feature_name = "label"
    feature_type = "Integral"
  }
  
  online_store_config {
    enable_online_store = true
    
    security_config {
      kms_key_id = aws_kms_key.sagemaker.id
    }
  }
  
  offline_store_config {
    s3_storage_config {
      s3_uri = "s3://${aws_s3_bucket.feature_store.id}/offline-store"
      
      kms_key_id = aws_kms_key.s3.id
    }
    
    data_catalog_config {
      database   = aws_glue_catalog_database.feature_store.name
      table_name = "${var.environment}_features"
    }
  }
  
  tags = {
    Environment = var.environment
  }
}

# Model registry
resource "aws_sagemaker_model_package_group" "main" {
  model_package_group_name = "${var.environment}-model-registry"
  model_package_group_description = "Model registry for ML models"
  
  tags = {
    Environment = var.environment
  }
}

# SageMaker model
resource "aws_sagemaker_model" "main" {
  name               = "${var.environment}-model"
  execution_role_arn = aws_iam_role.sagemaker_execution.arn
  
  primary_container {
    image          = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/ml-model:latest"
    model_data_url = "s3://${aws_s3_bucket.models.id}/model.tar.gz"
    
    environment = {
      SAGEMAKER_PROGRAM           = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY  = "s3://${aws_s3_bucket.models.id}/code"
      SAGEMAKER_ENABLE_CLOUDWATCH_METRICS = "true"
    }
  }
  
  vpc_config {
    subnets            = var.private_subnet_ids
    security_group_ids = [aws_security_group.sagemaker.id]
  }
  
  tags = {
    Environment = var.environment
  }
}

# Multi-model endpoint configuration
resource "aws_sagemaker_endpoint_configuration" "multi_model" {
  name = "${var.environment}-multi-model-config"
  
  production_variants {
    variant_name           = "AllTraffic"
    model_name            = aws_sagemaker_model.main.name
    initial_instance_count = 2
    instance_type         = "ml.m5.xlarge"
    initial_variant_weight = 1
    
    # Enable multi-model
    model_data_download_timeout_in_seconds = 600
    container_startup_health_check_timeout_in_seconds = 600
  }
  
  data_capture_config {
    enable_capture = true
    initial_sampling_percentage = 100
    destination_s3_uri = "s3://${aws_s3_bucket.model_data_capture.id}/"
    
    capture_options {
      capture_mode = "Input"
    }
    
    capture_options {
      capture_mode = "Output"
    }
    
    capture_content_type_header {
      json_content_types = ["application/json"]
    }
  }
  
  tags = {
    Environment = var.environment
  }
}

# SageMaker endpoint
resource "aws_sagemaker_endpoint" "main" {
  name                 = "${var.environment}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.multi_model.name
  
  deployment_config {
    blue_green_update_policy {
      traffic_routing_configuration {
        type = "LINEAR"
        wait_interval_in_seconds = 300
        
        linear_step_size {
          type  = "INSTANCE_COUNT"
          value = 1
        }
      }
      
      maximum_execution_timeout_in_seconds = 3600
    }
    
    auto_rollback_configuration {
      alarms {
        alarm_name = aws_cloudwatch_metric_alarm.endpoint_error_rate.alarm_name
      }
    }
  }
  
  tags = {
    Environment = var.environment
  }
}

# Auto-scaling for SageMaker endpoint
resource "aws_appautoscaling_target" "sagemaker_target" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "endpoint/${aws_sagemaker_endpoint.main.name}/variant/AllTraffic"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "sagemaker_target_tracking" {
  name               = "${var.environment}-sagemaker-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker_target.resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
    
    target_value = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

# SageMaker Pipeline
resource "aws_sagemaker_pipeline" "ml_pipeline" {
  pipeline_name         = "${var.environment}-ml-pipeline"
  pipeline_display_name = "ML Training Pipeline"
  role_arn             = aws_iam_role.sagemaker_execution.arn
  
  pipeline_definition = jsonencode({
    Version = "2020-12-01"
    Parameters = [
      {
        Name = "ProcessingInstanceCount"
        Type = "Integer"
        DefaultValue = 1
      },
      {
        Name = "TrainingInstanceType"
        Type = "String"
        DefaultValue = "ml.m5.xlarge"
      }
    ]
    Steps = [
      {
        Name = "DataProcessing"
        Type = "Processing"
        Arguments = {
          ProcessingResources = {
            ClusterConfig = {
              InstanceCount = { "Get" = "Parameters.ProcessingInstanceCount" }
              InstanceType = "ml.m5.xlarge"
              VolumeSizeInGB = 30
            }
          }
          AppSpecification = {
            ImageUri = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/processing:latest"
          }
          RoleArn = aws_iam_role.sagemaker_execution.arn
          ProcessingInputs = [
            {
              InputName = "input-data"
              S3Input = {
                S3Uri = "s3://${aws_s3_bucket.training_data.id}/raw/"
                LocalPath = "/opt/ml/processing/input"
                S3DataType = "S3Prefix"
                S3InputMode = "File"
                S3DataDistributionType = "FullyReplicated"
              }
            }
          ]
          ProcessingOutputConfig = {
            Outputs = [
              {
                OutputName = "processed-data"
                S3Output = {
                  S3Uri = "s3://${aws_s3_bucket.training_data.id}/processed/"
                  LocalPath = "/opt/ml/processing/output"
                  S3UploadMode = "EndOfJob"
                }
              }
            ]
          }
        }
      },
      {
        Name = "ModelTraining"
        Type = "Training"
        Arguments = {
          AlgorithmSpecification = {
            TrainingImage = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/training:latest"
            TrainingInputMode = "File"
            EnableSageMakerMetricsTimeSeries = true
          }
          RoleArn = aws_iam_role.sagemaker_execution.arn
          OutputDataConfig = {
            S3OutputPath = "s3://${aws_s3_bucket.models.id}/"
            KmsKeyId = aws_kms_key.s3.id
          }
          ResourceConfig = {
            InstanceCount = 1
            InstanceType = { "Get" = "Parameters.TrainingInstanceType" }
            VolumeSizeInGB = 30
          }
          StoppingCondition = {
            MaxRuntimeInSeconds = 86400
          }
          HyperParameters = {
            epochs = "10"
            batch_size = "32"
            learning_rate = "0.001"
          }
          InputDataConfig = [
            {
              ChannelName = "training"
              DataSource = {
                S3DataSource = {
                  S3DataType = "S3Prefix"
                  S3Uri = { "Get" = "Steps.DataProcessing.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri" }
                  S3DataDistributionType = "FullyReplicated"
                }
              }
              ContentType = "application/x-parquet"
              CompressionType = "None"
            }
          ]
        }
        DependsOn = ["DataProcessing"]
      },
      {
        Name = "RegisterModel"
        Type = "RegisterModel"
        Arguments = {
          ModelPackageGroupName = aws_sagemaker_model_package_group.main.model_package_group_name
          ModelMetrics = {
            ModelQuality = {
              Statistics = {
                ContentType = "application/json"
                S3Uri = "s3://${aws_s3_bucket.models.id}/evaluation/statistics.json"
              }
            }
          }
          InferenceSpecification = {
            Containers = [
              {
                Image = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/inference:latest"
                ModelDataUrl = { "Get" = "Steps.ModelTraining.ModelArtifacts.S3ModelArtifacts" }
              }
            ]
            SupportedContentTypes = ["application/json"]
            SupportedResponseMIMETypes = ["application/json"]
          }
          ModelApprovalStatus = "PendingManualApproval"
        }
        DependsOn = ["ModelTraining"]
      }
    ]
  })
  
  tags = {
    Environment = var.environment
  }
}

# CloudWatch monitoring for ML endpoints
resource "aws_cloudwatch_metric_alarm" "endpoint_error_rate" {
  alarm_name          = "${var.environment}-endpoint-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ModelInvocation4XXErrors"
  namespace           = "AWS/SageMaker"
  period              = "300"
  statistic           = "Average"
  threshold           = "0.05"
  alarm_description   = "This metric monitors endpoint error rate"
  alarm_actions       = [aws_sns_topic.ml_alerts.arn]
  
  dimensions = {
    EndpointName = aws_sagemaker_endpoint.main.name
    VariantName  = "AllTraffic"
  }
}

# Model monitoring schedule
resource "aws_sagemaker_monitoring_schedule" "model_quality" {
  name = "${var.environment}-model-quality-monitor"
  
  monitoring_schedule_config {
    monitoring_job_definition_name = aws_sagemaker_data_quality_job_definition.main.name
    monitoring_type                = "DataQuality"
    
    schedule_config {
      schedule_expression = "cron(0 * ? * * *)"  # Every hour
    }
  }
  
  tags = {
    Environment = var.environment
  }
}
```

## Infrastructure as Code Best Practices

### Terraform Advanced Patterns

```hcl
from aws_cdk import (
    core as cdk,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elbv2,
    aws_rds as rds,
    aws_secretsmanager as sm,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    custom_resources as cr
)
from constructs import Construct
import json

class MicroservicesStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Create VPC with custom configuration
        vpc = ec2.Vpc(
            self, "MicroservicesVPC",
            max_azs=3,
            nat_gateways=2,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Isolated",
                    subnet_type=ec2.SubnetType.ISOLATED,
                    cidr_mask=24
                )
            ]
        )
        
        # Create ECS Cluster with capacity providers
        cluster = ecs.Cluster(
            self, "Cluster",
            vpc=vpc,
            container_insights=True
        )
        
        # Add Fargate Spot capacity provider
        cluster.add_capacity_provider(
            ecs.FargateCapacityProvider(
                self, "FargateSpotProvider",
                spot=True
            )
        )
        
        # Create RDS Aurora Serverless v2
        db_secret = sm.Secret(
            self, "DBSecret",
            generate_secret_string=sm.SecretStringGenerator(
                secret_string_template=json.dumps({"username": "admin"}),
                generate_string_key="password",
                exclude_characters=" %+~`#$&*()|[]{}:;<>?!'/\\"
            )
        )
        
        db_cluster = rds.DatabaseCluster(
            self, "AuroraCluster",
            engine=rds.DatabaseClusterEngine.aurora_mysql(
                version=rds.AuroraMysqlEngineVersion.VER_3_01_0
            ),
            serverless_v2_scaling_configuration=rds.ServerlessV2ScalingConfiguration(
                min_capacity=0.5,
                max_capacity=2
            ),
            credentials=rds.Credentials.from_secret(db_secret),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.ISOLATED
            ),
            backup=rds.BackupProps(
                retention=cdk.Duration.days(7)
            ),
            deletion_protection=True
        )
        
        # Create shared ALB
        alb = elbv2.ApplicationLoadBalancer(
            self, "ALB",
            vpc=vpc,
            internet_facing=True,
            http2_enabled=True
        )
        
        # Add CloudWatch alarms
        alarm = cloudwatch.Alarm(
            self, "HighErrorRate",
            metric=alb.metric_target_response_time(),
            threshold=1000,
            evaluation_periods=2
        )
        
        # SNS topic for alarms
        alarm_topic = sns.Topic(
            self, "AlarmTopic",
            display_name="Microservices Alarms"
        )
        
        alarm.add_alarm_action(cw_actions.SnsAction(alarm_topic))
        
        # Deploy microservices
        self.deploy_microservice(
            cluster=cluster,
            alb=alb,
            service_name="users",
            image="users-service:latest",
            port=8080,
            priority=1,
            path_pattern="/users/*",
            environment={
                "DB_SECRET_ARN": db_secret.secret_arn,
                "DB_CLUSTER_ARN": db_cluster.cluster_arn
            }
        )
        
        self.deploy_microservice(
            cluster=cluster,
            alb=alb,
            service_name="orders",
            image="orders-service:latest",
            port=8081,
            priority=2,
            path_pattern="/orders/*",
            environment={
                "DB_SECRET_ARN": db_secret.secret_arn,
                "DB_CLUSTER_ARN": db_cluster.cluster_arn
            }
        )
        
        # Create API Gateway for serverless endpoints
        api = apigw.RestApi(
            self, "MicroservicesAPI",
            deploy_options=apigw.StageOptions(
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=True,
                tracing_enabled=True
            )
        )
        
        # Lambda function for async processing
        async_processor = lambda_.Function(
            self, "AsyncProcessor",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda"),
            vpc=vpc,
            environment={
                "DB_SECRET_ARN": db_secret.secret_arn
            },
            reserved_concurrent_executions=100,
            tracing=lambda_.Tracing.ACTIVE
        )
        
        # Grant permissions
        db_secret.grant_read(async_processor)
        db_cluster.grant_connect(async_processor)
        
        # Custom resource for database initialization
        db_init = cr.AwsCustomResource(
            self, "DBInit",
            on_create=cr.AwsSdkCall(
                service="RDS",
                action="executeStatement",
                parameters={
                    "resourceArn": db_cluster.cluster_arn,
                    "secretArn": db_secret.secret_arn,
                    "database": "mysql",
                    "sql": "CREATE DATABASE IF NOT EXISTS microservices;"
                },
                physical_resource_id=cr.PhysicalResourceId.of("DBInit")
            ),
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=[db_cluster.cluster_arn]
            )
        )
        
        # Output values
        cdk.CfnOutput(
            self, "ALBDNSName",
            value=alb.load_balancer_dns_name,
            description="ALB DNS Name"
        )
        
        cdk.CfnOutput(
            self, "APIEndpoint",
            value=api.url,
            description="API Gateway Endpoint"
        )
    
    def deploy_microservice(self,
                           cluster: ecs.Cluster,
                           alb: elbv2.ApplicationLoadBalancer,
                           service_name: str,
                           image: str,
                           port: int,
                           priority: int,
                           path_pattern: str,
                           environment: dict):
        """Deploy a microservice to ECS"""
        
        # Create task definition
        task_definition = ecs.FargateTaskDefinition(
            self, f"{service_name}TaskDef",
            memory_limit_mib=512,
            cpu=256
        )
        
        # Add container
        container = task_definition.add_container(
            f"{service_name}Container",
            image=ecs.ContainerImage.from_registry(image),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=service_name
            ),
            environment=environment,
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", f"curl -f http://localhost:{port}/health || exit 1"],
                interval=cdk.Duration.seconds(30),
                timeout=cdk.Duration.seconds(5),
                retries=3
            )
        )
        
        container.add_port_mappings(
            ecs.PortMapping(
                container_port=port,
                protocol=ecs.Protocol.TCP
            )
        )
        
        # Create service
        service = ecs.FargateService(
            self, f"{service_name}Service",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=2,
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(
                    capacity_provider="FARGATE_SPOT",
                    weight=2
                ),
                ecs.CapacityProviderStrategy(
                    capacity_provider="FARGATE",
                    weight=1
                )
            ],
            circuit_breaker=ecs.DeploymentCircuitBreaker(
                rollback=True
            )
        )
        
        # Configure auto-scaling
        scaling = service.auto_scale_task_count(
            min_capacity=2,
            max_capacity=10
        )
        
        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=cdk.Duration.seconds(60),
            scale_out_cooldown=cdk.Duration.seconds(60)
        )
        
        scaling.scale_on_request_count(
            "RequestScaling",
            requests_per_target=1000,
            target_group=alb.add_targets(
                f"{service_name}TG",
                port=port,
                targets=[service],
                health_check=elbv2.HealthCheck(
                    path=f"/{service_name}/health",
                    interval=cdk.Duration.seconds(30)
                )
            )
        )
        
        # Add ALB listener rule
        alb.add_listener(
            f"{service_name}Listener",
            port=80
        ).add_targets(
            f"{service_name}Targets",
            port=port,
            targets=[service],
            priority=priority,
            conditions=[
                elbv2.ListenerCondition.path_patterns([path_pattern])
            ]
        )
```

## Future Directions

1. **Serverless-First Architecture** - Continued evolution toward event-driven, serverless patterns
2. **Edge Computing** - AWS Wavelength and Local Zones for ultra-low latency
3. **Quantum-Classical Hybrid** - Integration of quantum computing with classical workloads
4. **AI/ML Democratization** - No-code/low-code ML solutions
5. **Sustainability** - Carbon-aware computing and green cloud initiatives
6. **Web3 Integration** - Blockchain and decentralized application support
7. **Advanced Observability** - AI-driven anomaly detection and predictive scaling

