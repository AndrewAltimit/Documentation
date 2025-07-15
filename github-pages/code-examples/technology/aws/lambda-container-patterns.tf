# Lambda functions using container images and advanced patterns
# This demonstrates Lambda with containers, layers, extensions, and auto-scaling

# ECR repository for Lambda container images
resource "aws_ecr_repository" "lambda_container" {
  name                 = "${var.environment}-lambda-functions"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ecr.arn
  }
}

# ECR lifecycle policy
resource "aws_ecr_lifecycle_policy" "lambda_container" {
  repository = aws_ecr_repository.lambda_container.name
  
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Expire untagged images after 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# Lambda function with container image
resource "aws_lambda_function" "container_function" {
  function_name = "${var.environment}-container-function"
  role          = aws_iam_role.lambda_execution.arn
  
  package_type = "Image"
  image_uri    = "${aws_ecr_repository.lambda_container.repository_url}:latest"
  
  timeout     = 900  # 15 minutes
  memory_size = 3008 # 3GB
  
  environment {
    variables = {
      ENVIRONMENT        = var.environment
      LOG_LEVEL          = "INFO"
      POWERTOOLS_SERVICE_NAME = "container-function"
      POWERTOOLS_METRICS_NAMESPACE = "${var.environment}/lambda"
    }
  }
  
  ephemeral_storage {
    size = 10240  # 10GB
  }
  
  tracing_config {
    mode = "Active"
  }
  
  architectures = ["x86_64"]  # or ["arm64"] for Graviton2
  
  # Reserved concurrent executions
  reserved_concurrent_executions = 100
  
  # Dead letter queue
  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq.arn
  }
  
  # VPC configuration for private resources
  vpc_config {
    subnet_ids         = var.private_subnet_ids
    security_group_ids = [aws_security_group.lambda.id]
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.lambda_vpc_access,
    aws_cloudwatch_log_group.lambda
  ]
}

# Lambda Layers for shared dependencies
resource "aws_lambda_layer_version" "powertools" {
  filename            = "powertools-layer.zip"
  layer_name          = "${var.environment}-aws-lambda-powertools"
  compatible_runtimes = ["python3.9", "python3.10"]
  description         = "AWS Lambda Powertools for Python"
  
  source_code_hash = filebase64sha256("powertools-layer.zip")
}

# Lambda function with layers and extensions
resource "aws_lambda_function" "advanced_function" {
  filename         = "function.zip"
  function_name    = "${var.environment}-advanced-function"
  role            = aws_iam_role.lambda_execution.arn
  handler         = "index.handler"
  runtime         = "python3.9"
  timeout         = 30
  memory_size     = 1024
  
  layers = [
    aws_lambda_layer_version.powertools.arn,
    "arn:aws:lambda:${var.aws_region}:901920570463:layer:aws-otel-python-amd64-ver-1-15-0:1"  # OpenTelemetry
  ]
  
  environment {
    variables = {
      ENVIRONMENT                   = var.environment
      POWERTOOLS_SERVICE_NAME       = "advanced-function"
      POWERTOOLS_METRICS_NAMESPACE  = "${var.environment}/lambda"
      POWERTOOLS_LOGGER_LOG_EVENT   = "true"
      POWERTOOLS_LOGGER_SAMPLE_RATE = "0.1"
      POWERTOOLS_TRACE_ENABLED      = "true"
      AWS_LAMBDA_EXEC_WRAPPER       = "/opt/otel-instrument"
    }
  }
  
  # Enable function URL with IAM auth
  publish = true
}

# Lambda function URL
resource "aws_lambda_function_url" "advanced_function" {
  function_name      = aws_lambda_function.advanced_function.function_name
  authorization_type = "AWS_IAM"
  
  cors {
    allow_credentials = true
    allow_origins     = ["https://example.com"]
    allow_methods     = ["GET", "POST"]
    allow_headers     = ["date", "keep-alive"]
    expose_headers    = ["date", "keep-alive"]
    max_age           = 86400
  }
}

# Lambda alias with weighted traffic shifting
resource "aws_lambda_alias" "live" {
  name             = "live"
  description      = "Live alias with traffic shifting"
  function_name    = aws_lambda_function.advanced_function.function_name
  function_version = aws_lambda_function.advanced_function.version
  
  routing_config {
    additional_version_weights = {
      "${aws_lambda_function.advanced_function.version}" = 0.1  # 10% to new version
    }
  }
}

# CloudWatch Logs
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.environment}-advanced-function"
  retention_in_days = 14
  kms_key_id        = aws_kms_key.logs.arn
}

# Lambda Insights
resource "aws_cloudwatch_log_group" "lambda_insights" {
  name              = "/aws/lambda-insights"
  retention_in_days = 7
}

# X-Ray tracing configuration
resource "aws_xray_sampling_rule" "lambda" {
  rule_name      = "${var.environment}-lambda-sampling"
  priority       = 1000
  version        = 1
  reservoir_size = 1
  fixed_rate     = 0.05  # 5% sampling
  url_path       = "*"
  host           = "*"
  http_method    = "*"
  service_type   = "*"
  service_name   = "*"
  resource_arn   = "*"
}

# Lambda destinations for async invocations
resource "aws_lambda_event_invoke_config" "advanced_function" {
  function_name = aws_lambda_function.advanced_function.function_name
  
  maximum_event_age_in_seconds = 21600  # 6 hours
  maximum_retry_attempts       = 2
  
  destination_config {
    on_success {
      destination = aws_sqs_queue.success_queue.arn
    }
    
    on_failure {
      destination = aws_sns_topic.failure_notifications.arn
    }
  }
}

# Lambda Provisioned Concurrency
resource "aws_lambda_provisioned_concurrency_config" "advanced_function" {
  function_name                     = aws_lambda_function.advanced_function.function_name
  provisioned_concurrent_executions = 5
  qualifier                         = aws_lambda_alias.live.name
}

# Application Auto Scaling for provisioned concurrency
resource "aws_appautoscaling_target" "lambda_concurrency" {
  max_capacity       = 100
  min_capacity       = 5
  resource_id        = "function:${aws_lambda_function.advanced_function.function_name}:provisioned-concurrency:${aws_lambda_alias.live.name}"
  scalable_dimension = "lambda:function:ProvisionedConcurrency"
  service_namespace  = "lambda"
}