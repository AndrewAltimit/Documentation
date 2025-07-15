# Event-driven architecture with EventBridge
# This demonstrates a comprehensive event-driven system with EventBridge, Lambda, Step Functions, and API Gateway

# Custom event bus for domain events
resource "aws_cloudwatch_event_bus" "domain_events" {
  name = "${var.environment}-domain-events"
  
  tags = {
    Environment = var.environment
    Purpose     = "Domain event processing"
  }
}

# Archive for event replay capability
resource "aws_cloudwatch_event_archive" "domain_events_archive" {
  name             = "${var.environment}-domain-events-archive"
  event_source_arn = aws_cloudwatch_event_bus.domain_events.arn
  retention_days   = 7
  
  event_pattern = jsonencode({
    account = [data.aws_caller_identity.current.account_id]
  })
}

# Dead letter queue for failed events
resource "aws_sqs_queue" "event_dlq" {
  name                      = "${var.environment}-event-dlq"
  message_retention_seconds = 1209600  # 14 days
  
  tags = {
    Environment = var.environment
    Purpose     = "EventBridge DLQ"
  }
}

# Lambda function for order processing
resource "aws_lambda_function" "order_processor" {
  filename         = "order_processor.zip"
  function_name    = "${var.environment}-order-processor"
  role            = aws_iam_role.lambda_execution.arn
  handler         = "index.handler"
  runtime         = "python3.9"
  timeout         = 30
  memory_size     = 512
  
  environment {
    variables = {
      ENVIRONMENT = var.environment
      TABLE_NAME  = aws_dynamodb_table.orders.name
    }
  }
  
  tracing_config {
    mode = "Active"
  }
  
  dead_letter_config {
    target_arn = aws_sqs_queue.event_dlq.arn
  }
}

# EventBridge rule for order processing
resource "aws_cloudwatch_event_rule" "order_processing" {
  name           = "${var.environment}-order-processing"
  description    = "Trigger order processing workflow"
  event_bus_name = aws_cloudwatch_event_bus.domain_events.name
  
  event_pattern = jsonencode({
    source      = ["order.service"]
    detail-type = ["Order Placed", "Order Updated"]
    detail = {
      status = ["pending", "processing"]
    }
  })
  
  tags = {
    Environment = var.environment
    Service     = "order-processing"
  }
}

# Lambda target for order processing rule
resource "aws_cloudwatch_event_target" "order_processor" {
  rule           = aws_cloudwatch_event_rule.order_processing.name
  event_bus_name = aws_cloudwatch_event_bus.domain_events.name
  target_id      = "order-processor-lambda"
  arn            = aws_lambda_function.order_processor.arn
  
  retry_policy {
    maximum_retry_attempts       = 2
    maximum_event_age_in_seconds = 3600
  }
  
  dead_letter_config {
    arn = aws_sqs_queue.event_dlq.arn
  }
  
  input_transformer {
    input_paths = {
      order_id = "$.detail.orderId"
      status   = "$.detail.status"
      amount   = "$.detail.amount"
    }
    
    input_template = <<EOF
{
  "orderId": <order_id>,
  "status": <status>,
  "amount": <amount>,
  "processedAt": "${timestamp()}"
}
EOF
  }
}

# Permission for EventBridge to invoke Lambda
resource "aws_lambda_permission" "eventbridge_invoke" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.order_processor.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.order_processing.arn
}

# Error handling rule
resource "aws_cloudwatch_event_rule" "error_handler" {
  name           = "${var.environment}-error-handler"
  description    = "Handle all error events"
  event_bus_name = aws_cloudwatch_event_bus.domain_events.name
  
  event_pattern = jsonencode({
    source      = [{ exists = true }]
    detail-type = [{ exists = true }]
    detail = {
      error = [{ exists = true }]
    }
  })
}

# SNS topic for error notifications
resource "aws_sns_topic" "error_notifications" {
  name = "${var.environment}-error-notifications"
  
  kms_master_key_id = aws_kms_key.sns.id
}

# SNS target for error handling
resource "aws_cloudwatch_event_target" "error_sns" {
  rule           = aws_cloudwatch_event_rule.error_handler.name
  event_bus_name = aws_cloudwatch_event_bus.domain_events.name
  target_id      = "error-sns-target"
  arn            = aws_sns_topic.error_notifications.arn
}

# Step Functions for complex workflows
resource "aws_sfn_state_machine" "order_workflow" {
  name     = "${var.environment}-order-workflow"
  role_arn = aws_iam_role.step_functions.arn
  
  definition = jsonencode({
    Comment = "Order processing workflow"
    StartAt = "ValidateOrder"
    States = {
      ValidateOrder = {
        Type     = "Task"
        Resource = aws_lambda_function.validate_order.arn
        Next     = "ProcessPayment"
        Retry = [
          {
            ErrorEquals     = ["States.TaskFailed"]
            IntervalSeconds = 2
            MaxAttempts     = 3
            BackoffRate     = 2
          }
        ]
        Catch = [
          {
            ErrorEquals = ["ValidationError"]
            Next        = "HandleValidationError"
          }
        ]
      }
      ProcessPayment = {
        Type = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.process_payment.arn
          "Payload.$" = "$"
        }
        End = true
      }
      HandleValidationError = {
        Type = "Task"
        Resource = aws_lambda_function.handle_error.arn
        End = true
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
}

# EventBridge target for Step Functions
resource "aws_cloudwatch_event_target" "step_functions" {
  rule           = aws_cloudwatch_event_rule.order_processing.name
  event_bus_name = aws_cloudwatch_event_bus.domain_events.name
  target_id      = "order-workflow-target"
  arn            = aws_sfn_state_machine.order_workflow.arn
  role_arn       = aws_iam_role.eventbridge_sfn.arn
}

# API Gateway for event publishing
resource "aws_api_gateway_rest_api" "event_api" {
  name        = "${var.environment}-event-api"
  description = "API for publishing domain events"
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

resource "aws_api_gateway_resource" "events" {
  rest_api_id = aws_api_gateway_rest_api.event_api.id
  parent_id   = aws_api_gateway_rest_api.event_api.root_resource_id
  path_part   = "events"
}

resource "aws_api_gateway_method" "post_event" {
  rest_api_id   = aws_api_gateway_rest_api.event_api.id
  resource_id   = aws_api_gateway_resource.events.id
  http_method   = "POST"
  authorization = "AWS_IAM"
}

# Integration with EventBridge
resource "aws_api_gateway_integration" "eventbridge" {
  rest_api_id = aws_api_gateway_rest_api.event_api.id
  resource_id = aws_api_gateway_resource.events.id
  http_method = aws_api_gateway_method.post_event.http_method
  
  integration_http_method = "POST"
  type                    = "AWS"
  uri                     = "arn:aws:apigateway:${var.aws_region}:events:path//"
  credentials             = aws_iam_role.api_gateway_eventbridge.arn
  
  request_templates = {
    "application/json" = <<EOF
#set($context.requestOverride.header.X-Amz-Target = "AWSEvents.PutEvents")
#set($context.requestOverride.header.Content-Type = "application/x-amz-json-1.1")
{
  "Entries": [
    {
      "Source": "$input.path('$.source')",
      "DetailType": "$input.path('$.detailType')",
      "Detail": "$util.escapeJavaScript($input.path('$.detail'))",
      "EventBusName": "${aws_cloudwatch_event_bus.domain_events.name}"
    }
  ]
}
EOF
  }
}