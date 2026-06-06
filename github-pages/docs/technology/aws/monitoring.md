---
layout: docs
title: "AWS Monitoring & Messaging"
permalink: /docs/technology/aws/monitoring.html
hide_title: true
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "server"
---

<div class="hero-section" style="background: linear-gradient(135deg, #ff9900 0%, #ffb84d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">AWS Monitoring &amp; Messaging</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">CloudWatch metrics, logs, and alarms plus SNS/SQS messaging — the observability and integration layer that keeps a cloud system healthy and loosely coupled.</p>
</div>

You can only operate what you can see, and resilient systems are built from components that do not block on each other. This page covers the two halves of that story: **CloudWatch** for monitoring and management, and **SNS/SQS** for messaging and integration. It closes with a proactive monitoring and troubleshooting toolkit you can reach for when something looks off.

---

## Monitoring and Management: Your Cloud Operations Center

### Amazon CloudWatch - Your Eyes in the Cloud
CloudWatch is AWS's monitoring service that collects and tracks metrics, logs, and events from every part of your infrastructure. Think of it as your 24/7 operations center.

**Key CloudWatch Features**:

1. **Metrics and Alarms**
   ```bash
   # Create billing alarm to avoid surprises
   aws cloudwatch put-metric-alarm \
     --alarm-name billing-alarm \
     --alarm-description "Alert when AWS charges exceed $100" \
     --metric-name EstimatedCharges \
     --namespace AWS/Billing \
     --statistic Maximum \
     --period 86400 \
     --threshold 100 \
     --comparison-operator GreaterThanThreshold \
     --dimensions Name=Currency,Value=USD
   ```

2. **Custom Metrics for Application Monitoring**
   ```python
   import boto3
   import time

   cloudwatch = boto3.client('cloudwatch')

   # Send custom metric
   def track_user_login(user_id):
       cloudwatch.put_metric_data(
           Namespace='MyApp',
           MetricData=[
               {
                   'MetricName': 'UserLogins',
                   'Dimensions': [
                       {
                           'Name': 'Environment',
                           'Value': 'Production'
                       }
                   ],
                   'Value': 1,
                   'Unit': 'Count',
                   'Timestamp': time.time()
               }
           ]
       )
   ```

3. **Log Insights for Troubleshooting**
   ```sql
   -- Find slowest API endpoints
   fields @timestamp, duration, path
   | filter duration > 1000
   | sort duration desc
   | limit 20

   -- Count errors by type
   fields @timestamp, error_type
   | filter @message like /ERROR/
   | stats count() by error_type
   ```

**Common CloudWatch Pitfall**: Not setting up log retention, leading to unexpected costs. Always set retention policies:
```bash
aws logs put-retention-policy \
  --log-group-name /aws/lambda/my-function \
  --retention-in-days 30
```

---

## Messaging and Integration: Connecting Your Services

### Amazon SNS - Simple Notification Service
SNS is a pub/sub messaging service that lets you send messages to multiple subscribers. Think of it as a broadcasting system for your applications.

**Real-world SNS Patterns**:

1. **Multi-Channel Notifications**
   ```python
   import boto3

   sns = boto3.client('sns')

   # Create topic
   topic = sns.create_topic(Name='order-updates')
   topic_arn = topic['TopicArn']

   # Subscribe email
   sns.subscribe(
       TopicArn=topic_arn,
       Protocol='email',
       Endpoint='customer@example.com'
   )

   # Subscribe SMS
   sns.subscribe(
       TopicArn=topic_arn,
       Protocol='sms',
       Endpoint='+1234567890'
   )

   # Subscribe Lambda for processing
   sns.subscribe(
       TopicArn=topic_arn,
       Protocol='lambda',
       Endpoint='arn:aws:lambda:region:account:function:process-order'
   )

   # Send notification to all subscribers
   sns.publish(
       TopicArn=topic_arn,
       Message='Order #12345 has been shipped!',
       Subject='Order Update'
   )
   ```

2. **Fan-out Pattern for Microservices**
   ```python
   # Order service publishes once
   def complete_order(order_id):
       sns.publish(
           TopicArn='arn:aws:sns:region:account:order-completed',
           Message=json.dumps({
               'orderId': order_id,
               'timestamp': datetime.now().isoformat(),
               'amount': 99.99
           })
       )

   # Multiple services subscribe and react
   # - Inventory service updates stock
   # - Email service sends confirmation
   # - Analytics service records sale
   # - Shipping service creates label
   ```

### Amazon SQS - Simple Queue Service
SQS provides reliable, scalable message queues. Unlike SNS (push), SQS is pull-based - consumers request messages when ready to process them.

**SQS Best Practices**:

1. **Decoupling with Standard Queues**
   ```python
   import boto3
   import json

   sqs = boto3.client('sqs')
   queue_url = 'https://sqs.region.amazonaws.com/account/my-queue'

   # Producer: Send messages
   def send_task(task_data):
       sqs.send_message(
           QueueUrl=queue_url,
           MessageBody=json.dumps(task_data),
           MessageAttributes={
               'Priority': {
                   'StringValue': 'High',
                   'DataType': 'String'
               }
           }
       )

   # Consumer: Process messages
   def process_messages():
       while True:
           response = sqs.receive_message(
               QueueUrl=queue_url,
               MaxNumberOfMessages=10,
               WaitTimeSeconds=20  # Long polling
           )

           for message in response.get('Messages', []):
               # Process message
               process_task(json.loads(message['Body']))

               # Delete after successful processing
               sqs.delete_message(
                   QueueUrl=queue_url,
                   ReceiptHandle=message['ReceiptHandle']
               )
   ```

2. **FIFO Queues for Order Guarantees**
   ```python
   # Create FIFO queue for ordered processing
   fifo_queue = sqs.create_queue(
       QueueName='payment-processing.fifo',
       Attributes={
           'FifoQueue': 'true',
           'ContentBasedDeduplication': 'true'
       }
   )

   # Send with message group ID for ordering
   sqs.send_message(
       QueueUrl=fifo_queue['QueueUrl'],
       MessageBody=json.dumps(payment_data),
       MessageGroupId=user_id  # All messages for user processed in order
   )
   ```

**Common SQS + SNS Pattern**: Fan-out with buffering
```
SNS Topic → SQS Queue 1 → Lambda/EC2 processors
          → SQS Queue 2 → Different processors
          → SQS Queue 3 → Analytics pipeline
```

This pattern combines SNS's broadcasting with SQS's reliable delivery and buffering.

---

## Proactive Monitoring Setup

Prevent issues before they happen:

```bash
# Create comprehensive CloudWatch dashboard
aws cloudwatch put-dashboard \
  --dashboard-name ProductionHealth \
  --dashboard-body file://dashboard.json

# Set up alerts for common issues
# High error rate
aws cloudwatch put-metric-alarm \
  --alarm-name high-error-rate \
  --alarm-description "Alert when error rate exceeds 1%" \
  --metric-name 4XXError \
  --namespace AWS/ApiGateway \
  --statistic Sum \
  --period 300 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold

# Database CPU
aws cloudwatch put-metric-alarm \
  --alarm-name rds-high-cpu \
  --alarm-description "RDS CPU above 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/RDS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold
```

### Tools Every AWS Developer Should Know

1. **AWS CLI with Query Powers**
   ```bash
   # Find specific resources quickly
   aws ec2 describe-instances \
     --query 'Reservations[*].Instances[?State.Name==`running`].[InstanceId,Tags[?Key==`Name`].Value|[0]]' \
     --output table
   ```

2. **Systems Manager Session Manager**
   ```bash
   # Connect without SSH keys or bastion hosts
   aws ssm start-session --target i-xxxxx
   ```

3. **CloudWatch Logs Insights**
   ```sql
   -- Find errors across all Lambda functions
   fields @timestamp, @message
   | filter @message like /ERROR/
   | stats count() by bin(5m)
   ```

4. **AWS Personal Health Dashboard**
   - Check for AWS service issues affecting you
   - Get advance notice of maintenance

---

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Instrument Before You Need It</h4>
    <p>CloudWatch metrics, logs, and alarms turn outages into diagnosable events. Add custom metrics and set log retention so data is there — and costs stay bounded — when something breaks.</p>
  </div>
  <div class="takeaway-card">
    <h4>Decouple with SNS and SQS</h4>
    <p>SNS broadcasts to many subscribers; SQS buffers work for reliable, at-your-own-pace processing. Combine them for fan-out with durable delivery.</p>
  </div>
  <div class="takeaway-card">
    <h4>FIFO When Order Matters</h4>
    <p>Reach for SQS FIFO queues and message group IDs when per-key ordering and exactly-once processing are required, such as payments.</p>
  </div>
</div>

---

## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Infrastructure as Code](iac.html) - CloudFormation and CDK
- [Troubleshooting & Emergency Response](troubleshooting.html) - Diagnosing production issues
- [Compute Services](compute.html) - EC2, Lambda, and Auto Scaling
- [Security](security.html) - IAM and security automation
