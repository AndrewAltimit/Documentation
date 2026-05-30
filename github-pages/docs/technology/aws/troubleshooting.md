---
layout: docs
title: "AWS Troubleshooting & Emergency Response"
hide_title: true
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "server"
---

<div class="hero-section" style="background: linear-gradient(135deg, #ff9900 0%, #ffb84d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">AWS Troubleshooting &amp; Emergency Response</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">A systematic guide to diagnosing the most common AWS failures, plus an emergency playbook for the moments when production is down.</p>
</div>

Even experienced cloud architects encounter issues. This page helps you diagnose and fix common AWS problems quickly, and gives you a calm, ordered checklist to run when production is on fire.

---

## AWS Troubleshooting Guide: When Things Go Wrong

### The Troubleshooting Mindset

Before diving into specific issues, adopt this systematic approach:
1. **Check the obvious first** - Is it plugged in? (Is the service running?)
2. **Isolate the problem** - What changed recently?
3. **Use AWS tools** - CloudWatch Logs, X-Ray, Systems Manager
4. **Document everything** - Future you will thank present you

### Common Issues and Solutions

#### 1. "Access Denied" - The Most Common AWS Error

**Symptoms**:
- API calls fail with "Access Denied"
- Console shows "You don't have permissions"
- Lambda functions can't access resources

**Diagnosis Checklist**:
```bash
# Check who you are
aws sts get-caller-identity

# Check attached policies
aws iam list-attached-user-policies --user-name $(aws sts get-caller-identity --query UserId --output text)

# Test specific permissions
aws iam simulate-principal-policy \
  --policy-source-arn $(aws sts get-caller-identity --query Arn --output text) \
  --action-names s3:GetObject \
  --resource-arns arn:aws:s3:::my-bucket/*
```

**Common Fixes**:

1. **Wrong Region**
   ```bash
   # Check current region
   aws configure get region

   # Set correct region
   export AWS_DEFAULT_REGION=us-east-1
   ```

2. **Missing Resource Permissions**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": "s3:GetObject",
       "Resource": "arn:aws:s3:::my-bucket/*"
     }]
   }
   ```

3. **Service-Linked Roles**
   ```bash
   # For Lambda accessing VPC
   aws iam create-service-linked-role --aws-service-name lambda.amazonaws.com
   ```

#### 2. "Instance Connection Timeout" - Can't SSH to EC2

**Symptoms**:
- SSH hangs or times out
- Can't reach web server on instance
- Instance is running but unreachable

**Systematic Diagnosis**:

1. **Check Security Group**
   ```bash
   # List security group rules
   aws ec2 describe-security-groups --group-ids sg-xxxxxx

   # Fix: Allow SSH
   aws ec2 authorize-security-group-ingress \
     --group-id sg-xxxxxx \
     --protocol tcp \
     --port 22 \
     --cidr 0.0.0.0/0  # Use your IP for security
   ```

2. **Check Network ACLs**
   ```bash
   # Default NACLs allow all - custom ones might not
   aws ec2 describe-network-acls --filters "Name=association.subnet-id,Values=subnet-xxxxx"
   ```

3. **Check Route Table**
   ```bash
   # Ensure route to Internet Gateway exists
   aws ec2 describe-route-tables --filters "Name=association.subnet-id,Values=subnet-xxxxx"
   ```

4. **Check Instance Status**
   ```bash
   # Both checks should pass
   aws ec2 describe-instance-status --instance-id i-xxxxx
   ```

**Quick Fix Script**:
```bash
#!/bin/bash
INSTANCE_ID="i-xxxxx"
SG_ID=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)

# Allow SSH from your IP
MY_IP=$(curl -s checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr $MY_IP/32

echo "SSH access enabled from $MY_IP"
```

#### 3. "Throttling Errors" - Rate Limit Exceeded

**Symptoms**:
- "Rate exceeded" errors
- Intermittent API failures
- Bulk operations failing

**Solutions**:

1. **Implement Exponential Backoff**
   ```python
   import time
   import random
   from botocore.exceptions import ClientError

   def retry_with_backoff(func, max_retries=5):
       for attempt in range(max_retries):
           try:
               return func()
           except ClientError as e:
               if e.response['Error']['Code'] == 'Throttling':
                   # Exponential backoff with jitter
                   wait_time = (2 ** attempt) + random.uniform(0, 1)
                   time.sleep(wait_time)
               else:
                   raise
       raise Exception(f"Max retries ({max_retries}) exceeded")
   ```

2. **Use Service Quotas**
   ```bash
   # Check current limits
   aws service-quotas get-service-quota \
     --service-code ec2 \
     --quota-code L-1216C47A  # Running On-Demand instances

   # Request increase
   aws service-quotas request-service-quota-increase \
     --service-code ec2 \
     --quota-code L-1216C47A \
     --desired-value 100
   ```

#### 4. "Out of Memory" - Lambda/Container Crashes

**Symptoms**:
- Lambda function fails with no clear error
- ECS tasks stopping unexpectedly
- Application becomes unresponsive

**Diagnosis**:

1. **Check Lambda Logs**
   ```bash
   # Find memory usage
   aws logs filter-log-events \
     --log-group-name /aws/lambda/my-function \
     --filter-pattern "[REPORT]" \
     --query 'events[*].message' \
     --output text | grep "Memory"
   ```

2. **Monitor with CloudWatch**
   ```python
   # Add memory tracking to Lambda
   import resource

   def lambda_handler(event, context):
       # Track memory usage
       memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
       print(f"Memory used: {memory_usage / 1024:.2f} MB")

       # Your code here
   ```

3. **Fix: Increase Memory or Optimize**
   ```bash
   # Update Lambda memory
   aws lambda update-function-configuration \
     --function-name my-function \
     --memory-size 1024
   ```

#### 5. "Slow Application Performance"

**Symptoms**:
- API responses taking seconds
- Database queries timing out
- Users complaining about speed

**Performance Troubleshooting Toolkit**:

1. **Enable X-Ray Tracing**
   ```python
   from aws_xray_sdk.core import xray_recorder
   from aws_xray_sdk.core import patch_all

   patch_all()  # Automatically trace AWS SDK calls

   @xray_recorder.capture('process_order')
   def process_order(order_id):
       # X-Ray will show time spent in each service
       validate_order(order_id)
       charge_payment(order_id)
       update_inventory(order_id)
   ```

2. **Analyze RDS Performance**
   ```sql
   -- Enable Performance Insights
   -- Then query slow operations
   SELECT
       query,
       calls,
       total_time,
       mean_time,
       max_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

3. **Check CloudFront Cache Hit Ratio**
   ```bash
   # Low cache hit = slow performance
   aws cloudwatch get-metric-statistics \
     --namespace AWS/CloudFront \
     --metric-name CacheHitRate \
     --dimensions Name=DistributionId,Value=XXXXX \
     --statistics Average \
     --start-time 2024-01-01T00:00:00Z \
     --end-time 2024-01-02T00:00:00Z \
     --period 3600
   ```

---

## Emergency Response Playbook

When production is down, follow this checklist:

### 1. Immediate Actions (First 5 Minutes)
```bash
# Check service health
aws health describe-events --filter eventTypeCategories=issue

# Check CloudWatch alarms
aws cloudwatch describe-alarms --state-value ALARM

# Recent changes?
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=UpdateStack \
  --max-items 10
```

### 2. Common Quick Fixes

**"Everything is Down!"**
- Check Route 53 health checks
- Verify load balancer target health
- Check Auto Scaling group size

**"Database Connection Errors"**
- Check RDS security groups
- Verify connection limits not exceeded
- Check if automated backup is running

**"API Gateway 5XX Errors"**
- Check Lambda function errors
- Verify integration timeout settings
- Check concurrent execution limits

### The Four Failure Categories

Remember: Most AWS issues fall into these categories:
- **Permissions** (IAM)
- **Networking** (Security Groups, NACLs)
- **Limits** (Service Quotas)
- **Configuration** (Wrong region, missing parameters)

Master troubleshooting these four areas and you'll solve 90% of AWS problems.

---

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Observe Before You Troubleshoot</h4>
    <p>CloudWatch metrics, logs, and traces turn outages into diagnosable events. Start from data — caller identity, recent changes, alarm state — not guesses.</p>
  </div>
  <div class="takeaway-card">
    <h4>Most Failures Are One of Four Things</h4>
    <p>Permissions, networking, service limits, and configuration cover the large majority of AWS problems. Check those four first.</p>
  </div>
  <div class="takeaway-card">
    <h4>Have a Playbook Ready</h4>
    <p>When production is down is the wrong time to improvise. A short, ordered checklist for the first five minutes keeps the response calm and effective.</p>
  </div>
</div>

---

## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Monitoring & Messaging](monitoring.html) - CloudWatch dashboards and proactive alerts
- [Security](security.html) - IAM permissions and access troubleshooting
- [Networking & Content Delivery](networking.html) - Security groups, NACLs, and routing
- [Compute Services](compute.html) - EC2 and Lambda operational limits
