---
layout: docs
title: AWS Developer's Guide
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "cog"
---


<!-- Custom styles are now loaded via main.scss -->

Think of AWS as a massive technology toolkit in the cloud. Instead of buying and maintaining your own servers, you rent computing power, storage, and dozens of other services from Amazon's data centers around the world. It's like having access to an entire IT department that scales with your needs - you only pay for what you use, and you can start small and grow as big as you need.

Why does this matter? Because it transforms how we build applications. You can launch a startup from your laptop, scale to millions of users, and only pay for the resources you actually use. No more guessing how many servers you'll need or waiting weeks for hardware to arrive.

## AWS Crash Course: From Zero to Cloud in 30 Minutes

If you're completely new to AWS, this crash course will get you up and running quickly. We'll build something real while learning the essentials.

### What You'll Build
In the next 30 minutes, you'll deploy a simple web application that demonstrates core AWS concepts. By the end, you'll have:
- A virtual server running in the cloud (EC2)
- A globally accessible website (S3 + CloudFront)
- A managed database (RDS)
- Monitoring and alerts (CloudWatch)
- All connected securely

### Prerequisites
- An AWS account (sign up at aws.amazon.com - credit card required but we'll stay in free tier)
- Basic command line knowledge
- A web browser
- 30 minutes of focused time

### Step 1: Secure Your Account (5 minutes)
**Never skip this step.** AWS gives you powerful tools that can cost money if misused.

1. **Enable MFA on Root Account**
   ```
   AWS Console → Your Account → Security Credentials → MFA
   ```
   Use Google Authenticator or similar. This prevents unauthorized access even if your password leaks.

2. **Create Your First IAM User**
   ```
   IAM → Users → Add User
   Username: your-name-admin
   Access: AWS Management Console access
   Permissions: AdministratorAccess (for learning only!)
   ```

3. **Set Up Billing Alerts**
   ```
   Billing → Billing Preferences → Receive Billing Alerts
   CloudWatch → Alarms → Create Alarm
   Metric: EstimatedCharges
   Threshold: $5
   ```

**Common Pitfall**: Using root account for daily work. This is like using admin/sudo for everything - dangerous and unnecessary.

### Step 2: Launch Your First Server (10 minutes)

Let's create a virtual server (EC2 instance) - your computer in the cloud.

1. **Navigate to EC2**
   ```
   Services → EC2 → Launch Instance
   ```

2. **Configure Your Instance**
   - **Name**: MyFirstServer
   - **OS**: Amazon Linux 2023 (free tier eligible)
   - **Instance Type**: t2.micro (1 CPU, 1GB RAM - free tier)
   - **Key Pair**: Create new → Download .pem file (guard this carefully!)
   - **Network**: Allow SSH (22) and HTTP (80)

3. **Connect to Your Server**
   ```bash
   # Mac/Linux
   chmod 400 your-key.pem
   ssh -i your-key.pem ec2-user@your-instance-ip

   # Windows (use PuTTY or WSL)
   ```

4. **Install a Web Server**
   ```bash
   sudo yum update -y
   sudo yum install httpd -y
   sudo systemctl start httpd
   sudo systemctl enable httpd
   echo "<h1>Hello from AWS!</h1>" | sudo tee /var/www/html/index.html
   ```

Visit your instance's public IP in a browser - you have a website!

**What Just Happened?**
- You rented a computer in AWS's data center
- You connected to it over the internet
- You installed software just like on any Linux machine
- Your website is now globally accessible

**Common Pitfalls**:
- Forgetting to allow HTTP in security group (your firewall rules)
- Losing your key pair (no key = no access)
- Leaving instances running (they cost money outside free tier hours)

### Step 3: Create Scalable Storage (5 minutes)

EC2 storage disappears when instances terminate. Let's use S3 for permanent storage.

1. **Create an S3 Bucket**
   ```
   S3 → Create Bucket
   Name: my-first-bucket-[random-numbers]
   Region: Same as your EC2
   Block all public access: OFF (for learning only!)
   ```

2. **Upload a File**
   Create a simple HTML file locally:
   ```html
   <!DOCTYPE html>
   <html>
   <head><title>My S3 Site</title></head>
   <body>
     <h1>This website runs from S3!</h1>
     <p>No servers needed.</p>
   </body>
   </html>
   ```
   Upload via console → Make public

3. **Enable Static Website Hosting**
   ```
   Bucket → Properties → Static website hosting → Enable
   Index document: index.html
   ```

Your website now runs without any servers! Access via the S3 website endpoint.

**Why This Matters**: S3 is virtually unlimited storage that's always available. It's perfect for images, videos, backups, and static websites.

### Step 4: Add a Database (5 minutes)

Most applications need to store data. Let's add a managed database.

1. **Create an RDS Instance**
   ```
   RDS → Create Database
   Engine: MySQL
   Template: Free tier
   DB Instance: db.t3.micro
   Username: admin
   Password: [choose-strong-password]
   ```

2. **Configure Access**
   - Same VPC as your EC2
   - Security group: Allow MySQL port (3306) from your EC2 security group

3. **Connect from EC2**
   ```bash
   sudo yum install mysql -y
   mysql -h your-rds-endpoint -u admin -p
   ```

**What's Happening**: AWS manages backups, updates, and availability. You just use the database.

### Step 5: Monitor Everything (5 minutes)

You can't improve what you don't measure. CloudWatch monitors everything.

1. **View EC2 Metrics**
   ```
   EC2 → Your Instance → Monitoring Tab
   ```
   See CPU, network, disk usage

2. **Create an Alarm**
   ```
   CloudWatch → Alarms → Create Alarm
   Select Metric: EC2 → Per-Instance Metrics
   CPU Utilization > 80% for 5 minutes
   Send notification to your email
   ```

3. **View Logs**
   Install CloudWatch agent on EC2:
   ```bash
   sudo yum install amazon-cloudwatch-agent -y
   ```

**Real Impact**: You'll know about problems before your users do.

### Your First Cloud Architecture

Congratulations! You've built a real cloud architecture:
```
Internet → EC2 (Web Server) → RDS (Database)
    ↓
S3 (Static Assets) → CloudFront (CDN)
    ↓
CloudWatch (Monitoring Everything)
```

### Next Steps After the Crash Course

1. **Clean Up** (Important!):
   - Terminate EC2 instance
   - Delete RDS instance
   - Empty and delete S3 bucket
   - This prevents charges

2. **What to Learn Next**:
   - **IAM Roles**: Give EC2 permission to access S3 without keys
   - **Auto Scaling**: Automatically add/remove servers based on load
   - **Load Balancers**: Distribute traffic across multiple servers
   - **VPC**: Create private networks for security

3. **Practice Projects**:
   - Deploy a WordPress blog (EC2 + RDS)
   - Create a serverless API (Lambda + API Gateway)
   - Build a data pipeline (S3 + Lambda + DynamoDB)

### Common Beginner Mistakes to Avoid

1. **The "It Works on My Machine" Trap**
   - Always test in an AWS environment
   - Local and cloud environments differ

2. **The "Infinite Scale" Misconception**
   - Everything has limits (even in cloud)
   - Plan for growth, but start small

3. **The "Set and Forget" Danger**
   - Cloud resources need maintenance
   - Automate everything possible

4. **The "All Public" Security Hole**
   - Start with least privilege
   - Only expose what's necessary

5. **The "Bill Shock" Surprise**
   - Set up billing alerts immediately
   - Understand pricing before using services
   - Use AWS Cost Explorer weekly

This crash course gives you hands-on experience with core AWS services. You've deployed real infrastructure and learned by doing. The rest of this guide builds on these foundations with deeper knowledge and advanced patterns.

## Core Concepts to Master First

Before diving into specific services, let's understand the fundamental concepts that make cloud computing powerful:

### The Cloud Mental Model
Traditional IT requires you to predict capacity, buy hardware, and maintain everything yourself. Cloud computing flips this model - you provision resources on-demand, scale instantly, and let AWS handle the infrastructure complexity.

### Regions and Availability Zones
AWS operates in multiple geographic regions worldwide. Each region contains multiple Availability Zones (AZs) - essentially separate data centers with independent power, cooling, and networking. This geographic distribution is your foundation for building resilient applications that can survive failures.

### The Pay-as-You-Go Model
Unlike traditional IT where you pay upfront for capacity you might not use, AWS charges based on actual consumption. Launch 100 servers for an hour? You pay for 100 server-hours. This model enables experimentation and scaling without massive capital investment.

### Security as a Shared Responsibility
AWS secures the infrastructure (the "security of the cloud"), while you secure your data and applications ("security in the cloud"). Understanding this division helps you build secure systems from day one.

### Identity and Access Management (IAM)
Before creating any resources, understand IAM - it's the foundation of AWS security. IAM controls who can do what in your AWS account. Start with these principles:
- Never use your root account for daily work
- Create individual users with specific permissions
- Use roles for applications, not hardcoded credentials
- Enable MFA (Multi-Factor Authentication) everywhere

## Your First Steps with AWS

Now that you understand the core concepts, let's get practical. AWS can feel overwhelming with its 200+ services, but you don't need to learn them all at once. Here's a progressive path from beginner to advanced cloud architect.

### Getting Your Hands Dirty

Start with the AWS Free Tier - it gives you 12 months of free access to core services with generous limits. This is your playground for learning without worrying about costs.

1. **Create an AWS Account**: Set up billing alerts immediately (even on free tier)
2. **Secure Your Account**: Enable MFA on your root account and create an IAM user for daily work
3. **Launch Your First EC2 Instance**: Think of it as renting a computer in the cloud
4. **Store Files in S3**: Upload some files and understand object storage
5. **Set Up a Simple Website**: Combine EC2 and S3 to host a basic web application

Each step builds on the previous one, gradually introducing you to how AWS services work together.

## Essential Services for Every Developer

Let's explore the services you'll use most often, understanding not just what they do, but why they matter for real applications.

### Compute Services: Your Application's Brain

#### Amazon EC2 - Virtual Servers
EC2 (Elastic Compute Cloud) is like renting computers in the cloud. You choose the operating system, processing power, memory, and storage. Need a small server for testing? Launch a t2.micro. Building a data processing pipeline? Spin up a compute-optimized instance.

**Real-world example**: A startup begins with one EC2 instance running their web application. As traffic grows, they add more instances behind a load balancer. During Black Friday, they scale to 50 instances, then scale back down afterward.

**Practical EC2 Patterns**:

1. **Right-Sizing Instances**
   ```bash
   # Check current utilization
   aws cloudwatch get-metric-statistics \
     --namespace AWS/EC2 \
     --metric-name CPUUtilization \
     --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
     --statistics Average \
     --start-time 2024-01-01T00:00:00Z \
     --end-time 2024-01-02T00:00:00Z \
     --period 3600
   ```
   
   If CPU < 20% consistently, you're overpaying. Consider:
   - t3.micro → t3.nano (save 50%)
   - m5.large → t3.medium (save 60% for variable workloads)

2. **Spot Instances for Batch Processing**
   ```python
   # Example: Processing logs with 90% cost savings
   import boto3
   
   ec2 = boto3.client('ec2')
   
   # Request Spot instance
   response = ec2.request_spot_instances(
       SpotPrice='0.10',  # Max price you'll pay
       InstanceCount=1,
       LaunchSpecification={
           'ImageId': 'ami-12345678',
           'InstanceType': 'c5.large',
           'UserData': base64.b64encode('''#!/bin/bash
               aws s3 cp s3://my-logs/raw/ /tmp/logs/ --recursive
               /opt/process-logs.sh
               aws s3 cp /tmp/results/ s3://my-logs/processed/ --recursive
               shutdown -h now  # Self-terminate to save money
           '''.encode()).decode()
       }
   )
   ```

3. **Auto-Recovery for Critical Instances**
   ```bash
   # Create CloudWatch alarm for auto-recovery
   aws cloudwatch put-metric-alarm \
     --alarm-name ec2-recovery-i-1234567890abcdef0 \
     --alarm-description "Recover instance when it fails" \
     --metric-name StatusCheckFailed_System \
     --namespace AWS/EC2 \
     --statistic Maximum \
     --period 60 \
     --evaluation-periods 2 \
     --threshold 0 \
     --comparison-operator GreaterThanThreshold \
     --alarm-actions arn:aws:automate:region:ec2:recover
   ```

**Common EC2 Pitfalls and Solutions**:

1. **The "Zombie Instance" Problem**
   - **Symptom**: Instances running but not doing anything useful
   - **Solution**: Tag instances with purpose and owner, run weekly audits
   ```bash
   # Find instances without Owner tag
   aws ec2 describe-instances \
     --query 'Reservations[].Instances[?!Tags[?Key==`Owner`]].[InstanceId,State.Name]' \
     --output table
   ```

2. **The "SSH Key Lost Forever" Disaster**
   - **Symptom**: Can't access instance after losing private key
   - **Prevention**: Use Systems Manager Session Manager
   ```bash
   # Connect without SSH keys
   aws ssm start-session --target i-1234567890abcdef0
   ```

3. **The "Wrong Instance Family" Performance Issue**
   - **Symptom**: Application runs slower than expected
   - **Solution**: Match instance to workload
   - CPU-intensive → C5 (compute optimized)
   - Memory-intensive → R5 (memory optimized)  
   - Balanced → M5 (general purpose)
   - Bursty → T3 (burstable performance)

4. **The "Public IP Changed" Confusion**
   - **Symptom**: IP address changes after stop/start
   - **Solution**: Use Elastic IPs for static addresses
   ```bash
   # Allocate and associate Elastic IP
   ALLOC_ID=$(aws ec2 allocate-address --query 'AllocationId' --output text)
   aws ec2 associate-address --instance-id i-1234567890abcdef0 --allocation-id $ALLOC_ID
   ```

#### AWS Lambda - Serverless Computing
Lambda represents a paradigm shift. Instead of managing servers, you upload your code and AWS runs it in response to events. You pay only for the milliseconds your code executes.

**Real-world example**: An e-commerce site uses Lambda to resize product images. When a seller uploads a photo, Lambda automatically creates multiple sizes for different devices. No servers to manage, automatic scaling, and you only pay when images are processed.

**Practical Lambda Patterns**:

1. **Event-Driven Image Processing**
   ```python
   import json
   import boto3
   from PIL import Image
   import io
   
   s3 = boto3.client('s3')
   
   def lambda_handler(event, context):
       # Triggered by S3 upload
       bucket = event['Records'][0]['s3']['bucket']['name']
       key = event['Records'][0]['s3']['object']['key']
       
       # Download image
       response = s3.get_object(Bucket=bucket, Key=key)
       img = Image.open(io.BytesIO(response['Body'].read()))
       
       # Create thumbnails
       sizes = [(128, 128), (256, 256), (512, 512)]
       for size in sizes:
           resized = img.resize(size, Image.LANCZOS)
           
           # Save to S3
           buffer = io.BytesIO()
           resized.save(buffer, 'JPEG')
           buffer.seek(0)
           
           new_key = f"thumbnails/{size[0]}x{size[1]}/{key}"
           s3.put_object(Bucket=bucket, Key=new_key, Body=buffer.getvalue())
       
       return {'statusCode': 200, 'body': json.dumps('Thumbnails created')}
   ```

2. **API Backend Without Servers**
   ```python
   # Lambda function for REST API
   import json
   import boto3
   from decimal import Decimal
   
   dynamodb = boto3.resource('dynamodb')
   table = dynamodb.Table('Users')
   
   def lambda_handler(event, context):
       http_method = event['httpMethod']
       
       if http_method == 'GET':
           # Get user by ID
           user_id = event['pathParameters']['id']
           response = table.get_item(Key={'userId': user_id})
           
           return {
               'statusCode': 200,
               'headers': {'Content-Type': 'application/json'},
               'body': json.dumps(response.get('Item', {}), default=str)
           }
       
       elif http_method == 'POST':
           # Create new user
           body = json.loads(event['body'])
           table.put_item(Item=body)
           
           return {
               'statusCode': 201,
               'headers': {'Content-Type': 'application/json'},
               'body': json.dumps({'message': 'User created'})
           }
   ```

3. **Scheduled Data Processing**
   ```python
   # CloudWatch Events triggers this daily
   def lambda_handler(event, context):
       # Clean up old data
       s3 = boto3.client('s3')
       
       # List objects older than 30 days
       response = s3.list_objects_v2(Bucket='my-temp-bucket')
       
       for obj in response.get('Contents', []):
           if (datetime.now(timezone.utc) - obj['LastModified']).days > 30:
               s3.delete_object(Bucket='my-temp-bucket', Key=obj['Key'])
               print(f"Deleted: {obj['Key']}")
   ```

**Common Lambda Pitfalls and Solutions**:

1. **The "Cold Start" Surprise**
   - **Symptom**: First invocation takes 3-5 seconds
   - **Solutions**:
     ```python
     # 1. Provisioned Concurrency (costs money)
     # 2. Keep Lambda warm with scheduled pings
     # 3. Optimize package size
     
     # Bad: Large deployment package
     # requirements.txt
     pandas==1.5.0  # 40MB
     numpy==1.23.0  # 20MB
     
     # Good: Use Lambda Layers or lighter alternatives
     # Use AWS SDK (boto3) - pre-installed
     # Use json instead of pandas for simple data
     ```

2. **The "15-Minute Timeout" Wall**
   - **Symptom**: Function times out for long processes
   - **Solution**: Break into smaller functions or use Step Functions
   ```python
   # Step Functions state machine for long workflows
   {
     "Comment": "Process large dataset",
     "StartAt": "SplitData",
     "States": {
       "SplitData": {
         "Type": "Task",
         "Resource": "arn:aws:lambda:region:account:function:split-data",
         "Next": "ProcessInParallel"
       },
       "ProcessInParallel": {
         "Type": "Map",
         "ItemsPath": "$.chunks",
         "MaxConcurrency": 10,
         "Iterator": {
           "StartAt": "ProcessChunk",
           "States": {
             "ProcessChunk": {
               "Type": "Task",
               "Resource": "arn:aws:lambda:region:account:function:process-chunk",
               "End": true
             }
           }
         },
         "Next": "MergeResults"
       }
     }
   }
   ```

3. **The "Out of Memory" Crisis**
   - **Symptom**: Function fails with memory errors
   - **Solution**: Stream data instead of loading all at once
   ```python
   # Bad: Loading entire file
   def lambda_handler(event, context):
       s3 = boto3.client('s3')
       obj = s3.get_object(Bucket='bucket', Key='huge-file.csv')
       data = obj['Body'].read()  # Boom! Out of memory
   
   # Good: Stream processing
   def lambda_handler(event, context):
       s3 = boto3.client('s3')
       obj = s3.get_object(Bucket='bucket', Key='huge-file.csv')
       
       for line in obj['Body'].iter_lines():
           process_line(line)  # Process one line at a time
   ```

4. **The "Deployment Package Too Large" Block**
   - **Symptom**: Cannot upload function (>50MB zipped)
   - **Solution**: Use Layers and container images
   ```bash
   # Create a Lambda Layer for dependencies
   pip install -r requirements.txt -t python/
   zip -r layer.zip python/
   
   aws lambda publish-layer-version \
     --layer-name my-dependencies \
     --zip-file fileb://layer.zip \
     --compatible-runtimes python3.9
   
   # Or use container images (up to 10GB)
   FROM public.ecr.aws/lambda/python:3.9
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY app.py .
   CMD ["app.lambda_handler"]
   ```

### Storage Services: Your Application's Memory

#### Amazon S3 - Object Storage
S3 (Simple Storage Service) stores files (called objects) in buckets. Unlike traditional file systems, S3 is designed for the internet age - accessible from anywhere, virtually unlimited capacity, and extremely durable.

**Real-world example**: Netflix stores its entire video library in S3. When you stream a movie, it's delivered from S3 through CloudFront (CDN) to your device. S3's durability ensures those files won't disappear, while its scalability handles millions of concurrent viewers.

#### Amazon EBS - Block Storage
EBS provides traditional hard drives for your EC2 instances. Unlike S3, EBS acts like a normal disk drive attached to your server.

**Real-world example**: A database server uses EBS volumes for storing data files. The volumes can be backed up as snapshots, resized on demand, and moved between instances.

### Database Services: Your Application's Long-term Memory

#### Amazon RDS - Managed Relational Databases
RDS runs traditional databases (MySQL, PostgreSQL, etc.) but handles the tedious parts - backups, patching, replication. You focus on your schema and queries while AWS keeps the database running smoothly.

**Real-world example**: A SaaS application uses RDS PostgreSQL for customer data. RDS automatically backs up the database nightly, replicates to a standby instance for high availability, and can scale up during busy periods.

#### Amazon DynamoDB - NoSQL at Scale
DynamoDB is a NoSQL database designed for applications that need consistent performance at any scale. It can handle millions of requests per second with single-digit millisecond latency.

**Real-world example**: A mobile game uses DynamoDB to store player profiles and game state. Whether 100 or 10 million players are online, DynamoDB maintains consistent performance.

### Networking: Connecting Your Application

#### Amazon VPC - Your Private Cloud Network
VPC (Virtual Private Cloud) lets you create isolated networks in AWS. Think of it as your own private data center in the cloud, complete with subnets, routing rules, and security controls.

This is where things get more complex, but understanding VPC is crucial for production applications. Let's build up gradually:

1. **Basic VPC**: A simple network with public and private subnets
2. **Internet Access**: Add an Internet Gateway for public-facing resources
3. **Security Groups**: Virtual firewalls controlling traffic to your resources
4. **Multi-AZ Design**: Spread resources across Availability Zones for resilience

**Real-world example**: An enterprise application runs web servers in public subnets (accessible from internet) and databases in private subnets (only accessible from web servers). This layered security approach protects sensitive data while serving public traffic.

### Monitoring and Management: Your Cloud Operations Center

#### Amazon CloudWatch - Your Eyes in the Cloud
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

#### AWS CloudFormation - Infrastructure as Code
CloudFormation lets you define your entire infrastructure in JSON or YAML templates. Instead of clicking through the console, you describe what you want and CloudFormation builds it.

**Why CloudFormation Matters**: 
- Version control your infrastructure
- Replicate environments exactly
- Roll back changes if something breaks
- Share templates with your team

**Practical CloudFormation Example**:
```yaml
# template.yml - Complete web application stack
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Web application with auto-scaling'

Parameters:
  KeyName:
    Type: AWS::EC2::KeyPair::KeyName
    Description: EC2 Key Pair for SSH access

Resources:
  # Application Load Balancer
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Subnets: 
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref LoadBalancerSecurityGroup

  # Auto Scaling Group
  AutoScalingGroup:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      MinSize: 2
      MaxSize: 10
      DesiredCapacity: 4
      LaunchTemplate:
        LaunchTemplateId: !Ref LaunchTemplate
        Version: !GetAtt LaunchTemplate.LatestVersionNumber
      TargetGroupARNs:
        - !Ref TargetGroup
      HealthCheckType: ELB
      HealthCheckGracePeriod: 300

  # Scaling Policy
  ScaleUpPolicy:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      AutoScalingGroupName: !Ref AutoScalingGroup
      PolicyType: TargetTrackingScaling
      TargetTrackingConfiguration:
        PredefinedMetricSpecification:
          PredefinedMetricType: ASGAverageCPUUtilization
        TargetValue: 70

Outputs:
  LoadBalancerDNS:
    Description: DNS name of load balancer
    Value: !GetAtt LoadBalancer.DNSName
    Export:
      Name: !Sub ${AWS::StackName}-LoadBalancer-DNS
```

**Deploy with**:
```bash
aws cloudformation create-stack \
  --stack-name my-web-app \
  --template-body file://template.yml \
  --parameters ParameterKey=KeyName,ParameterValue=my-key
```

### Messaging and Integration: Connecting Your Services

#### Amazon SNS - Simple Notification Service
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

#### Amazon SQS - Simple Queue Service
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

### Essential Resources for Your Journey

As you progress through your AWS learning journey, these resources will accelerate your growth:

#### Learning Resources
- **[AWS Free Tier](https://aws.amazon.com/free/)**: Your risk-free playground with 12 months of free services
- **[AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)**: Learn how AWS experts design systems
- **[AWS Architecture Center](https://aws.amazon.com/architecture/)**: Real-world reference architectures and patterns
- **[AWS Training and Certification](https://aws.amazon.com/training/)**: Structured learning paths from beginner to expert

#### Developer Tools
- **[AWS SDKs](https://aws.amazon.com/tools/)**: Integrate AWS services into your applications
- **[AWS CDK](https://aws.amazon.com/cdk/)**: Define infrastructure using TypeScript, Python, or Java
- **[AWS Amplify](https://aws.amazon.com/amplify/)**: Fastest way to build full-stack applications

#### Stay Connected
- **[AWS Blog](https://aws.amazon.com/blogs/aws/)**: Daily updates on new features and best practices
- **[AWS Developer Forums](https://forums.aws.amazon.com/index.jspa)**: Get help from the community
- **[AWS re:Invent Videos](https://reinvent.awsevents.com/)**: Hundreds of free technical sessions

Pro tip: Start with the Free Tier and Well-Architected Framework. These two resources alone will accelerate your learning by months.

## Building Real Applications: Architecture Patterns

Now that you understand individual services, let's see how they work together to solve real problems. These patterns progress from simple to complex, each building on concepts from the previous ones.

### Pattern 1: Static Website Hosting (Beginner)

Let's start with the simplest cloud architecture - hosting a static website. This pattern introduces core concepts with minimal complexity.

**Components:**
- **S3**: Stores your HTML, CSS, and JavaScript files
- **CloudFront**: Delivers content globally with low latency
- **Route 53**: Manages your domain name

**Why this architecture?** It's serverless (no EC2 instances to manage), globally distributed (CloudFront edge locations), and costs pennies per month for most sites. Perfect for portfolios, documentation, or marketing sites.

**Evolution path**: Add API Gateway and Lambda for dynamic features, turning your static site into a full serverless application.

### Pattern 2: Traditional Web Application (Intermediate)

The classic three-tier architecture, modernized for the cloud. This pattern teaches you networking, security, and scaling concepts.

**Components:**
- **VPC**: Your isolated network with public/private subnets
- **EC2 + Auto Scaling**: Web servers that scale based on traffic
- **Application Load Balancer**: Distributes traffic across instances
- **RDS Multi-AZ**: Managed database with automatic failover
- **ElastiCache**: Redis/Memcached for session storage and caching

**Why this architecture?** It mirrors traditional on-premise setups but with cloud benefits - automatic scaling, managed databases, and high availability across multiple data centers.

**Real-world example**: An e-commerce platform starts with 2 EC2 instances. During sales events, Auto Scaling launches up to 20 instances. RDS handles thousands of concurrent transactions while ElastiCache reduces database load by caching product catalogs.

### Pattern 3: Serverless Microservices (Advanced)

Embrace modern cloud-native development. No servers to manage, automatic scaling, and pay-per-request pricing.

**Components:**
- **API Gateway**: RESTful API endpoint management
- **Lambda**: Individual functions for each microservice
- **DynamoDB**: NoSQL database with single-digit millisecond performance
- **Step Functions**: Orchestrate complex workflows
- **EventBridge**: Decouple services with event-driven architecture

**Why this architecture?** Each microservice scales independently, deploys separately, and costs nothing when idle. Perfect for variable workloads and rapid development.

**Real-world example**: A food delivery app uses Lambda functions for order processing, restaurant notifications, and driver assignments. DynamoDB stores order data with automatic scaling. Step Functions coordinate the entire delivery workflow. During lunch rush, the system handles 10,000 orders per minute without any manual scaling.

### Pattern 4: Data Analytics Pipeline (Advanced)

Process massive amounts of data in real-time and batch modes. This pattern introduces big data concepts and tools.

**Components:**
- **Kinesis Data Streams**: Ingest real-time data from thousands of sources
- **Kinesis Data Firehose**: Load streaming data into data stores
- **S3 Data Lake**: Central repository for all your data
- **AWS Glue**: ETL service for data preparation
- **Athena**: Query data directly in S3 using SQL
- **QuickSight**: Create dashboards and visualizations

**Why this architecture?** It separates data ingestion, storage, processing, and analysis into specialized services. Each component scales independently and you only pay for what you process.

**Real-world example**: An IoT company collects sensor data from millions of devices. Kinesis ingests 1TB per hour, Glue transforms it for analysis, and data scientists query historical data with Athena. Business users create real-time dashboards in QuickSight showing device health and usage patterns.

### Pattern 5: Container-Based Microservices (Expert)

For teams needing more control than serverless offers. Containers provide consistency across development and production.

**Components:**
- **ECS or EKS**: Container orchestration (ECS for simplicity, EKS for Kubernetes)
- **Fargate**: Serverless compute for containers
- **ECR**: Container registry for your Docker images
- **App Mesh**: Service mesh for microservice communication
- **CloudMap**: Service discovery for dynamic environments

**Why this architecture?** Containers offer portability, consistency, and fine-grained resource control. Service mesh provides advanced traffic management and observability.

**Real-world example**: A fintech platform runs 50+ microservices in EKS. Each team owns their services, deploying independently. App Mesh handles service-to-service authentication and implements canary deployments. During market hours, critical services auto-scale based on trading volume.

### Pattern 6: Multi-Region Global Application (Expert)

For applications requiring global presence, low latency, and extreme availability.

**Components:**
- **Route 53**: Geolocation and latency-based routing
- **CloudFront**: Global content delivery
- **DynamoDB Global Tables**: Multi-region replication
- **Aurora Global Database**: Cross-region read replicas
- **AWS Global Accelerator**: Improve global application availability

**Why this architecture?** Users get low latency regardless of location. The application survives entire region failures. Data replicates globally in seconds.

**Real-world example**: A social media platform serves users across continents. Route 53 directs users to the nearest region. DynamoDB Global Tables replicate user posts worldwide in under a second. If the US-East region fails, traffic automatically routes to US-West with minimal disruption.

## Real-World AWS Case Studies: Learning from Production

These detailed case studies show how real companies solved complex problems with AWS. Each includes architecture decisions, challenges faced, and lessons learned.

### Case Study 1: Netflix - Streaming at Planetary Scale

**The Challenge**: Stream video to 200+ million subscribers worldwide with perfect reliability and quality.

**Architecture Overview**:
```
Users → Route 53 → CloudFront (CDN) → Application Load Balancers
                                     ↓
                        EC2 Auto Scaling Groups (Microservices)
                                     ↓
                        DynamoDB (User Data) + S3 (Video Files)
                                     ↓
                        Kinesis (Real-time Analytics) → EMR (Big Data)
```

**Key AWS Services**:
- **EC2**: Thousands of instances running microservices
- **S3**: Stores the entire video catalog (petabytes)
- **DynamoDB**: Handles billions of reads/writes for user data
- **CloudFront**: Delivers video content globally
- **Kinesis**: Processes billions of events for recommendations

**Technical Decisions**:

1. **Chaos Engineering with Chaos Monkey**
   ```python
   # Randomly terminate instances to test resilience
   def chaos_monkey():
       if random.random() < 0.1:  # 10% chance
           instance = select_random_instance()
           terminate_instance(instance)
           log_termination(instance)
   ```

2. **Multi-Region Active-Active**
   - Every region can serve any user
   - Data replicates globally in seconds
   - Automatic failover between regions

3. **Microservices Architecture**
   - 700+ microservices
   - Each team owns their service completely
   - Deploy hundreds of times per day

**Challenges and Solutions**:

**Challenge**: Thundering herd when popular shows release
**Solution**: Pre-scaling based on ML predictions
```python
def predict_and_scale(show_id):
    predicted_viewers = ml_model.predict(show_id)
    required_capacity = calculate_capacity(predicted_viewers)
    
    # Pre-scale 30 minutes before release
    schedule_scaling(
        time=release_time - timedelta(minutes=30),
        capacity=required_capacity
    )
```

**Challenge**: Cost optimization at scale
**Solution**: Reserved Instances + Spot for batch processing
- 75% Reserved Instances for baseline
- 20% On-Demand for peaks
- 5% Spot for analytics workloads

**Lessons Learned**:
1. Design for failure - everything will fail eventually
2. Automate everything - manual processes don't scale
3. Data-driven decisions - measure everything
4. Small teams with full ownership work best

### Case Study 2: Airbnb - Global Marketplace Platform

**The Challenge**: Match millions of guests with hosts worldwide, handling payments, messaging, and trust.

**Architecture Evolution**:
```
2008: Monolithic Ruby on Rails → Single MySQL database
2012: Added caching layer → Memcached
2015: Service-oriented architecture → Multiple databases
2020: Kubernetes on AWS → Microservices
```

**Current Architecture**:
```
Mobile/Web → API Gateway → ALB → EKS (Kubernetes)
                                    ↓
            Service Mesh (Envoy) → Microservices
                                    ↓
    RDS (Transactions) + DynamoDB (Sessions) + S3 (Images)
                                    ↓
            Kinesis → Data Lake (S3) → Athena/Spark
```

**Key Technical Innovations**:

1. **Smart Pricing Algorithm**
   ```python
   # Lambda function for dynamic pricing
   def calculate_optimal_price(listing_id, date):
       factors = {
           'seasonality': get_seasonal_demand(date),
           'local_events': check_events_api(listing.location, date),
           'competitor_prices': analyze_nearby_listings(listing_id),
           'historical_booking': get_booking_patterns(listing_id)
       }
       
       base_price = listing.base_price
       optimal_price = ml_model.predict(base_price, factors)
       
       return {
           'price': optimal_price,
           'confidence': ml_model.confidence,
           'factors': factors
       }
   ```

2. **Fraud Detection System**
   - Real-time analysis with Kinesis Analytics
   - Graph database for relationship mapping
   - ML models retrained daily on EMR

3. **Image Processing Pipeline**
   ```python
   # Step Functions workflow for image processing
   {
       "ProcessListingImages": {
           "Type": "Parallel",
           "Branches": [
               {
                   "StartAt": "GenerateThumbnails",
                   "States": {
                       "GenerateThumbnails": {
                           "Type": "Task",
                           "Resource": "arn:aws:lambda:function:resize-images"
                       }
                   }
               },
               {
                   "StartAt": "DetectInappropriateContent",
                   "States": {
                       "DetectInappropriateContent": {
                           "Type": "Task",
                           "Resource": "arn:aws:lambda:function:content-moderation"
                       }
                   }
               },
               {
                   "StartAt": "ExtractMetadata",
                   "States": {
                       "ExtractMetadata": {
                           "Type": "Task",
                           "Resource": "arn:aws:lambda:function:image-analysis"
                       }
                   }
               }
           ]
       }
   }
   ```

**Scaling Challenges**:

1. **Search Performance**
   - Solution: ElasticSearch with custom ranking
   - Geographical sharding for faster queries
   - Cache warming for popular destinations

2. **Payment Processing**
   - Challenge: Handle payments in 190+ countries
   - Solution: Step Functions for complex workflows
   - SQS for reliable payment retry logic

**Key Metrics**:
- 4 million listings worldwide
- 1 billion+ searches per day
- 99.99% uptime SLA

### Case Study 3: Slack - Real-Time Messaging at Scale

**The Challenge**: Deliver messages instantly to millions of concurrent users with perfect reliability.

**Architecture Highlights**:
```
WebSocket Connections → ELB → EC2 Fleet (Connection Servers)
                                        ↓
                    Message Queue (Kafka on EC2) 
                                        ↓
        Worker Fleet (Process messages, send notifications)
                                        ↓
            DynamoDB (Message history) + S3 (File uploads)
```

**Real-Time Architecture**:

1. **WebSocket Management**
   ```python
   class ConnectionManager:
       def __init__(self):
           self.connections = {}  # In Redis
           
       async def handle_connection(self, websocket, user_id):
           # Register connection
           connection_id = str(uuid.uuid4())
           await self.register(user_id, connection_id, websocket)
           
           # Handle messages
           try:
               async for message in websocket:
                   await self.route_message(user_id, message)
           finally:
               await self.unregister(user_id, connection_id)
       
       async def broadcast_to_channel(self, channel_id, message):
           # Get all users in channel
           users = await self.get_channel_users(channel_id)
           
           # Send to all connected clients
           tasks = []
           for user_id in users:
               connections = await self.get_user_connections(user_id)
               for conn in connections:
                   tasks.append(conn.send(message))
           
           await asyncio.gather(*tasks, return_exceptions=True)
   ```

2. **Message Delivery Guarantees**
   - At-least-once delivery with idempotency
   - Message ordering per channel
   - Offline queue for disconnected users

3. **Search Infrastructure**
   - Every message indexed in near real-time
   - Elasticsearch cluster per workspace
   - Query optimization for emoji and reactions

**Scaling Milestones**:

| Year | Daily Active Users | Messages/Day | Architecture Change |
|------|-------------------|--------------|-------------------|
| 2014 | 100K | 10M | Single database |
| 2016 | 4M | 100M | Sharded MySQL |
| 2018 | 8M | 1B | DynamoDB migration |
| 2020 | 12M | 5B | Multi-region active |

**Performance Optimizations**:

1. **Connection Pooling**
   ```python
   # Efficient database connection management
   class ShardedConnectionPool:
       def __init__(self, shard_map):
           self.pools = {
               shard_id: ConnectionPool(config)
               for shard_id, config in shard_map.items()
           }
       
       def get_connection(self, workspace_id):
           shard_id = self.get_shard(workspace_id)
           return self.pools[shard_id].get_connection()
   ```

2. **Caching Strategy**
   - User presence in Redis (15-second TTL)
   - Channel membership in ElastiCache
   - Recent messages in memory

**Lessons for Real-Time Apps**:
1. Design for connection drops - mobile networks are unreliable
2. Batch operations where possible
3. Use backpressure to prevent overload
4. Monitor everything - latency matters

### Case Study 4: Robinhood - Financial Services Platform

**The Challenge**: Process millions of stock trades with zero downtime and SEC compliance.

**Regulatory Requirements**:
- Every transaction must be logged
- Data retention for 7 years
- Disaster recovery with < 1-hour RPO
- Encryption at rest and in transit

**Architecture**:
```
Mobile Apps → API Gateway → WAF → ALB
                                   ↓
            ECS Fargate (Microservices)
                                   ↓
    Aurora (Transactions) + DynamoDB (Market Data)
                                   ↓
        Kinesis Data Firehose → S3 (Compliance Archive)
                                   ↓
                    Redshift (Analytics)
```

**Critical Components**:

1. **Order Execution Engine**
   ```python
   class OrderExecutor:
       def __init__(self):
           self.market_connection = MarketConnection()
           self.risk_checker = RiskChecker()
           
       async def execute_order(self, order):
           # Pre-trade compliance checks
           compliance_result = await self.check_compliance(order)
           if not compliance_result.passed:
               return OrderResult(status='rejected', reason=compliance_result.reason)
           
           # Risk checks
           risk_result = await self.risk_checker.check(order)
           if risk_result.score > RISK_THRESHOLD:
               return OrderResult(status='rejected', reason='risk_limit')
           
           # Execute with retry logic
           for attempt in range(3):
               try:
                   result = await self.market_connection.submit(order)
                   await self.log_execution(order, result)
                   return result
               except MarketUnavailable:
                   await asyncio.sleep(0.1 * (attempt + 1))
           
           return OrderResult(status='failed', reason='market_unavailable')
   ```

2. **Real-Time Market Data Pipeline**
   - 100,000+ price updates per second
   - Sub-millisecond latency requirements
   - DynamoDB with DAX for caching

3. **Compliance and Audit System**
   - Every API call logged to Kinesis
   - Immutable audit trail in S3
   - Daily reports generated with Athena

**Scaling for Market Events**:

```python
# Auto-scaling based on market volatility
def calculate_required_capacity():
    volatility = get_market_volatility()
    normal_capacity = 100
    
    if volatility > HIGH_VOLATILITY_THRESHOLD:
        return normal_capacity * 5  # 5x during high volatility
    elif volatility > MEDIUM_VOLATILITY_THRESHOLD:
        return normal_capacity * 2
    else:
        return normal_capacity

# Pre-scale before market open
schedule.every().day.at("09:00").do(
    lambda: scale_to_capacity(calculate_required_capacity())
)
```

**Security Architecture**:
- All data encrypted with KMS
- Network isolation with PrivateLink
- API Gateway with rate limiting
- WAF rules for common attacks

### Case Study 5: Pinterest - Visual Discovery Engine

**The Challenge**: Serve billions of images with personalized recommendations to 400+ million users.

**Data Scale**:
- 300+ billion Pins
- 5 billion boards
- 600 million searches per month
- 2 billion recommendations per day

**Architecture**:
```
CDN (CloudFront) → Image Servers (EC2 + S3)
        ↓
API Gateway → Service Mesh → Microservices (EKS)
        ↓
Graph Database (Neptune) + Feature Store (DynamoDB)
        ↓
ML Pipeline (SageMaker) → Recommendation Service
```

**Key Innovations**:

1. **Visual Search System**
   ```python
   class VisualSearchEngine:
       def __init__(self):
           self.feature_extractor = load_model('resnet50')
           self.index = FaissIndex()  # Billion-scale similarity search
           
       def process_image(self, image_url):
           # Extract visual features
           image = download_image(image_url)
           features = self.feature_extractor.extract(image)
           
           # Store in feature database
           image_id = generate_id(image_url)
           self.store_features(image_id, features)
           
           # Find similar images
           similar = self.index.search(features, k=100)
           return self.rank_results(similar)
       
       def build_index_shard(self, shard_id):
           # Build index for billions of images
           features = self.load_features_for_shard(shard_id)
           index = FaissIndex()
           
           # Add in batches for efficiency
           for batch in chunks(features, 10000):
               index.add_batch(batch)
           
           # Save to S3
           index.save_to_s3(f"index/shard_{shard_id}")
   ```

2. **Personalization Pipeline**
   - User signals processed in real-time
   - Graph neural networks for recommendations
   - A/B testing framework for algorithms

3. **Content Moderation**
   - ML models detect inappropriate content
   - Human review queue with SQS
   - Feedback loop to improve models

**Performance Optimizations**:
- Image serving through CloudFront
- Aggressive caching at every layer
- Progressive image loading
- WebP format for modern browsers

**Lessons Learned**:
1. **Cache Everything**: 99% cache hit rate saves millions
2. **Precompute When Possible**: Recommendations generated offline
3. **Shard by User**: Better cache locality
4. **Monitor User Experience**: Not just system metrics

### Key Takeaways from All Case Studies

1. **Start Simple, Evolve Gradually**
   - Every company started with basic architecture
   - Complexity added only when needed
   - Technical debt managed actively

2. **Data is Everything**
   - Instrument everything from day one
   - Use data to drive decisions
   - Build data pipelines early

3. **Failure is Normal**
   - Design for failure at every level
   - Practice failure scenarios
   - Automate recovery procedures

4. **Scale Horizontally**
   - Vertical scaling hits limits quickly
   - Design for distributed systems
   - Embrace eventual consistency

5. **Security Cannot Be an Afterthought**
   - Build security into architecture
   - Automate security scanning
   - Regular security audits

These case studies demonstrate that successful AWS architectures share common patterns: they start simple, measure everything, automate aggressively, and evolve based on real needs rather than predicted ones.

## Progressive Learning Path: From Zero to Cloud Architect

AWS offers 200+ services, but you don't need to learn them all. Here's a curated learning path that builds your skills progressively. Each tier assumes mastery of the previous one.

### Tier 1: Foundation Services (Weeks 1-4)
Master these first. Every AWS architect uses these daily.

**Compute**
- **EC2**: Virtual servers - learn instance types, AMIs, and basic networking
- **Lambda**: Serverless functions - start with simple Python/Node.js functions

**Storage**
- **S3**: Object storage - understand buckets, objects, and basic permissions
- **EBS**: Block storage for EC2 - learn volume types and snapshots

**Networking**
- **VPC**: Virtual networks - grasp subnets, routing, and security groups
- **CloudFront**: CDN basics - cache static content from S3

**Security**
- **IAM**: Identity management - users, roles, and policies

### Tier 2: Production Essentials (Weeks 5-8)
Services that make applications production-ready.

**Databases**
- **RDS**: Managed SQL databases - focus on MySQL or PostgreSQL
- **DynamoDB**: NoSQL basics - understand partition keys and queries

**Application Integration**
- **SQS**: Message queuing - decouple application components
- **SNS**: Notifications - email and SMS alerts

**Monitoring**
- **CloudWatch**: Metrics, logs, and alarms
- **X-Ray**: Distributed tracing basics

**Developer Tools**
- **CodeDeploy**: Automated deployments
- **Systems Manager**: Parameter Store for configuration

### Tier 3: Scaling and Resilience (Weeks 9-12)
Build applications that scale and survive failures.

**Advanced Compute**
- **Auto Scaling**: Dynamic capacity management
- **ECS**: Container orchestration basics
- **Elastic Load Balancing**: ALB for HTTP/HTTPS traffic

**Advanced Storage**
- **EFS**: Shared file storage across EC2 instances
- **S3 Lifecycle Policies**: Automated data archiving

**Advanced Networking**
- **Route 53**: DNS management and health checks
- **API Gateway**: RESTful API development

**Data Processing**
- **Kinesis**: Real-time data streaming
- **Athena**: Query data in S3 with SQL

### Tier 4: Enterprise Patterns (Months 4-6)
Services for complex, enterprise-grade applications.

**Container Orchestration**
- **EKS**: Kubernetes on AWS
- **Fargate**: Serverless containers
- **ECR**: Container registry

**Advanced Data**
- **Redshift**: Data warehousing
- **EMR**: Big data processing with Spark/Hadoop
- **Glue**: ETL and data cataloging

**Machine Learning**
- **SageMaker**: ML model training and deployment
- **Rekognition**: Image and video analysis

**Enterprise Integration**
- **Step Functions**: Complex workflow orchestration
- **EventBridge**: Event-driven architectures
- **AppSync**: Managed GraphQL

### Tier 5: Specialized Services (Months 6+)
Domain-specific services for specialized use cases.

**IoT and Edge**
- **IoT Core**: Device connectivity and management
- **Greengrass**: Edge computing

**Media Services**
- **Elemental MediaConvert**: Video transcoding
- **Kinesis Video Streams**: Video ingestion

**Quantum Computing**
- **Braket**: Quantum computing experiments

**Game Development**
- **GameLift**: Game server hosting
- **Lumberyard**: Game engine integration

## Complete Service Reference

Here's a comprehensive reference of AWS services organized by category. Use this as a lookup guide as you encounter services in architectures and documentation.

### Compute Services

**Virtual Servers and Containers**
- **EC2 (Elastic Compute Cloud)**: Rent virtual servers with full control over the operating system
- **ECS (Elastic Container Service)**: Run Docker containers with AWS-native orchestration
- **EKS (Elastic Kubernetes Service)**: Managed Kubernetes for teams already using K8s
- **Fargate**: Run containers without managing servers (serverless containers)

**Serverless Compute**
- **Lambda**: Run code in response to events, pay only for compute time used
- **Batch**: Run batch computing workloads at any scale
- **Lightsail**: Simple virtual private servers for basic workloads

### Storage Services

**Object Storage**
- **S3 (Simple Storage Service)**: Store and retrieve any amount of data from anywhere
- **S3 Glacier**: Long-term archive storage at extremely low cost

**Block Storage**
- **EBS (Elastic Block Store)**: Persistent block storage for EC2 instances
- **Instance Store**: Temporary block storage physically attached to EC2 host

**File Storage**
- **EFS (Elastic File System)**: Managed NFS for EC2 instances
- **FSx**: Fully managed Windows file servers and high-performance computing file systems

**Hybrid Storage**
- **Storage Gateway**: Connect on-premises applications to AWS storage
- **AWS Backup**: Centralized backup across AWS services

### Database Services

**Relational Databases**
- **RDS (Relational Database Service)**: Managed MySQL, PostgreSQL, Oracle, SQL Server, MariaDB
- **Aurora**: AWS-built MySQL/PostgreSQL compatible database with 5x performance
- **Aurora Serverless**: Auto-scaling version of Aurora for variable workloads

**NoSQL Databases**
- **DynamoDB**: Fast, flexible NoSQL database with single-digit millisecond latency
- **DocumentDB**: MongoDB-compatible document database
- **Keyspaces**: Managed Apache Cassandra-compatible database

**In-Memory Databases**
- **ElastiCache**: Managed Redis and Memcached for microsecond latency
- **MemoryDB for Redis**: Redis-compatible, durable in-memory database

**Specialty Databases**
- **Neptune**: Graph database for highly connected datasets
- **Timestream**: Time series database for IoT and operational applications
- **QLDB**: Ledger database with immutable transaction log

### Networking and Content Delivery

**Core Networking**
- **VPC (Virtual Private Cloud)**: Isolated cloud resources in a virtual network
- **Route 53**: Scalable DNS and domain registration
- **CloudFront**: Global content delivery network (CDN)

**Network Connectivity**
- **Direct Connect**: Dedicated network connection to AWS
- **VPN**: Secure connections between on-premises networks and VPC
- **Transit Gateway**: Connect VPCs and on-premises networks through a central hub

**Load Balancing**
- **Elastic Load Balancing**: Distribute traffic across multiple targets
- **Application Load Balancer**: HTTP/HTTPS traffic with advanced routing
- **Network Load Balancer**: Ultra-high performance for TCP/UDP traffic

### Security, Identity, and Compliance

**Identity and Access**
- **IAM (Identity and Access Management)**: Control access to AWS resources
- **Cognito**: User authentication for mobile and web apps
- **SSO**: Centralized access to multiple AWS accounts and applications

**Detection and Response**
- **GuardDuty**: Threat detection using machine learning
- **Security Hub**: Unified view of security alerts and compliance status
- **Detective**: Analyze and visualize security data to investigate incidents

**Data Protection**
- **KMS (Key Management Service)**: Create and control encryption keys
- **Secrets Manager**: Rotate, manage, and retrieve secrets
- **Certificate Manager**: Provision and manage SSL/TLS certificates

### Application Integration

**Messaging**
- **SQS (Simple Queue Service)**: Managed message queuing service
- **SNS (Simple Notification Service)**: Pub/sub messaging and mobile notifications
- **EventBridge**: Serverless event bus connecting applications

**Workflow Orchestration**
- **Step Functions**: Coordinate distributed applications using visual workflows
- **SWF (Simple Workflow)**: Build scalable, resilient applications (legacy, use Step Functions)

**API Management**
- **API Gateway**: Create, publish, and manage APIs at any scale
- **AppSync**: Managed GraphQL service with real-time data sync

### Analytics Services

**Data Streaming and Processing**
- **Kinesis Data Streams**: Real-time data streaming at scale
- **Kinesis Data Firehose**: Load streaming data into data stores
- **Kinesis Data Analytics**: Process streaming data with SQL or Apache Flink

**Big Data Processing**
- **EMR (Elastic MapReduce)**: Managed Hadoop, Spark, HBase, and Presto
- **Glue**: Serverless ETL service for data preparation
- **Data Pipeline**: Orchestrate data movement and transformation

**Data Warehousing and Query**
- **Redshift**: Petabyte-scale data warehouse
- **Athena**: Query data in S3 using standard SQL
- **Lake Formation**: Build secure data lakes in days instead of months

**Business Intelligence**
- **QuickSight**: Scalable business intelligence service
- **DataZone**: Discover, share, and govern data at scale

### Machine Learning and AI

**ML Platform**
- **SageMaker**: Build, train, and deploy machine learning models
- **SageMaker Studio**: Integrated development environment for ML
- **SageMaker Feature Store**: Store and share ML features

**AI Services (Pre-trained Models)**
- **Rekognition**: Image and video analysis
- **Textract**: Extract text and data from documents
- **Comprehend**: Natural language processing
- **Translate**: Neural machine translation
- **Polly**: Text-to-speech
- **Transcribe**: Speech-to-text
- **Lex**: Build conversational interfaces
- **Personalize**: Real-time personalization and recommendations
- **Forecast**: Time-series forecasting

### Developer Tools

**CI/CD Pipeline**
- **CodeCommit**: Git-based source control
- **CodeBuild**: Compile and test code
- **CodeDeploy**: Automated code deployment
- **CodePipeline**: Continuous delivery service
- **CodeStar**: Unified interface for DevOps

**Development Productivity**
- **Cloud9**: Cloud-based IDE
- **CloudShell**: Browser-based shell with AWS CLI
- **CodeGuru**: ML-powered code reviews
- **CodeWhisperer**: AI coding companion

### Management and Governance

**Monitoring and Observability**
- **CloudWatch**: Metrics, logs, and alarms
- **X-Ray**: Distributed application tracing
- **CloudTrail**: API activity logging
- **Systems Manager**: Operational insights and actions

**Resource Management**
- **CloudFormation**: Infrastructure as code
- **Service Catalog**: Create and manage approved products
- **Control Tower**: Set up multi-account environments
- **Config**: Track resource configurations

**Cost Management**
- **Cost Explorer**: Visualize and manage costs
- **Budgets**: Set custom cost and usage budgets
- **Trusted Advisor**: Best practice recommendations

### Migration and Transfer Services

**Database Migration**
- **Database Migration Service (DMS)**: Migrate databases with minimal downtime
- **Schema Conversion Tool**: Convert database schemas between engines

**Data Transfer**
- **DataSync**: Transfer data between on-premises and AWS
- **Transfer Family**: Managed file transfers (SFTP, FTPS, FTP)
- **Snow Family**: Physical devices for petabyte-scale data transfer

**Application Migration**
- **Application Migration Service**: Lift-and-shift applications to AWS
- **Application Discovery Service**: Plan migration projects

## Advanced Cloud Architecture: Building at Scale

Once you've mastered the basics, these advanced patterns help you build enterprise-grade systems. Each pattern solves specific challenges that emerge as applications grow.

### Why Multi-Account Architecture?

As your AWS usage grows, managing everything in a single account becomes risky and complex. Imagine running development experiments in the same account as production customer data - one mistake could be catastrophic. Multi-account architecture solves this by creating boundaries.

#### Understanding AWS Organizations

Think of AWS Organizations as a company structure for your cloud resources. Just as companies have departments (Engineering, Finance, HR), you create separate AWS accounts for different purposes:

**Account Structure That Makes Sense:**
- **Production Account**: Customer-facing applications only
- **Development Account**: Safe playground for experiments
- **Security Account**: Centralized logging and compliance tools
- **Shared Services Account**: Common resources like CI/CD pipelines

**Real-world example**: A startup begins with one AWS account. After a developer accidentally deletes production data while testing, they implement Organizations. Now developers can't even access production resources without explicit permission. The security team monitors all accounts from a central location.

**Service Control Policies (SCPs)**: These are like company-wide rules. For example, "No one can launch EC2 instances outside approved regions" or "All S3 buckets must be encrypted." SCPs enforce these rules across all accounts, preventing costly mistakes.

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

### Event-Driven Architecture: Building Reactive Systems

Traditional applications work like phone calls - one service directly calls another and waits for a response. Event-driven architecture works like text messages - services broadcast events and interested parties respond when ready. This decoupling transforms how we build scalable systems.

#### Why Events Matter

Imagine an e-commerce order: payment processing, inventory updates, shipping notifications, and analytics all need to happen. In traditional architecture, your order service would call each system sequentially. If shipping is slow, the entire order process slows down. With events, the order service simply announces "Order Placed!" and each system reacts independently.

#### EventBridge: Your Event Router

EventBridge acts as the central nervous system of your application. Services publish events without knowing who consumes them. New features can listen to existing events without changing the publishers.

**Pattern in Action: Order Processing**

1. **Order Service** publishes "OrderCreated" event
2. **Payment Service** processes payment and publishes "PaymentCompleted"
3. **Inventory Service** reserves items and publishes "ItemsReserved"
4. **Shipping Service** creates labels when all prerequisites are met
5. **Analytics Service** updates dashboards with each event

**Resilience Built-In:**
- **Dead Letter Queues**: Failed events aren't lost, they're queued for investigation
- **Event Replay**: Replay historical events to recover from failures or test new features
- **Error Notifications**: Get alerted when event processing fails

**Real-world example**: A food delivery platform uses EventBridge to coordinate restaurants, drivers, and customers. When an order is placed, events trigger kitchen notifications, driver assignments, and customer updates. Each service scales independently - during lunch rush, driver assignment might process 1000 events/second while payment processing handles 100/second.

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

## Performance at Scale: Making Applications Fast

Performance isn't just about speed - it's about delivering consistent experiences whether you have 10 or 10 million users. AWS provides tools to optimize every layer of your application.

### The Performance Journey

Most applications follow a predictable performance evolution:

1. **Single Region, Basic Setup**: Works fine for hundreds of users
2. **Caching Added**: Handles thousands without breaking a sweat
3. **Multi-Region Deployment**: Serves millions with low latency globally
4. **Edge Optimization**: Delivers content in milliseconds worldwide

Let's explore each optimization technique and when to apply it.

### CloudFront: Your Global Accelerator

CloudFront is AWS's content delivery network (CDN). Instead of users fetching data from your servers in Virginia, CloudFront caches content at 400+ edge locations worldwide. Users get data from the nearest location, reducing latency from seconds to milliseconds.

**When to Use CloudFront:**
- Static assets (images, CSS, JavaScript) - immediate 10x performance boost
- API responses that don't change frequently
- Video streaming - adaptive bitrate based on user connection
- Global applications - consistent performance worldwide

**Real-world impact**: A news website serving images from S3 in US-East to users in Australia saw 2-second load times. After adding CloudFront, Australian users get 200ms load times from the Sydney edge location.

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

## Cost Optimization: Spending Smart in the Cloud

The cloud's pay-as-you-go model is powerful, but without proper management, costs can spiral. The key is understanding how pricing works and implementing automated controls from the start.

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

## The Future of Cloud: Emerging Technologies

AWS continuously launches new services that push the boundaries of what's possible. While you don't need these for most applications, understanding emerging technologies helps you prepare for the future.

### Quantum Computing: From Science Fiction to Reality

AWS Braket makes quantum computing accessible to developers. Instead of building a quantum computer (which requires near-absolute-zero temperatures), you can run quantum algorithms on actual quantum hardware through AWS.

#### When Quantum Computing Matters

Quantum computers excel at specific problems:
- **Optimization**: Finding the best route among millions of possibilities
- **Simulation**: Modeling molecular interactions for drug discovery
- **Cryptography**: Breaking and creating unbreakable codes
- **Machine Learning**: Training models on complex patterns

**Real-world example**: A logistics company uses Braket to optimize delivery routes across 1,000 cities. Classical computers would take years to find the optimal solution; quantum algorithms explore multiple possibilities simultaneously, finding near-optimal routes in hours.

**Getting Started with Quantum**:
1. Start with quantum simulators (free) to learn quantum programming
2. Test algorithms on actual quantum hardware (pay per task)
3. Compare results with classical computing to understand advantages

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

### Machine Learning: AI for Every Developer

Machine learning used to require PhD-level expertise and massive infrastructure. AWS SageMaker democratizes ML, letting any developer build, train, and deploy models.

#### The ML Journey Simplified

1. **Data Preparation**: Clean and organize your training data
2. **Model Selection**: Choose from pre-built models or create custom ones
3. **Training**: SageMaker handles the compute infrastructure
4. **Deployment**: One-click deployment with automatic scaling
5. **Monitoring**: Track model performance and retrain as needed

**From Weeks to Hours**: What traditionally took weeks of infrastructure setup now takes hours. SageMaker provides Jupyter notebooks for experimentation, distributed training for large datasets, and managed endpoints for serving predictions.

**Real-world example**: An e-commerce company builds a recommendation engine:
- Upload purchase history to S3
- Use SageMaker's built-in recommendation algorithm
- Train on historical data (SageMaker provisions GPU instances automatically)
- Deploy model to an endpoint
- Call the endpoint from their app to get real-time recommendations

Total time: 2 days instead of 2 months.

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

## Infrastructure as Code: Never Click Again

The biggest shift in cloud operations? Treating infrastructure like software. Instead of clicking through the AWS console, you define infrastructure in code. This enables version control, peer review, and automated deployments.

### Why Infrastructure as Code Changes Everything

**The Old Way**: 
- Click through AWS console to create resources
- Document steps in a wiki (that nobody updates)
- Hope you can recreate it in another region
- Fear making changes that might break production

**The IaC Way**:
- Define infrastructure in configuration files
- Version control shows exactly what changed and when
- Deploy identical environments with one command
- Test changes in staging before production

### Choosing Your IaC Tool

#### CloudFormation (AWS Native)
- **Pros**: Deep AWS integration, no extra tools needed
- **Cons**: Verbose syntax, AWS-only
- **Best for**: Teams fully committed to AWS

#### Terraform (Multi-Cloud)
- **Pros**: Works across cloud providers, huge community
- **Cons**: Requires learning HCL syntax
- **Best for**: Multi-cloud strategies or teams wanting flexibility

#### AWS CDK (Developer-Friendly)
- **Pros**: Use familiar programming languages (Python, TypeScript)
- **Cons**: Newer tool, smaller community
- **Best for**: Development teams wanting to use existing skills

### Real-World IaC Evolution

A startup's infrastructure journey:

1. **Month 1**: Everything created via console clicks
2. **Month 3**: Production breaks, nobody remembers how to rebuild
3. **Month 4**: Team adopts Terraform, documents existing infrastructure
4. **Month 6**: All changes go through pull requests
5. **Year 1**: Disaster recovery test - entire production rebuilt in 30 minutes

### Advanced Patterns That Save Your Sanity

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

## AWS Troubleshooting Guide: When Things Go Wrong

Even experienced cloud architects encounter issues. This guide helps you diagnose and fix common AWS problems quickly.

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
       "Resource": "arn:aws:s3:::my-bucket/*"  // Don't forget the /*
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

### Emergency Response Playbook

When production is down, follow this checklist:

#### 1. Immediate Actions (First 5 Minutes)
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

#### 2. Common Quick Fixes

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

### Proactive Monitoring Setup

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

Remember: Most AWS issues fall into these categories:
- **Permissions** (IAM)
- **Networking** (Security Groups, NACLs)
- **Limits** (Service Quotas)
- **Configuration** (Wrong region, missing parameters)

Master troubleshooting these four areas and you'll solve 90% of AWS problems.

## Your Cloud Journey: From Here to Mastery

You've learned the core concepts, explored architecture patterns, and understand optimization strategies. Where do you go from here?

### The Path Forward

#### Next 30 Days: Build Your Foundation
1. **Get Hands-On**: Launch your first EC2 instance, create an S3 bucket, set up a simple website
2. **Break Things**: Experiment in a sandbox account - failure is the best teacher
3. **Automate One Thing**: Convert a manual process to Lambda or use CloudFormation
4. **Monitor Costs**: Set up billing alerts and understand your first bill

#### Next 90 Days: Develop Expertise
1. **Build a Real Project**: Create something you'll actually use
2. **Master One Service Deeply**: Whether it's Lambda, DynamoDB, or ECS
3. **Practice Troubleshooting**: Learn to read CloudWatch logs and traces
4. **Join the Community**: AWS user groups, re:Invent videos, forums

#### Next Year: Achieve Mastery
1. **Design for Scale**: Build systems that can grow 100x
2. **Optimize Everything**: Cost, performance, security, operations
3. **Share Knowledge**: Blog, speak, mentor others
4. **Stay Current**: AWS releases new features daily - follow what matters to you

### Emerging Trends to Watch

**Serverless Everything**: The trend toward managed services accelerates. Focus on business logic, not infrastructure.

**AI-Powered Operations**: From cost optimization to security, AI will automate routine cloud management tasks.

**Edge Computing**: Processing moves closer to users. 5G and IoT drive computing to the edge.

**Sustainability Focus**: Carbon-aware computing becomes standard. Green architectures will be the default.

### Remember: Cloud is a Journey, Not a Destination

AWS evolves constantly. The services you master today will have new features tomorrow. The architectures you build will need to adapt. That's not a bug - it's the feature that makes cloud computing exciting.

Start small, think big, and build amazing things. The cloud is your platform for innovation. What will you create?

